"""FastAPI search service: text -> products, image -> similar products, plus a web page at "/".

Environment variables:
    ARTIFACT_DIR       folder with index.faiss, meta.parquet (default: artifacts)
    MODEL_PATH         fine-tuned checkpoint (default: $ARTIFACT_DIR/model.pt if present)
    MODEL_NAME         OpenCLIP architecture (default: ViT-B-32)
    PRETRAINED         OpenCLIP weights tag used to build the model (default: laion2b_s34b_b79k)
    ARTIFACTS_S3_URI   if set, download artifacts from s3://bucket/prefix/ at startup
    IMAGE_BASE_URL     where product photos are served from (default: the public ABO bucket)
    LOCAL_IMAGE_DIR    optional: serve photos from this folder at /images (local testing)
"""
import io
import os
import re
import sys
import time
import urllib.request
from contextlib import asynccontextmanager
from functools import lru_cache

import faiss
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import load_model  # noqa: E402

STATE = {}
HERE = os.path.dirname(os.path.abspath(__file__))
ABO_IMAGES = "https://amazon-berkeley-objects.s3.amazonaws.com/images/small/"
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "/images/" if os.getenv("LOCAL_IMAGE_DIR") else ABO_IMAGES)
SAFE_IMAGE_PATH = re.compile(r"^[0-9a-zA-Z_\-]{1,8}/[0-9a-zA-Z_\-]{1,64}\.(jpg|jpeg|png)$")
torch.set_num_threads(int(os.getenv("TORCH_THREADS", "2")))


def download_from_s3(uri, dest):
    import boto3
    bucket, _, prefix = uri.replace("s3://", "").partition("/")
    s3 = boto3.client("s3")
    os.makedirs(dest, exist_ok=True)
    for fn in ["index.faiss", "meta.parquet", "index_info.json", "model.pt"]:
        try:
            s3.download_file(bucket, prefix.rstrip("/") + "/" + fn, os.path.join(dest, fn))
        except Exception as e:  # model.pt and index_info.json are optional
            print(f"skip {fn}: {e}")


@asynccontextmanager
async def lifespan(app):
    art = os.getenv("ARTIFACT_DIR", "artifacts")
    if os.getenv("ARTIFACTS_S3_URI"):
        download_from_s3(os.environ["ARTIFACTS_S3_URI"], art)
    ckpt = os.getenv("MODEL_PATH") or (os.path.join(art, "model.pt") if os.path.exists(os.path.join(art, "model.pt")) else None)
    model, _, pre_val, tok = load_model(os.getenv("MODEL_NAME", "ViT-B-32"), os.getenv("PRETRAINED", "laion2b_s34b_b79k"), ckpt)
    model.eval()
    meta = pd.read_parquet(os.path.join(art, "meta.parquet"))
    STATE.update(model=model, preprocess=pre_val, tokenizer=tok,
                 index=faiss.read_index(os.path.join(art, "index.faiss")), meta=meta,
                 path_to_row={p: i for i, p in enumerate(meta.image_path)})
    print(f"loaded {STATE['index'].ntotal:,} items, checkpoint={ckpt}")
    yield
    STATE.clear()


app = FastAPI(title="Multimodal Product Search", lifespan=lifespan)

if os.getenv("LOCAL_IMAGE_DIR"):
    from fastapi.staticfiles import StaticFiles
    app.mount("/images", StaticFiles(directory=os.environ["LOCAL_IMAGE_DIR"]), name="images")


def describe(text):
    """Listing text is 'Name. product type. color X. brand Y. bullet' -> readable fields."""
    parts = [p.strip() for p in text.split(". ") if p.strip()]
    name = parts[0] if parts else text
    brand = next((p[6:] for p in parts if p.startswith("brand ")), "")
    color = next((p[6:] for p in parts if p.startswith("color ")), "")
    category = parts[1] if len(parts) > 1 and not parts[1].startswith(("brand ", "color ")) else ""
    return name[:140], category[:40], brand[:40], color[:30]


def search_vector(vec, k, exclude_row=None):
    scores, ids = STATE["index"].search(vec.numpy().astype("float32"), k + (1 if exclude_row is not None else 0))
    meta, out = STATE["meta"], []
    for s, i in zip(scores[0], ids[0]):
        if i < 0 or i == exclude_row:
            continue
        name, category, brand, color = describe(meta.text.iat[i])
        path = meta.image_path.iat[i]
        out.append({"item_id": meta.item_id.iat[i], "score": round(float(s), 4), "title": meta.text.iat[i][:120],
                    "name": name, "category": category, "brand": brand, "color": color,
                    "image_path": path, "image_url": IMAGE_BASE_URL + path})
    return out[:k]


def encode_image(img):
    with torch.no_grad():
        return F.normalize(STATE["model"].encode_image(STATE["preprocess"](img).unsqueeze(0)).float(), dim=-1)


@lru_cache(maxsize=512)
def fetch_catalog_image(image_path):
    if os.getenv("LOCAL_IMAGE_DIR"):
        return Image.open(os.path.join(os.environ["LOCAL_IMAGE_DIR"], image_path)).convert("RGB")
    with urllib.request.urlopen(ABO_IMAGES + image_path, timeout=10) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(os.path.join(HERE, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "items": int(STATE["index"].ntotal)}


@app.get("/search")
def search(q: str = Query(..., min_length=1, max_length=200), k: int = Query(10, ge=1, le=100)):
    t0 = time.perf_counter()
    with torch.no_grad():
        vec = F.normalize(STATE["model"].encode_text(STATE["tokenizer"]([q])).float(), dim=-1)
    results = search_vector(vec, k)
    return {"query": q, "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "items_searched": int(STATE["index"].ntotal), "results": results}


@app.post("/search_by_image")
async def search_by_image(file: UploadFile = File(...), k: int = Query(10, ge=1, le=100)):
    data = await file.read()
    if len(data) > 10 * 2**20:
        raise HTTPException(413, "Image is larger than 10 MB. Try a smaller photo.")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read that file as an image. Use a JPG or PNG.")
    t0 = time.perf_counter()
    results = search_vector(encode_image(img), k)
    return {"latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "items_searched": int(STATE["index"].ntotal), "results": results}


@app.get("/similar")
def similar(image_path: str = Query(...), k: int = Query(10, ge=1, le=100)):
    """'More like this': embed a catalog product's own photo and search with it."""
    row = STATE["path_to_row"].get(image_path)
    if row is None or not SAFE_IMAGE_PATH.match(image_path):
        raise HTTPException(404, "That product is not in the catalog.")
    try:
        img = fetch_catalog_image(image_path)
    except Exception:
        raise HTTPException(502, "Could not load the product photo. Try again in a moment.")
    t0 = time.perf_counter()
    results = search_vector(encode_image(img), k, exclude_row=row)
    name, *_ = describe(STATE["meta"].text.iat[row])
    return {"source": {"name": name, "image_url": IMAGE_BASE_URL + image_path},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "items_searched": int(STATE["index"].ntotal), "results": results}
