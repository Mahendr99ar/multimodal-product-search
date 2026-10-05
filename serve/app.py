"""FastAPI search service: text -> products, and image -> similar products.

Environment variables:
    ARTIFACT_DIR       folder with index.faiss, meta.parquet (default: artifacts)
    MODEL_PATH         fine-tuned checkpoint (default: $ARTIFACT_DIR/model.pt if present)
    MODEL_NAME         OpenCLIP architecture (default: ViT-B-32)
    PRETRAINED         OpenCLIP weights tag used to build the model (default: laion2b_s34b_b79k)
    ARTIFACTS_S3_URI   if set, download artifacts from s3://bucket/prefix/ at startup
"""
import io
import os
import sys
import time
from contextlib import asynccontextmanager

import faiss
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import load_model  # noqa: E402

STATE = {}
torch.set_num_threads(int(os.getenv("TORCH_THREADS", "2")))


def download_from_s3(uri, dest):
    import boto3
    bucket, _, prefix = uri.replace("s3://", "").partition("/")
    s3 = boto3.client("s3")
    os.makedirs(dest, exist_ok=True)
    for fn in ["index.faiss", "meta.parquet", "index_info.json", "model.pt"]:
        try:
            s3.download_file(bucket, prefix.rstrip("/") + "/" + fn, os.path.join(dest, fn))
        except Exception as e:  # model.pt is optional (zero-shot serving)
            print(f"skip {fn}: {e}")


@asynccontextmanager
async def lifespan(app):
    art = os.getenv("ARTIFACT_DIR", "artifacts")
    if os.getenv("ARTIFACTS_S3_URI"):
        download_from_s3(os.environ["ARTIFACTS_S3_URI"], art)
    ckpt = os.getenv("MODEL_PATH") or (os.path.join(art, "model.pt") if os.path.exists(os.path.join(art, "model.pt")) else None)
    model, _, pre_val, tok = load_model(os.getenv("MODEL_NAME", "ViT-B-32"), os.getenv("PRETRAINED", "laion2b_s34b_b79k"), ckpt)
    model.eval()
    STATE.update(model=model, preprocess=pre_val, tokenizer=tok,
                 index=faiss.read_index(os.path.join(art, "index.faiss")),
                 meta=pd.read_parquet(os.path.join(art, "meta.parquet")))
    print(f"loaded {STATE['index'].ntotal:,} items, checkpoint={ckpt}")
    yield
    STATE.clear()


app = FastAPI(title="Multimodal Product Search", lifespan=lifespan)


def search_vector(vec, k):
    scores, ids = STATE["index"].search(vec.numpy().astype("float32"), k)
    meta = STATE["meta"]
    return [{"item_id": meta.item_id.iat[i], "score": round(float(s), 4), "title": meta.text.iat[i][:120],
             "image_path": meta.image_path.iat[i]} for s, i in zip(scores[0], ids[0]) if i >= 0]


@app.get("/health")
def health():
    return {"status": "ok", "items": int(STATE["index"].ntotal)}


@app.get("/search")
def search(q: str = Query(..., min_length=1), k: int = Query(10, ge=1, le=100)):
    t0 = time.perf_counter()
    with torch.no_grad():
        vec = F.normalize(STATE["model"].encode_text(STATE["tokenizer"]([q])).float(), dim=-1)
    results = search_vector(vec, k)
    return {"query": q, "latency_ms": round((time.perf_counter() - t0) * 1000, 2), "results": results}


@app.post("/search_by_image")
async def search_by_image(file: UploadFile = File(...), k: int = Query(10, ge=1, le=100)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not read image")
    t0 = time.perf_counter()
    with torch.no_grad():
        vec = F.normalize(STATE["model"].encode_image(STATE["preprocess"](img).unsqueeze(0)).float(), dim=-1)
    results = search_vector(vec, k)
    return {"latency_ms": round((time.perf_counter() - t0) * 1000, 2), "results": results}
