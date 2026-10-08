"""Turn the trained search system into a website that runs entirely in the browser (free static hosting).

Reads the normal serving artifacts (model.pt, index.faiss, meta.parquet) and writes:
  <out>/model/text.onnx, vision.onnx     CLIP text / image encoders, INT8 (run with onnxruntime-web)
  <out>/model/tokenizer.json             CLIP BPE vocab + merges (the page tokenizes in JavaScript)
  <out>/index/*.bin + info.json          the IVF-PQ index in a flat binary layout (~6 MB for 121K products)
  <out>/meta/NNN.json                    product names / photos, in shards loaded on demand
  <out>/index.html, README.md            the page and the Space card

    python scripts/export_static.py --artifacts artifacts --out static_site
    python scripts/export_static.py --artifacts artifacts --out static_site --push <hf-username>   # needs HF_TOKEN

--push uploads the encoders to a model repo <user>/clip-abo-search-onnx and the site to a free *static* Space
<user>/multimodal-product-search.
"""
import argparse
import json
import os
import shutil
import sys
import time

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from common import load_model  # noqa: E402

SHARD = 2000


class TextEncoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        return F.normalize(self.m.encode_text(input_ids), dim=-1)


class ImageEncoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return F.normalize(self.m.encode_image(pixel_values), dim=-1)


def onnx_export(module, example, path, input_name):
    module.eval()
    kw = dict(input_names=[input_name], output_names=["embedding"], opset_version=17)
    try:
        torch.onnx.export(module, (example,), path, dynamo=False, **kw)
    except Exception as e:  # newer torch without the legacy exporter
        print(f"  legacy exporter unavailable ({type(e).__name__}); using dynamo exporter")
        torch.onnx.export(module, (example,), path, dynamo=True, **kw)
        import onnx
        m = onnx.load(path)
        onnx.save(m, path)  # inline external data into one file


def quantize(src, dst):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm", "Gather"])


def run_onnx(path, name, x):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return s.run(None, {name: x})[0]


def export_encoders(model, preprocess, tokenizer, out):
    os.makedirs(os.path.join(out, "model"), exist_ok=True)
    report = {}
    texts = ["red running shoes", "wooden coffee table with drawers", "stainless steel water bottle 1 litre",
             "black leather office chair", "a phone case"]
    ids = tokenizer(texts)
    for kind, module, example, name, check in [
        ("text", TextEncoder(model), ids[:1], "input_ids", ids),
        ("vision", ImageEncoder(model), torch.randn(1, 3, 224, 224), "pixel_values", torch.randn(4, 3, 224, 224)),
    ]:
        t0 = time.time()
        fp32 = os.path.join(out, "model", f"{kind}_fp32.onnx")
        final = os.path.join(out, "model", f"{kind}.onnx")
        onnx_export(module, example, fp32, name)
        quantize(fp32, final)
        with torch.no_grad():
            ref = module(check).numpy()
        got = np.concatenate([run_onnx(final, name, check[i:i + 1].numpy()) for i in range(len(check))])
        cos = float((ref * got).sum(1).min())
        report[kind] = {"fp32_mb": round(os.path.getsize(fp32) / 2**20, 1),
                        "int8_mb": round(os.path.getsize(final) / 2**20, 1),
                        "min_cosine_vs_pytorch": round(cos, 4), "seconds": round(time.time() - t0, 1)}
        print(f"  {kind}: {report[kind]}")
        os.remove(fp32)
        assert cos > 0.97, f"{kind} encoder changed too much after INT8 quantization (cosine {cos:.3f})"
    return report


def export_tokenizer(tokenizer, out):
    vocab = [None] * len(tokenizer.encoder)
    for tok, i in tokenizer.encoder.items():
        vocab[i] = tok
    merges = [" ".join(p) for p, _ in sorted(tokenizer.bpe_ranks.items(), key=lambda kv: kv[1])]
    with open(os.path.join(out, "model", "tokenizer.json"), "w") as f:
        json.dump({"vocab": vocab, "merges": merges, "sot": tokenizer.sot_token_id, "eot": tokenizer.eot_token_id,
                   "context_length": tokenizer.context_length,
                   "byte_encoder": {str(k): v for k, v in tokenizer.byte_encoder.items()}}, f, ensure_ascii=False)
    # test strings for checking the JavaScript tokenizer against this one
    samples = ["Red running shoes!", "wooden coffee-table, 3 drawers", "iPhone 15 case (black)", "café crème 100ml",
               "kids' toys & games", "  lots   of   spaces  ", "USB-C to HDMI 4K@60Hz adapter"]
    with open(os.path.join(out, "model", "tokenizer_tests.json"), "w") as f:
        json.dump([{"text": s, "ids": tokenizer([s])[0].tolist()} for s in samples], f)


def export_index(index, out):
    """Write the IVF-PQ index as plain arrays. Score(q, item) = q.coarse[list] + sum_m q_m.pq[m][code_m]."""
    os.makedirs(os.path.join(out, "index"), exist_ok=True)
    n = index.ntotal
    ivf = None
    try:
        ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
    except Exception:
        pass
    if isinstance(ivf, faiss.IndexIVFPQ):
        d, nlist, M, nbits = ivf.d, ivf.nlist, ivf.pq.M, ivf.pq.nbits
        assert nbits == 8, "only 8-bit PQ codes are supported"
        coarse = ivf.quantizer.reconstruct_n(0, nlist).astype("float32")
        if not ivf.by_residual:
            coarse[:] = 0
        pq = faiss.vector_to_array(ivf.pq.centroids).reshape(M, 256, d // M).astype("float32")
        lists = np.zeros(n, dtype="uint16")
        codes = np.zeros((n, M), dtype="uint8")
        inv = ivf.invlists
        for l in range(nlist):
            sz = inv.list_size(l)
            if sz == 0:
                continue
            ids = faiss.rev_swig_ptr(inv.get_ids(l), sz).copy()
            c = faiss.rev_swig_ptr(inv.get_codes(l), sz * inv.code_size).copy().reshape(sz, M)
            lists[ids] = l
            codes[ids] = c
        kind = "ivfpq"
    else:  # exact index: compress the stored vectors with a product quantizer for the browser
        vecs = index.reconstruct_n(0, n).astype("float32")
        d, nlist, M = vecs.shape[1], 1, 64
        p = faiss.ProductQuantizer(d, M, 8)
        p.train(vecs)
        pq = faiss.vector_to_array(p.centroids).reshape(M, 256, d // M).astype("float32")
        codes = p.compute_codes(vecs).reshape(n, M)
        coarse = np.zeros((1, d), dtype="float32")
        lists = np.zeros(n, dtype="uint16")
        kind = "pq"
    coarse.tofile(os.path.join(out, "index", "coarse.bin"))
    pq.tofile(os.path.join(out, "index", "pq.bin"))
    lists.tofile(os.path.join(out, "index", "lists.bin"))
    codes.tofile(os.path.join(out, "index", "codes.bin"))

    # check the plain-array scoring against faiss itself (exhaustive search)
    rng = np.random.default_rng(0)
    q = rng.standard_normal((8, d)).astype("float32")
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    tables = np.einsum("qmd,mkd->qmk", q.reshape(8, M, d // M), pq)
    scores = (q @ coarse.T)[:, lists] + tables[:, np.arange(M)[None, :], codes].sum(-1)
    top_np = np.argsort(-scores, axis=1)[:, :10]
    if ivf is not None:
        ivf.nprobe = ivf.nlist
    _, top_faiss = index.search(q, 10)
    overlap = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top_np, top_faiss)]))
    print(f"  index: {kind}, {n:,} items, d={d}, nlist={nlist}, M={M}; top-10 overlap with faiss = {overlap:.2f}")
    if kind == "ivfpq":
        assert overlap > 0.95, "browser scoring does not match faiss"
    size = sum(os.path.getsize(os.path.join(out, "index", f)) for f in os.listdir(os.path.join(out, "index")))
    return {"kind": kind, "n": int(n), "d": int(d), "nlist": int(nlist), "M": int(M),
            "overlap_with_faiss": round(overlap, 3), "bytes": int(size)}


def describe(text):
    parts = [p.strip() for p in text.split(". ") if p.strip()]
    name = parts[0] if parts else text
    brand = next((p[6:] for p in parts if p.startswith("brand ")), "")
    color = next((p[6:] for p in parts if p.startswith("color ")), "")
    category = parts[1] if len(parts) > 1 and not parts[1].startswith(("brand ", "color ")) else ""
    return [name[:110], category[:40], brand[:40], color[:30]]


def export_meta(meta, out):
    os.makedirs(os.path.join(out, "meta"), exist_ok=True)
    for s in range(0, len(meta), SHARD):
        rows = [[p] + describe(t) for p, t in zip(meta.image_path.iloc[s:s + SHARD], meta.text.iloc[s:s + SHARD])]
        with open(os.path.join(out, "meta", f"{s // SHARD:03d}.json"), "w") as f:
            json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))


def push(out, user):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    model_repo, space = f"{user}/clip-abo-search-onnx", f"{user}/multimodal-product-search"
    api.create_repo(model_repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=os.path.join(out, "model"), repo_id=model_repo, repo_type="model",
                      commit_message="CLIP encoders (INT8 ONNX) for in-browser product search")
    api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(folder_path=out, repo_id=space, repo_type="space", ignore_patterns=["model/*"],
                      commit_message="In-browser multimodal product search")
    print(f"\nmodels : https://huggingface.co/{model_repo}")
    print(f"website: https://huggingface.co/spaces/{space}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="static_site")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--push", default=None, metavar="HF_USER")
    ap.add_argument("--github", default="https://github.com/Mahendr99ar/multimodal-product-search")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 2)

    ckpt = next((os.path.join(args.artifacts, f) for f in ("model.pt", "best.pt")
                 if os.path.exists(os.path.join(args.artifacts, f))), None)
    print(f"checkpoint: {ckpt or 'none (base CLIP weights)'}")
    model, _, preprocess, tokenizer = load_model(args.model, args.pretrained, ckpt)
    model.eval()
    shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(args.out)

    print("1/4 encoders -> ONNX INT8")
    enc = export_encoders(model, preprocess, tokenizer, args.out)
    export_tokenizer(tokenizer, args.out)
    print("2/4 index")
    idx = export_index(faiss.read_index(os.path.join(args.artifacts, "index.faiss")), args.out)
    print("3/4 product info")
    meta = pd.read_parquet(os.path.join(args.artifacts, "meta.parquet"))
    assert len(meta) == idx["n"], "meta.parquet and index.faiss have different sizes"
    export_meta(meta, args.out)

    print("4/4 page")
    model_base = f"https://huggingface.co/{args.push}/clip-abo-search-onnx/resolve/main/" if args.push else "model/"
    norm = next(t for t in preprocess.transforms if type(t).__name__ == "Normalize")
    size = next(t.size for t in preprocess.transforms if type(t).__name__ == "CenterCrop")
    prep = {"size": int(size[0] if isinstance(size, (tuple, list)) else size),
            "mean": [float(x) for x in norm.mean], "std": [float(x) for x in norm.std]}
    info = {"index": idx, "encoders": enc, "preprocess": prep, "shard": SHARD, "model_base": model_base, "github": args.github,
            "image_base": "https://amazon-berkeley-objects.s3.amazonaws.com/images/small/"}
    with open(os.path.join(args.out, "index", "info.json"), "w") as f:
        json.dump(info, f, indent=1)
    site = os.path.join(HERE, "..", "static")
    for fn in ("index.html", "README.md"):
        shutil.copy(os.path.join(site, fn), os.path.join(args.out, fn))
    print(json.dumps({"index_mb": round(idx["bytes"] / 2**20, 1), **enc}, indent=1))
    if args.push:
        push(args.out, args.push)


if __name__ == "__main__":
    main()
