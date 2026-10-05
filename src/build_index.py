"""Embed the whole product catalog with the fine-tuned image encoder and build a FAISS index.

    python src/build_index.py --catalog-csv data/catalog.csv --image-root abo/images/small \
        --checkpoint runs/clip_abo/best.pt --out artifacts/ --index-type ivfpq \
        --s3-uri s3://<your-bucket>/product-search/   # optional upload

flat  : exact inner-product search (fine up to a few hundred thousand items)
ivfpq : inverted file + product quantisation, ~16x smaller, approximate, scales to millions
"""
import argparse
import json
import os
import time

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import load_model


class ImageOnly(Dataset):
    def __init__(self, df, root, preprocess):
        self.paths, self.root, self.pre = df.image_path.tolist(), root, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.pre(Image.open(os.path.join(self.root, self.paths[i])).convert("RGB"))


def build_faiss(emb, index_type, nlist):
    d = emb.shape[1]
    if index_type == "flat" or len(emb) < 39 * nlist:  # IVF needs enough points to train its centroids
        index = faiss.IndexFlatIP(d)
    else:
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, 32, 8, faiss.METRIC_INNER_PRODUCT)
        index.train(emb)
        index.nprobe = 16
    index.add(emb)
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog-csv", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--index-type", choices=["flat", "ivfpq"], default="flat")
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--s3-uri", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, pre_val, _ = load_model(args.model, args.pretrained, args.checkpoint, device)
    model.eval()
    df = pd.read_csv(args.catalog_csv)
    dl = DataLoader(ImageOnly(df, args.image_root, pre_val), batch_size=args.batch_size, num_workers=args.workers)

    t0, chunks = time.time(), []
    with torch.no_grad(), torch.autocast("cuda", enabled=device == "cuda"):
        for x in dl:
            chunks.append(F.normalize(model.encode_image(x.to(device)).float(), dim=-1).cpu().numpy())
    emb = np.concatenate(chunks).astype("float32")
    embed_sec = time.time() - t0
    print(f"embedded {len(emb):,} images in {embed_sec:.1f}s ({len(emb) / embed_sec:.0f} img/s)")

    index = build_faiss(emb, args.index_type, args.nlist)
    os.makedirs(args.out, exist_ok=True)
    faiss.write_index(index, os.path.join(args.out, "index.faiss"))
    df[["item_id", "image_path", "text"]].to_parquet(os.path.join(args.out, "meta.parquet"), index=False)
    info = {"n_items": len(emb), "dim": int(emb.shape[1]), "index": type(index).__name__,
            "embed_images_per_sec": round(len(emb) / embed_sec, 1), "model": args.model}
    with open(os.path.join(args.out, "index_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(info)

    if args.s3_uri:
        import boto3
        bucket, _, prefix = args.s3_uri.replace("s3://", "").partition("/")
        s3 = boto3.client("s3")
        files = ["index.faiss", "meta.parquet", "index_info.json"]
        for fn in files:
            s3.upload_file(os.path.join(args.out, fn), bucket, prefix.rstrip("/") + "/" + fn)
        if args.checkpoint:
            s3.upload_file(args.checkpoint, bucket, prefix.rstrip("/") + "/model.pt")
        print(f"uploaded to {args.s3_uri}")


if __name__ == "__main__":
    main()
