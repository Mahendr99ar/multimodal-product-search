"""Report test-set Recall@K for zero-shot CLIP vs your fine-tuned checkpoint.

    python src/evaluate.py --test-csv data/test.csv --image-root abo/images/small \
        --checkpoint runs/clip_abo/best.pt --out runs/clip_abo/test_metrics.json
"""
import argparse
import json

import torch

from common import PairDataset, embed_pairs, load_model, make_loader, retrieval_metrics


def run(args, checkpoint, device):
    model, _, pre_val, tok = load_model(args.model, args.pretrained, checkpoint, device)
    dl = make_loader(PairDataset(args.test_csv, args.image_root, pre_val, tok), args.batch_size, False, args.workers)
    return retrieval_metrics(*embed_pairs(model, dl, device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="test_metrics.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    result = {"zero_shot": run(args, None, device), "fine_tuned": run(args, args.checkpoint, device)}
    for name, m in result.items():
        print(f"{name:>10}: " + "  ".join(f"{k}={v}" for k, v in m.items()))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
