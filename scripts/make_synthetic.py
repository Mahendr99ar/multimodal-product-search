"""Tiny fake catalog (coloured shapes + captions) so the whole pipeline can be tested without ABO."""
import argparse
import os
import random

import pandas as pd
from PIL import Image, ImageDraw

COLORS = {"red": (220, 40, 40), "green": (40, 180, 70), "blue": (40, 80, 220), "yellow": (240, 210, 40)}
SHAPES = ["circle", "square", "triangle"]


def draw(shape, rgb, size=96):
    img = Image.new("RGB", (size, size), (245, 245, 245))
    d, m = ImageDraw.Draw(img), random.randint(8, 20)
    if shape == "circle":
        d.ellipse([m, m, size - m, size - m], fill=rgb)
    elif shape == "square":
        d.rectangle([m, m, size - m, size - m], fill=rgb)
    else:
        d.polygon([(size // 2, m), (m, size - m), (size - m, size - m)], fill=rgb)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic")
    ap.add_argument("--n", type=int, default=96)
    args = ap.parse_args()
    random.seed(0)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    rows = []
    for i in range(args.n):
        c, s = random.choice(list(COLORS)), random.choice(SHAPES)
        path = f"images/{i:05d}.jpg"
        draw(s, COLORS[c]).save(os.path.join(args.out, path))
        rows.append((f"ITEM{i:05d}", path, f"{c} {s} shaped product"))
    df = pd.DataFrame(rows, columns=["item_id", "image_path", "text"])
    n = len(df)
    df.iloc[: int(n * .7)].to_csv(os.path.join(args.out, "train.csv"), index=False)
    df.iloc[int(n * .7): int(n * .85)].to_csv(os.path.join(args.out, "val.csv"), index=False)
    df.iloc[int(n * .85):].to_csv(os.path.join(args.out, "test.csv"), index=False)
    df.to_csv(os.path.join(args.out, "catalog.csv"), index=False)
    print(f"wrote {n} synthetic items to {args.out}")


if __name__ == "__main__":
    main()
