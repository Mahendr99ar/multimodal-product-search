"""Turn the raw Amazon Berkeley Objects (ABO) dump into image-text pair CSVs.

Expected layout after downloading (see README):
    <abo_root>/listings/metadata/listings_*.json.gz
    <abo_root>/images/metadata/images.csv.gz
    <abo_root>/images/small/<path from images.csv>

Outputs train.csv / val.csv / test.csv / catalog.csv with columns:
    item_id, image_path, text
The split is done by item_id so no product leaks between train and test.
"""
import argparse
import glob
import gzip
import json
import os
import random

import pandas as pd


def pick_en(values):
    """ABO stores multilingual fields as [{language_tag, value}, ...]. Keep English."""
    if not values:
        return None
    for v in values:
        if str(v.get("language_tag", "")).startswith("en"):
            return v.get("value")
    return None


def build_text(rec, max_bullets):
    name = pick_en(rec.get("item_name"))
    if not name:
        return None
    parts = [name.strip()]
    ptype = rec.get("product_type")
    if ptype:
        parts.append(ptype[0]["value"].replace("_", " ").lower())
    color = pick_en(rec.get("color"))
    if color:
        parts.append(f"color {color}")
    brand = pick_en(rec.get("brand"))
    if brand:
        parts.append(f"brand {brand}")
    bullets = [b["value"] for b in rec.get("bullet_point") or []
               if str(b.get("language_tag", "")).startswith("en")][:max_bullets]
    parts.extend(bullets)
    # CLIP's tokenizer truncates at 77 tokens; keep the string short anyway.
    return ". ".join(parts)[:400]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abo-root", required=True)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--max-bullets", type=int, default=1)
    ap.add_argument("--use-other-images", action="store_true",
                    help="add secondary product photos as extra training pairs")
    ap.add_argument("--max-items", type=int, default=0, help="subsample for quick runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Search recursively so it works whatever folder the tar files were extracted into.
    img_meta = glob.glob(os.path.join(args.abo_root, "**/images.csv.gz"), recursive=True)
    listing_files = sorted(glob.glob(os.path.join(args.abo_root, "**/listings_*.json.gz"), recursive=True))
    if not img_meta or not listing_files:
        raise SystemExit(f"could not find images.csv.gz / listings_*.json.gz under {args.abo_root}")
    small_dir = os.path.join(os.path.dirname(os.path.dirname(img_meta[0])), "small")
    print(f"image metadata: {img_meta[0]}\nuse --image-root {small_dir} for training\n{len(listing_files)} listing files")
    images = pd.read_csv(img_meta[0])
    id2path = dict(zip(images.image_id, images.path))

    items = {}
    for fn in listing_files:
        with gzip.open(fn, "rt") as f:
            for line in f:
                rec = json.loads(line)
                main_id = rec.get("main_image_id")
                if main_id not in id2path or rec["item_id"] in items:
                    continue
                text = build_text(rec, args.max_bullets)
                if not text:
                    continue
                others = [id2path[i] for i in rec.get("other_image_id") or [] if i in id2path]
                items[rec["item_id"]] = (text, id2path[main_id], others)
    print(f"usable items: {len(items):,}")

    ids = sorted(items)
    random.Random(args.seed).shuffle(ids)
    if args.max_items:
        ids = ids[: args.max_items]
    n_val, n_test = int(len(ids) * args.val_frac), int(len(ids) * args.test_frac)
    splits = {"val": ids[:n_val], "test": ids[n_val:n_val + n_test], "train": ids[n_val + n_test:]}

    os.makedirs(args.out_dir, exist_ok=True)
    for split, split_ids in splits.items():
        rows = []
        for iid in split_ids:
            text, main_path, others = items[iid]
            rows.append((iid, main_path, text))
            if split == "train" and args.use_other_images:
                rows += [(iid, p, text) for p in others]
        df = pd.DataFrame(rows, columns=["item_id", "image_path", "text"])
        df.to_csv(os.path.join(args.out_dir, f"{split}.csv"), index=False)
        print(f"{split}: {len(split_ids):,} items, {len(df):,} pairs")

    # The search catalog = every item's main image (what the API searches over).
    cat = pd.DataFrame([(i, items[i][1], items[i][0]) for i in ids], columns=["item_id", "image_path", "text"])
    cat.to_csv(os.path.join(args.out_dir, "catalog.csv"), index=False)
    print(f"catalog: {len(cat):,} items")


if __name__ == "__main__":
    main()
