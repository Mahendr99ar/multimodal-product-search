"""Shared pieces: model loading, dataset, embedding and retrieval metrics."""
import os

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def load_model(name="ViT-B-32", pretrained="laion2b_s34b_b79k", checkpoint=None, device="cpu"):
    """Create an OpenCLIP model. pretrained='none' gives random init (used by the offline smoke test)."""
    pretrained = None if pretrained in (None, "", "none") else pretrained
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(name)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(device), preprocess_train, preprocess_val, tokenizer


class PairDataset(Dataset):
    """Rows of (item_id, image_path, text) -> (image tensor, token ids, row index)."""

    def __init__(self, csv_path, image_root, preprocess, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.root, self.preprocess, self.tokenizer = image_root, preprocess, tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(os.path.join(self.root, row.image_path)).convert("RGB")
        return self.preprocess(img), self.tokenizer([row.text])[0], i


def make_loader(ds, batch_size, shuffle, workers=4, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last)


@torch.no_grad()
def embed_pairs(model, loader, device, amp=True):
    """Return L2-normalised image and text embeddings for every row in the loader."""
    model.eval()
    img_out, txt_out = [], []
    use_amp = amp and device.startswith("cuda")
    for images, tokens, _ in loader:
        with torch.autocast("cuda", enabled=use_amp):
            img_out.append(F.normalize(model.encode_image(images.to(device)).float(), dim=-1).cpu())
            txt_out.append(F.normalize(model.encode_text(tokens.to(device)).float(), dim=-1).cpu())
    return torch.cat(img_out), torch.cat(txt_out)


def recall_at_k(query_emb, gallery_emb, ks=(1, 5, 10), chunk=4096):
    """Query i's correct match is gallery i. Returns {R@k: percent}."""
    n = query_emb.shape[0]
    target = torch.arange(n)
    ranks = torch.empty(n, dtype=torch.long)
    for s in range(0, n, chunk):
        sim = query_emb[s:s + chunk] @ gallery_emb.T
        correct = sim[torch.arange(sim.shape[0]), target[s:s + chunk]].unsqueeze(1)
        ranks[s:s + chunk] = (sim > correct).sum(1)  # how many gallery items beat the true match
    return {f"R@{k}": round(100.0 * (ranks < k).float().mean().item(), 2) for k in ks}


def retrieval_metrics(img_emb, txt_emb):
    t2i = recall_at_k(txt_emb, img_emb)
    i2t = recall_at_k(img_emb, txt_emb)
    out = {f"text2image_{k}": v for k, v in t2i.items()}
    out.update({f"image2text_{k}": v for k, v in i2t.items()})
    out["mean_recall"] = round(float(np.mean(list(t2i.values()) + list(i2t.values()))), 2)
    return out
