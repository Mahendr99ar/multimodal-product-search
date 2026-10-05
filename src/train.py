"""Fine-tune CLIP on product image-text pairs with a symmetric contrastive (InfoNCE) loss.

Records zero-shot Recall@K first, so you get a clean "before vs after" number.

Example (Kaggle / Colab GPU):
    python src/train.py --train-csv data/train.csv --val-csv data/val.csv \
        --image-root abo/images/small --epochs 5 --batch-size 256 --out runs/clip_abo
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from common import PairDataset, embed_pairs, load_model, make_loader, retrieval_metrics


def clip_loss(img_feat, txt_feat, logit_scale):
    """Symmetric InfoNCE: matching (image, text) pairs sit on the diagonal of the similarity matrix."""
    img_feat, txt_feat = F.normalize(img_feat, dim=-1), F.normalize(txt_feat, dim=-1)
    logits = logit_scale * img_feat @ txt_feat.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def cosine_with_warmup(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def evaluate(model, loader, device):
    img, txt = embed_pairs(model, loader, device)
    return retrieval_metrics(img, txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--unlocked-image-groups", type=int, default=-1,
                    help="-1 = train the whole image tower; N = freeze all but the last N layer groups")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    ap.add_argument("--out", default="runs/clip_abo")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    model, pre_train, pre_val, tok = load_model(args.model, args.pretrained, device=device)
    if args.unlocked_image_groups >= 0:
        model.lock_image_tower(unlocked_groups=args.unlocked_image_groups, freeze_bn_stats=True)

    train_dl = make_loader(PairDataset(args.train_csv, args.image_root, pre_train, tok),
                           args.batch_size, shuffle=True, workers=args.workers, drop_last=True)
    val_dl = make_loader(PairDataset(args.val_csv, args.image_root, pre_val, tok),
                         args.batch_size, shuffle=False, workers=args.workers)

    # No weight decay on biases, norms, logit_scale (standard CLIP recipe).
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 or "logit_scale" in n else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    total = args.epochs * len(train_dl)
    if args.max_steps:
        total = min(total, args.max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_with_warmup(s, total, args.warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    history = {"zero_shot": evaluate(model, val_dl, device), "epochs": []}
    print("zero-shot val:", history["zero_shot"])
    best, step = history["zero_shot"]["mean_recall"], 0

    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for i, (images, tokens, _) in enumerate(train_dl):
            images, tokens = images.to(device, non_blocking=True), tokens.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=device == "cuda"):
                img_f = model.encode_image(images)
                txt_f = model.encode_text(tokens)
                loss = clip_loss(img_f.float(), txt_f.float(), model.logit_scale.exp())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            with torch.no_grad():  # CLIP clamps temperature so logits don't explode
                model.logit_scale.clamp_(0, math.log(100))
            running += loss.item()
            step += 1
            if i % 50 == 0:
                print(f"epoch {epoch} step {i}/{len(train_dl)} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}")
            if args.max_steps and step >= args.max_steps:
                break

        metrics = evaluate(model, val_dl, device)
        metrics.update(epoch=epoch, train_loss=round(running / (i + 1), 4), epoch_sec=round(time.time() - t0, 1))
        history["epochs"].append(metrics)
        print("val:", metrics)
        torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.out, "last.pt"))
        if metrics["mean_recall"] > best:
            best = metrics["mean_recall"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.out, "best.pt"))
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump(history, f, indent=2)
        if args.max_steps and step >= args.max_steps:
            break

    if not os.path.exists(os.path.join(args.out, "best.pt")):  # fine-tuning never beat zero-shot
        torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.out, "best.pt"))
    print(f"done. best val mean recall {best:.2f} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()
