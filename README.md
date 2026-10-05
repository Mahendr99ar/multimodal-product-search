# Multimodal Product Search (CLIP + FAISS + AWS)

Search a product catalog with natural language ("black leather office chair") or with a photo.
CLIP is fine-tuned on Amazon Berkeley Objects (ABO) product listings. The fine-tuned image encoder embeds the whole catalog into a FAISS index, and a FastAPI service in Docker serves it, with artifacts stored in S3.

```
                 ┌───────────── offline (GPU) ─────────────┐        ┌──────── online (CPU, Docker) ──────-──┐
ABO listings ──► prepare_abo.py ──► train.py (InfoNCE)     │        │  GET  /search?q=...  ─► text encoder ─┐
ABO images        (item-level split)   │  zero-shot vs     │        │  POST /search_by_image ─► img encoder ┤
                                       │  fine-tuned R@K   │        │                                       ▼
                                       ▼                   │  S3    │                             FAISS (flat / IVF-PQ)
                         build_index.py: embed catalog ────┼──────► │                                       ▼
                         → index.faiss + meta.parquet      │        │                            top-k products + scores
                 └─────────────────────────────────────────┘        └──────────────────────────────────────┘
```

| Piece | What it shows |
|---|---|
| `scripts/prepare_abo.py` | Parses ~150K multilingual listings into English image-text pairs, split by item to avoid leakage |
| `src/train.py` | Contrastive fine-tuning: symmetric InfoNCE, learnable temperature, AdamW, warmup + cosine LR, mixed precision, optional partial freezing |
| `src/evaluate.py` | Text→image and image→text Recall@1/5/10, **zero-shot vs fine-tuned** on held-out items |
| `src/build_index.py` | Batch-embeds the catalog, builds an exact (Flat) or compressed approximate (IVF-PQ) FAISS index, uploads to S3 |
| `serve/app.py` + `Dockerfile` | FastAPI text and image search, loads artifacts from S3 at startup |
| `scripts/bench_latency.py` | p50/p95/p99 latency of the running service |

## 1. Get the data (~3.5 GB)

ABO is a public dataset from Amazon. You can use Kaggle, Colab or any Linux box.

```bash
mkdir abo && cd abo
wget https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar
wget https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar
tar xf abo-listings.tar && tar xf abo-images-small.tar && cd ..
# alternative: aws s3 sync s3://amazon-berkeley-objects/listings/ abo/listings/ --no-sign-request (same for images/)

pip install -r requirements.txt
python scripts/prepare_abo.py --abo-root abo --out-dir data --use-other-images
```
The script prints the `--image-root` to use (the `images/small` folder). It also prints the item counts, which you should note for your resume. The rest of this guide assumes `abo/images/small`.

## 2. Fine-tune (GPU: Kaggle T4/P100 works)

```bash
python src/train.py --train-csv data/train.csv --val-csv data/val.csv --image-root abo/images/small \
    --epochs 5 --batch-size 128 --lr 1e-5 --out runs/clip_abo
```
- The run first logs **zero-shot** validation recall. That is your baseline.
- `runs/clip_abo/metrics.json` stores the per-epoch recall, loss and timing.
- Out of GPU memory? Lower `--batch-size`, or add `--unlocked-image-groups 4` to train only the last image blocks.
- Contrastive learning benefits from larger batches, because every other item in the batch acts as a negative. Use the largest batch that fits.
- Kaggle tip: train with `--workers 2`, and save `runs/` to `/kaggle/working`.

## 3. Evaluate on the held-out test items

```bash
python src/evaluate.py --test-csv data/test.csv --image-root abo/images/small \
    --checkpoint runs/clip_abo/best.pt --out runs/clip_abo/test_metrics.json
```

## 4. Build the search index

```bash
python src/build_index.py --catalog-csv data/catalog.csv --image-root abo/images/small \
    --checkpoint runs/clip_abo/best.pt --index-type ivfpq --out artifacts \
    --s3-uri s3://<your-bucket>/product-search/      # optional
cp runs/clip_abo/best.pt artifacts/model.pt           # when serving from a local folder
```

## 5. Serve

```bash
# local
ARTIFACT_DIR=artifacts uvicorn serve.app:app --port 8000
curl "localhost:8000/search?q=wooden%20coffee%20table&k=5"
curl -X POST -F "file=@some_product.jpg" "localhost:8000/search_by_image?k=5"
python scripts/bench_latency.py --url http://localhost:8000 --n 300

# docker
docker build -t product-search .
docker run -p 8000:8000 -v $PWD/artifacts:/app/artifacts product-search
```

## 6. Deploy on AWS

1. **S3.** Upload the artifacts with `build_index.py --s3-uri` as in step 4. This uploads `index.faiss`, `meta.parquet` and `model.pt`.
2. **ECR.** Push the image:
   ```bash
   aws ecr create-repository --repository-name product-search
   aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
   docker tag product-search <acct>.dkr.ecr.<region>.amazonaws.com/product-search:latest
   docker push <acct>.dkr.ecr.<region>.amazonaws.com/product-search:latest
   ```
3. **Run.** Use App Runner or ECS Fargate (2 vCPU / 4 GB) with environment variable `ARTIFACTS_S3_URI=s3://<bucket>/product-search/`. Give the task role `s3:GetObject` on that prefix.
   - Lambda also works as a container image (raise memory to 3–4 GB), but cold starts take several seconds.
4. **Shut it down** when you're done to stay within the free tier. Take screenshots and record a short demo first.

## 7. What to record for your resume

After your runs, fill these from `metrics.json`, `test_metrics.json`, `index_info.json` and the benchmark output:

> Fine-tuned CLIP (ViT-B/32) with a contrastive InfoNCE objective on **[N]K** Amazon Berkeley Objects image-text pairs, improving text→image Recall@10 on held-out products from **[zero-shot]%** to **[fine-tuned]%**. Built a FAISS IVF-PQ index over **[M]K** catalog items and served text- and image-based search through a Dockerised FastAPI service on AWS (S3, ECR, App Runner) at **[p95] ms** p95 latency.

Be ready to explain in interviews:
- Why InfoNCE needs large batches.
- What the learnable temperature does.
- Why the split is by item rather than by image.
- The trade-off between Flat and IVF-PQ: recall vs memory and speed. `nprobe` controls it.
- How you would scale to 100M items: sharding, GPU FAISS, and embedding with Spark or AWS Batch.

## Smoke test (no GPU, no downloads)

```bash
bash tests/smoke_test.sh   # synthetic catalog, random-init model, exercises every script and the API
```
