# Multimodal Product Search

Type "black leather office chair" or upload a photo, and get the closest matches from 121,549 Amazon product listings.

**Live demo:** https://huggingface.co/spaces/Mahendra99ar/multimodal-product-search
**Models (ONNX):** https://huggingface.co/Mahendra99ar/clip-abo-search-onnx

The demo runs entirely in the browser. There is no backend: the page downloads the text encoder once (about 61 MB), then encodes your query and scans the index on your machine. The same model also runs as a Dockerised FastAPI service, which I deployed on AWS EC2.

## Results

I fine-tuned OpenCLIP ViT-B/32 (LAION-2B weights) on product photo and title pairs from the Amazon Berkeley Objects (ABO) dataset. The test set is 6,077 products that were not used in training; the split is by product, so no product has images in both train and test.

| Held-out test, 6,077 products | Zero-shot CLIP | Fine-tuned |
|---|---|---|
| Text → image Recall@1 | 41.6% | **82.4%** |
| Text → image Recall@5 | 65.8% | **96.2%** |
| Text → image Recall@10 | 74.2% | **98.0%** |
| Image → text Recall@1 | 48.7% | **81.5%** |
| Image → text Recall@10 | 80.9% | **97.8%** |

Recall@1 means the exact product described was the first result. Training took 5 epochs of about 7.7 minutes each on a Kaggle GPU; validation recall rose every epoch (mean recall 63.5 zero-shot, then 85.5, 88.9, 90.0, 91.5, 91.8).

Serving numbers:

- FastAPI + FAISS on an EC2 instance with 2 vCPUs: **79 ms p95** per text query (`scripts/bench_latency.py`).
- Browser version: the page shows the measured time for every search, split into model time and index scan time.

## How it works

```
 Training (Kaggle GPU)                          Serving
 ─────────────────────                          ───────
 ABO listings + photos                          A) FastAPI in Docker on AWS EC2
   │ prepare_abo.py: English titles,                model.pt, index.faiss, meta.parquet
   │ split by product                               loaded from S3 through the instance's IAM role
   ▼
 train.py: symmetric InfoNCE,                   B) Static website on Hugging Face (no server)
 learnable temperature                              export_static.py: encoders → ONNX → INT8,
   ▼                                                IVF-PQ index → plain arrays (6.8 MB)
 build_index.py: embed 121,549 catalog              browser: ONNX Runtime Web + a JavaScript
 photos → FAISS IVF-PQ (1024 lists,                 scan of the PQ codes
 32 × 8-bit codes)
```

**Index.** Each product photo is a 512-d vector. IVF-PQ stores each one as 32 bytes plus a list id, so the whole catalog index is about 7 MB instead of 249 MB for raw float32 vectors. The browser version scores every product with the same IVF-PQ formula (coarse centroid plus PQ lookup tables), so it reads all 1,024 lists instead of the 16 the API probes. On the real index, for random query vectors, its top 10 overlaps 98.8% with FAISS searching all lists.

**"More like this"** needs no image model in the browser. The product's vector is rebuilt from its PQ code and used as the query.

**Photo search** loads the image encoder (about 91 MB) only when someone uploads a photo.

**Tokenizer.** The page reimplements open_clip's byte-level BPE in about 50 lines of JavaScript. `export_static.py` writes test strings with the Python token ids, and the page reproduces them exactly.

## Repository

| Path | Contents |
|---|---|
| `scripts/prepare_abo.py` | Turns ABO listings into English image-text pairs, split by product |
| `src/train.py` | Contrastive fine-tuning: symmetric InfoNCE, learnable temperature, AdamW, warmup then cosine LR, mixed precision |
| `src/evaluate.py` | Text → image and image → text Recall@1/5/10, zero-shot vs fine-tuned |
| `src/build_index.py` | Embeds the catalog and builds a Flat or IVF-PQ FAISS index, optional upload to S3 |
| `serve/` + `Dockerfile` | FastAPI service and its web page; loads artifacts from a folder or from S3 |
| `scripts/bench_latency.py` | p50/p95/p99 latency against the running API |
| `scripts/export_static.py` | Builds the browser version: INT8 ONNX encoders, index arrays, product info, page |
| `static/` | The browser page and the Hugging Face Space card |
| `tests/smoke_test.sh` | Runs every script end to end on a synthetic catalog with a random model, no GPU or downloads |

## Reproduce

**1. Data (about 3.5 GB).**
```bash
mkdir abo && cd abo
wget https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar
wget https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar
tar xf abo-listings.tar && tar xf abo-images-small.tar && cd ..
pip install -r requirements.txt
python scripts/prepare_abo.py --abo-root abo --out-dir data --use-other-images
```

**2. Fine-tune and evaluate (one T4 is enough).**
```bash
python src/train.py --train-csv data/train.csv --val-csv data/val.csv --image-root abo/images/small \
    --epochs 5 --batch-size 128 --lr 1e-5 --out runs/clip_abo
python src/evaluate.py --test-csv data/test.csv --image-root abo/images/small \
    --checkpoint runs/clip_abo/best.pt --out runs/clip_abo/test_metrics.json
```
If the GPU runs out of memory, lower `--batch-size` or add `--unlocked-image-groups 4`. Larger batches help contrastive training because every other item in the batch is a negative.

**3. Build the index.**
```bash
python src/build_index.py --catalog-csv data/catalog.csv --image-root abo/images/small \
    --checkpoint runs/clip_abo/best.pt --index-type ivfpq --out artifacts
cp runs/clip_abo/best.pt artifacts/model.pt
```

**4a. Run the API.**
```bash
docker build -t product-search .
docker run -p 8000:8000 -v $PWD/artifacts:/app/artifacts product-search
curl "localhost:8000/search?q=wooden%20coffee%20table&k=5"
curl -X POST -F "file=@some_product.jpg" "localhost:8000/search_by_image?k=5"
```
On EC2 I ran the same image with `-e ARTIFACTS_S3_URI=s3://<bucket>/product-search/` and an instance role that can read that prefix, so no keys live on the machine.

**4b. Publish the browser version.**
```bash
export HF_TOKEN=hf_...   # Hugging Face write token
python scripts/export_static.py --artifacts artifacts --out static_site --push <hf-username>
```
This creates `<hf-username>/clip-abo-search-onnx` for the encoders and a free static Space for the page. `scripts/colab_publish_search.ipynb` does the same in Colab.

## Limitations

- Recall is measured on the 6,077 test products, not against the full 121,549-item catalog, where there are more look-alike products to confuse.
- The index searches product photos. A query about something that does not show in a photo (warranty, battery life) matches poorly.
- The first visit downloads about 61 MB; the browser caches it afterwards.
- Product photos load from the public ABO bucket. If that bucket moves, results still work but without pictures.

Data: Amazon Berkeley Objects, CC BY 4.0 (Collins et al., 2022).
