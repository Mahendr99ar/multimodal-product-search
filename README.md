# Multimodal Product Search

*Describe it or photograph it. Find it in 121,549 products.*

A fine-tuned CLIP model that searches Amazon product listings by text or by photo, served two ways: a FastAPI service on AWS, and a website that does the whole search inside your browser.

[![Live demo](https://img.shields.io/badge/demo-Hugging%20Face%20Space-FFD21E.svg)](https://huggingface.co/spaces/Mahendra99ar/multimodal-product-search)
[![ONNX models](https://img.shields.io/badge/models-ONNX%20INT8-005CED.svg)](https://huggingface.co/Mahendra99ar/clip-abo-search-onnx)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Text to image R@1](https://img.shields.io/badge/text%E2%86%92image%20R%401-82.4%25-2E7D32.svg)](#results)
[![No backend](https://img.shields.io/badge/browser%20demo-no%20server-555.svg)](static/index.html)

> **This is not a shopping site.** Nothing is for sale and prices are not shown. It is a retrieval system trained and measured on a public research dataset, Amazon Berkeley Objects.

**Live demo:** [huggingface.co/spaces/Mahendra99ar/multimodal-product-search](https://huggingface.co/spaces/Mahendra99ar/multimodal-product-search)
**Models:** [Mahendra99ar/clip-abo-search-onnx](https://huggingface.co/Mahendra99ar/clip-abo-search-onnx)

---

## What this is

Type "black leather office chair" or upload a photo of one, and get the closest matching products.

Off-the-shelf CLIP already understands images and text. It does not understand product listings: titles stuffed with brand names, model numbers and sizes, photographed on white backgrounds. Fine-tuning it on photo and title pairs from the catalog roughly doubles how often the right product comes first.

---

## Why not just use CLIP as it is?

Because on this catalog it puts the right product first 41.6% of the time. After fine-tuning, 82.4%.

| Held-out test, 6,077 products | Zero-shot CLIP | Fine-tuned |
|---|---|---|
| Text → image Recall@1 | 41.6% | **82.4%** |
| Text → image Recall@5 | 65.8% | **96.2%** |
| Text → image Recall@10 | 74.2% | **98.0%** |
| Image → text Recall@1 | 48.7% | **81.5%** |
| Image → text Recall@10 | 80.9% | **97.8%** |

Recall@1 means the exact product described was the first result. The split is by product, so no product has images in both train and test.

---

## How it works

### Stage 1: Turn listings into pairs

`scripts/prepare_abo.py` reads the ABO listings, keeps English titles, pairs them with product photos, and splits by product into train, validation and test.

### Stage 2: Fine-tune

`src/train.py` trains OpenCLIP ViT-B/32 (LAION-2B weights) with symmetric InfoNCE and a learnable temperature: AdamW, warmup then cosine learning rate, mixed precision. Every other item in a batch is a negative, so bigger batches help. Five epochs of about 7.7 minutes each on a Kaggle GPU. Mean validation recall went 63.5 (zero-shot), 85.5, 88.9, 90.0, 91.5, 91.8.

### Stage 3: Evaluate

`src/evaluate.py` reports Recall@1/5/10 in both directions, zero-shot against fine-tuned, on the held-out products.

### Stage 4: Index the catalog

`src/build_index.py` embeds 121,549 catalog photos into 512-d vectors and builds a FAISS IVF-PQ index: 1,024 lists, 32 × 8-bit codes. Each product costs 32 bytes plus a list id, so the index is about 7 MB instead of 249 MB of raw float32.

### Stage 5a: Serve it as an API

`serve/app.py` loads the model and index, from a folder or from S3 through the instance's IAM role, so no keys live on the machine. It runs in Docker on AWS EC2.

### Stage 5b: Serve it with no server at all

`scripts/export_static.py` exports both encoders to ONNX, quantizes them to INT8, and flattens the index into plain arrays (6.8 MB). The page runs the text encoder with ONNX Runtime Web and scans the PQ codes in JavaScript.

---

## Architecture

```
 Training (Kaggle GPU)                          Serving
 ---------------------                          -------
 ABO listings + photos                          A) FastAPI in Docker on AWS EC2
   | prepare_abo.py: English titles,               model.pt, index.faiss, meta.parquet
   | split by product                              loaded from S3 through an IAM role
   v                                               probes 16 of 1,024 lists
 train.py: symmetric InfoNCE,
 learnable temperature                          B) Static site on Hugging Face, no server
   v                                               export_static.py: encoders -> ONNX -> INT8,
 build_index.py: embed 121,549 catalog             IVF-PQ index -> plain arrays (6.8 MB)
 photos -> FAISS IVF-PQ                            browser: ONNX Runtime Web + a JavaScript
 (1024 lists, 32 x 8-bit codes)                    scan of every PQ code
```

A few details that took the most work:

- **The browser scans everything.** It scores every product with the same IVF-PQ formula FAISS uses (coarse centroid plus PQ lookup tables), reading all 1,024 lists. For random queries its top 10 overlaps 98.8% with FAISS searching all lists.
- **"More like this" needs no image model.** The product's vector is rebuilt from its PQ code and used as the query.
- **Photo search loads lazily.** The image encoder (about 91 MB) downloads only when someone uploads a photo.
- **The tokenizer is hand-written.** open_clip's byte-level BPE is reimplemented in about 50 lines of JavaScript. `export_static.py` writes test strings with the Python token ids, and the page reproduces them exactly.

---

## Results

| | Value |
|---|---|
| Text → image Recall@1, held out | 82.4% (zero-shot 41.6%) |
| Image → text Recall@1, held out | 81.5% (zero-shot 48.7%) |
| API latency, EC2 with 2 vCPUs | 79 ms p95 per text query |
| Browser index size | 6.8 MB for 121,549 products |
| Browser top 10 vs FAISS all-lists | 98.8% overlap |
| First visit download | about 61 MB (text encoder), cached afterwards |

API latency comes from `scripts/bench_latency.py`. The browser page prints its own timing on every search, split into model time and index scan time.

---

## How it compares

| | This project | Zero-shot CLIP | Keyword search |
|---|---|---|---|
| Search by photo | ✓ | ✓ | ✗ |
| Understands product titles | ✓ | partly | ✓ |
| Finds a match with no shared words | ✓ | ✓ | ✗ |
| Runs with no server | ✓ | ✗ | ✗ |
| Text → image Recall@1 on this catalog | 82.4% | 41.6% | not measured |

---

## Run it yourself

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

Out of GPU memory: lower `--batch-size` or add `--unlocked-image-groups 4`.

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

On EC2, add `-e ARTIFACTS_S3_URI=s3://<bucket>/product-search/` and give the instance a role that can read that prefix.

**4b. Publish the browser version.**

```bash
export HF_TOKEN=hf_...   # a Hugging Face write token
python scripts/export_static.py --artifacts artifacts --out static_site --push <hf-username>
```

This creates `<hf-username>/clip-abo-search-onnx` for the encoders and a free static Space for the page.

**Smoke test, no GPU and no downloads.** Runs every script end to end on a synthetic catalog with a random model:

```bash
bash tests/smoke_test.sh
```

---

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/` | Search page |
| GET | `/health` | Health check, also used by the Docker `HEALTHCHECK` |
| GET | `/search?q=...&k=10` | Text query, returns the closest products |
| POST | `/search_by_image?k=10` | Upload a photo, returns the closest products |
| GET | `/similar` | More products like a given one |

---

## Repository structure

```
multimodal-product-search/
├── scripts/
│   ├── prepare_abo.py        ABO listings to English image-text pairs, split by product
│   ├── export_static.py      INT8 ONNX encoders, index arrays, product info, page
│   ├── bench_latency.py      p50 / p95 / p99 against the running API
│   ├── make_synthetic.py     tiny fake catalog for the smoke test
│   └── publish_static.sh
├── src/
│   ├── train.py              contrastive fine-tuning
│   ├── evaluate.py           Recall@1/5/10, zero-shot vs fine-tuned
│   ├── build_index.py        Flat or IVF-PQ FAISS index, optional upload to S3
│   └── common.py
├── serve/                    FastAPI service and its page
├── static/                   browser page and the Hugging Face Space card
├── tests/smoke_test.sh
├── Dockerfile                CPU image, base weights baked in
└── requirements.txt
```

---

## What it will not do

- Answer questions a photo cannot. Warranty, battery life and materials that don't show in a picture match poorly, because the index is built from photos.
- Send your photo anywhere in the browser version. The encoder runs on your machine.
- Search products outside the ABO catalog.

## Limitations

- Recall is measured on the 6,077 test products, not against the full 121,549-item catalog, where there are more look-alikes to confuse.
- The first visit downloads about 61 MB. Slow connections will notice.
- Product photos load from the public ABO bucket. If that bucket moves, results still work, without pictures.
- The 79 ms p95 is one instance type with one query mix. It is not a load test.

---

## Data and credit

- Amazon Berkeley Objects, CC BY 4.0 (Collins et al., 2022)
- [OpenCLIP](https://github.com/mlfoundations/open_clip), [FAISS](https://github.com/facebookresearch/faiss), [ONNX Runtime Web](https://onnxruntime.ai)

## License

Code: MIT. See [LICENSE](LICENSE). The dataset keeps its own CC BY 4.0 license.

---

*Built by [Mahendra Meena](https://www.linkedin.com/in/mahendra-meena-72047b201/). Half the work was training the model; the other half was making it fit in a browser tab.*
