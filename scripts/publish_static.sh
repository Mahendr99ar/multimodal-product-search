#!/bin/bash
# One command to publish the in-browser version of the search to a free Hugging Face *static* Space.
# Run on the EC2 instance (it already has the Docker image and access to the artifacts):
#     export HF_TOKEN=hf_...            # your Hugging Face WRITE token
#     bash scripts/publish_static.sh    # optional: HF username as first argument
set -e
HF_USER=${1:-Mahendra99ar}
REPO=$(cd "$(dirname "$0")/.." && pwd)
ART=$HOME/artifacts_copy
DOCKER="docker"; $DOCKER ps >/dev/null 2>&1 || DOCKER="sudo docker"

if [ -z "$HF_TOKEN" ]; then echo "ERROR: first run   export HF_TOKEN=hf_...   (your Hugging Face WRITE token)"; exit 1; fi

echo "== 1/3 finding the trained model + index"
if [ ! -f "$ART/index.faiss" ]; then
  mkdir -p "$ART"
  if $DOCKER cp search:/app/artifacts/. "$ART/" 2>/dev/null && [ -f "$ART/index.faiss" ]; then
    echo "copied from the 'search' container"
  else
    S3=$($DOCKER inspect search --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | grep '^ARTIFACTS_S3_URI=' | cut -d= -f2-)
    if [ -n "$S3" ]; then aws s3 cp "$S3" "$ART/" --recursive
    else echo "ERROR: could not find index.faiss. Run   aws s3 ls   and send the output."; exit 1; fi
  fi
fi
[ -f "$ART/model.pt" ] || [ -f "$ART/best.pt" ] || echo "WARNING: no model.pt found - the base (not fine-tuned) CLIP weights will be used"
ls -lh "$ART"

echo "== 2/3 Docker image"
IMG=$($DOCKER inspect search --format '{{.Config.Image}}' 2>/dev/null || true)
if [ -z "$IMG" ]; then echo "building the image (5-10 min)"; $DOCKER build -t search "$REPO"; IMG=search; fi
echo "using image: $IMG"

echo "== 3/3 export to ONNX + upload to Hugging Face (about 5 min)"
$DOCKER run --rm -e HF_TOKEN="$HF_TOKEN" -v "$ART":/app/artifacts -v "$REPO/scripts":/app/scripts -v "$REPO/static":/app/static "$IMG" \
  sh -c "pip install -q onnx onnxruntime && python scripts/export_static.py --artifacts artifacts --out /tmp/site --push $HF_USER"
