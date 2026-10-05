#!/usr/bin/env bash
# End-to-end check on a synthetic catalog, CPU only, no downloads (random-init model).
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(mktemp -d)
python scripts/make_synthetic.py --out "$T/syn" --n 96
python src/train.py --train-csv "$T/syn/train.csv" --val-csv "$T/syn/val.csv" --image-root "$T/syn" \
    --pretrained none --epochs 1 --batch-size 16 --max-steps 3 --warmup 1 --workers 0 --out "$T/run"
python src/evaluate.py --test-csv "$T/syn/test.csv" --image-root "$T/syn" --checkpoint "$T/run/best.pt" \
    --pretrained none --batch-size 16 --workers 0 --out "$T/run/test_metrics.json"
python src/build_index.py --catalog-csv "$T/syn/catalog.csv" --image-root "$T/syn" --checkpoint "$T/run/best.pt" \
    --pretrained none --workers 0 --batch-size 32 --out "$T/art"
cp "$T/run/best.pt" "$T/art/model.pt"
ARTIFACT_DIR="$T/art" PRETRAINED=none uvicorn serve.app:app --port 8765 > "$T/server.log" 2>&1 &
PID=$!; trap 'kill $PID 2>/dev/null' EXIT
for i in $(seq 1 60); do curl -sf localhost:8765/health >/dev/null && break; sleep 1; done
curl -sf "localhost:8765/search?q=red%20circle&k=3" | python -m json.tool
curl -sf -X POST -F "file=@$T/syn/images/00000.jpg" "localhost:8765/search_by_image?k=3" | python -m json.tool
python scripts/bench_latency.py --url http://localhost:8765 --n 30
echo "SMOKE TEST PASSED"
