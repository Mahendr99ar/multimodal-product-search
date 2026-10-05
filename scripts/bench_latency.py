"""Measure end-to-end /search latency (p50 / p95 / p99) against a running server.

    python scripts/bench_latency.py --url http://localhost:8000 --n 300
"""
import argparse
import json
import random
import time

import httpx
import numpy as np

QUERIES = ["red running shoes", "wooden dining table", "stainless steel water bottle", "black leather office chair",
           "phone case with card holder", "blue cotton bedsheet", "ceramic coffee mug", "kids backpack",
           "led desk lamp", "yoga mat non slip", "men's formal shirt", "kitchen knife set"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    with httpx.Client(base_url=args.url, timeout=30) as c:
        for q in QUERIES[:3]:  # warm-up
            c.get("/search", params={"q": q, "k": args.k})
        lat, server = [], []
        for _ in range(args.n):
            t0 = time.perf_counter()
            r = c.get("/search", params={"q": random.choice(QUERIES), "k": args.k})
            lat.append((time.perf_counter() - t0) * 1000)
            server.append(r.json()["latency_ms"])
    p = lambda a, q: round(float(np.percentile(a, q)), 2)
    out = {"requests": args.n, "client_p50_ms": p(lat, 50), "client_p95_ms": p(lat, 95), "client_p99_ms": p(lat, 99),
           "server_p50_ms": p(server, 50), "server_p95_ms": p(server, 95)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
