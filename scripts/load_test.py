"""Measure latency and throughput of a running API."""

from __future__ import annotations

import argparse
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import numpy as np
import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    run = uuid.uuid4().hex[:8]

    def call(i: int) -> float:
        payload = {"transaction_id": f"load-{run}-{i}", "customer_id": f"load-{run}-{i}",
                   "timestamp": datetime.now(UTC).isoformat(), "amount": 42.5,
                   "merchant_id": "M-retail-001", "category": "retail", "channel": "pos",
                   "country": "IN"}
        start = time.perf_counter()
        requests.post(f"{args.url}/v1/predict", json=payload,
                      headers={"x-api-key": args.api_key}, timeout=30).raise_for_status()
        return (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as pool:
        latency = list(pool.map(call, range(args.requests)))
    elapsed = time.perf_counter() - start
    p50, p95, p99 = np.percentile(latency, [50, 95, 99])
    print(f"requests={args.requests} concurrency={args.concurrency}")
    print(f"throughput={args.requests / elapsed:.1f} req/s")
    print(f"latency ms: p50={p50:.1f} p95={p95:.1f} p99={p99:.1f}")


if __name__ == "__main__":
    main()
