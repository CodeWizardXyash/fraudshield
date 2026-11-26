"""Send a normal history, then a card-testing burst, to a running API."""

from __future__ import annotations

import argparse
import uuid
from datetime import UTC, datetime, timedelta

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()
    run = uuid.uuid4().hex[:8]
    now = datetime.now(UTC)

    def send(index, when, **fields):
        payload = {"transaction_id": f"demo-{run}-{index}", "customer_id": f"demo-{run}",
                   "timestamp": when.isoformat(), "country": "IN", **fields}
        response = requests.post(f"{args.url}/v1/predict", json=payload,
                                 headers={"x-api-key": args.api_key}, timeout=10)
        response.raise_for_status()
        return response.json()

    print("Normal history")
    for i in range(20):
        result = send(i, now - timedelta(days=20 - i), amount=18 + i % 5,
                      merchant_id=f"M-grocery-00{i % 3}", category="grocery", channel="pos")
    print(f"  last: p={result['fraud_probability']:.4f} {result['decision']}")

    print("Card-testing burst from a new device")
    for i in range(8):
        result = send(100 + i, now - timedelta(minutes=8 - i), amount=1.0 + i / 10,
                      merchant_id=f"M-digital_goods-{i:03d}", category="digital_goods",
                      channel="online", device_id="unknown-device")
        reasons = ", ".join(r["feature"] for r in result["reasons"])
        print(f"  #{i + 1}: p={result['fraud_probability']:.4f} {result['decision']:7} {reasons}")


if __name__ == "__main__":
    main()
