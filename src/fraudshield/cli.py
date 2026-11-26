"""Command line interface: simulate, train, monitor, serve."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from fraudshield.config import load_config


def read_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fraudshield")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("simulate", help="generate a synthetic dataset")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default="data/transactions.csv.gz")

    p = commands.add_parser("train", help="train, evaluate and register a model")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data", default="data/transactions.csv.gz")
    p.add_argument("--models", default="models")
    p.add_argument("--reports", default="reports")

    p = commands.add_parser("monitor", help="drift report for a batch (exit code 2 on alert)")
    p.add_argument("--data", required=True, help="time-ordered CSV, including warm-up history")
    p.add_argument("--since", help="only report rows at or after this ISO timestamp")
    p.add_argument("--models", default="models")

    p = commands.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "simulate":
        from fraudshield.data.simulate import simulate

        frame = simulate(load_config(args.config).data)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
        print(f"wrote {len(frame):,} rows ({frame.is_fraud.mean():.2%} fraud) to {args.out}")
    elif args.command == "train":
        from fraudshield.models.train import train

        summary = train(read_data(Path(args.data)), load_config(args.config),
                        Path(args.models), Path(args.reports))
        print(json.dumps(summary["test_metrics"], indent=2))
        print(f"{summary['algorithm']} {summary['version']}: {summary['gate_reason']}")
        return 0 if summary["promoted"] else 1
    elif args.command == "monitor":
        from fraudshield.features.store import build_features
        from fraudshield.models.registry import load_champion
        from fraudshield.monitoring.drift import drift_report
        from fraudshield.schema import validate_frame

        bundle, _ = load_champion(Path(args.models))
        frame = read_data(Path(args.data))
        validate_frame(frame)
        features, _store = build_features(frame)
        if args.since:
            features = features[frame["timestamp"] >= pd.Timestamp(args.since, tz="UTC")]
        if features.empty:
            parser.error("no rows to monitor")
        features = features.assign(score=bundle["model"].predict_proba(features)[:, 1])
        report = drift_report(bundle["profiles"], features)
        print(json.dumps(report, indent=2))
        return 2 if report["status"] == "alert" else 0
    else:
        # Must be set before scikit-learn loads: thread pools slow single-row scoring.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        import uvicorn

        # One worker only: customer state lives in process memory.
        uvicorn.run("fraudshield.serving.api:app_factory", factory=True,
                    host=args.host, port=args.port, workers=1, access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
