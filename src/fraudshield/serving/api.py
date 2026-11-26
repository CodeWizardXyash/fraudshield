"""ASGI application (Starlette + Pydantic)."""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
from collections import Counter
from datetime import UTC, datetime, timedelta

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from fraudshield.features.store import OutOfOrderError
from fraudshield.schema import Transaction
from fraudshield.serving.service import DuplicateConflictError, ScoringService

log = logging.getLogger("fraudshield.api")
MAX_BODY = 16_384
MAX_CLOCK_SKEW = timedelta(minutes=5)
BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


class Metrics:
    """Prometheus text-format counters (per process)."""

    def __init__(self) -> None:
        self.requests: Counter = Counter()
        self.decisions: Counter = Counter()
        self.buckets: Counter = Counter()
        self.latency_sum = 0.0
        self.latency_count = 0

    def observe(self, seconds: float) -> None:
        self.latency_sum += seconds
        self.latency_count += 1
        for edge in BUCKETS:
            if seconds <= edge:
                self.buckets[edge] += 1

    def render(self) -> str:
        lines = ["# TYPE fraudshield_requests_total counter"]
        lines += [f'fraudshield_requests_total{{status="{k}"}} {v}'
                  for k, v in sorted(self.requests.items())]
        lines.append("# TYPE fraudshield_decisions_total counter")
        lines += [f'fraudshield_decisions_total{{decision="{k}"}} {v}'
                  for k, v in sorted(self.decisions.items())]
        lines.append("# TYPE fraudshield_latency_seconds histogram")
        lines += [f'fraudshield_latency_seconds_bucket{{le="{e}"}} {self.buckets[e]}'
                  for e in BUCKETS]
        lines += [
            f'fraudshield_latency_seconds_bucket{{le="+Inf"}} {self.latency_count}',
            f"fraudshield_latency_seconds_sum {self.latency_sum}",
            f"fraudshield_latency_seconds_count {self.latency_count}",
        ]
        return "\n".join(lines) + "\n"


def _error(status: int, message, metrics: Metrics) -> JSONResponse:
    metrics.requests[status] += 1
    return JSONResponse({"error": message}, status_code=status)


def create_app(service: ScoringService, api_key: str | None = None) -> Starlette:
    metrics = Metrics()

    def authorised(request: Request) -> bool:
        if not api_key:
            return True
        supplied = request.headers.get("x-api-key", "")
        return hmac.compare_digest(supplied.encode(), api_key.encode())

    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def ready(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ready", "model_version": service.version})

    async def predict(request: Request) -> JSONResponse:
        started = time.perf_counter()
        if not authorised(request):
            return _error(401, "invalid or missing API key", metrics)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY:
                return _error(413, "request body too large", metrics)
        try:
            txn = Transaction.model_validate_json(bytes(body))
        except ValidationError as exc:
            details = exc.errors(include_url=False, include_input=False, include_context=False)
            return _error(422, details, metrics)
        # A far-future timestamp would block all later requests for this customer.
        if txn.timestamp > datetime.now(UTC) + MAX_CLOCK_SKEW:
            return _error(422, "timestamp is in the future", metrics)
        try:
            result = await run_in_threadpool(service.score, txn)
        except (OutOfOrderError, DuplicateConflictError) as exc:
            return _error(409, str(exc), metrics)
        except Exception:
            log.exception("scoring failed")
            return _error(500, "internal error", metrics)
        elapsed = time.perf_counter() - started
        metrics.observe(elapsed)
        metrics.requests[200] += 1
        if not result["duplicate"]:
            metrics.decisions[result["decision"]] += 1
        # Log identifiers and outcomes only, never the raw payload.
        log.info(json.dumps({"transaction_id": txn.transaction_id,
                             "decision": result["decision"],
                             "latency_ms": round(elapsed * 1000, 2)}))
        return JSONResponse(result)

    async def prometheus(request: Request) -> PlainTextResponse:
        if not authorised(request):
            return PlainTextResponse("unauthorised", status_code=401)
        return PlainTextResponse(metrics.render())

    async def drift(request: Request) -> JSONResponse:
        if not authorised(request):
            return _error(401, "invalid or missing API key", metrics)
        return JSONResponse(await run_in_threadpool(service.drift))

    return Starlette(routes=[
        Route("/health", health),
        Route("/ready", ready),
        Route("/v1/predict", predict, methods=["POST"]),
        Route("/v1/drift", drift),
        Route("/metrics", prometheus),
    ])


def app_factory() -> Starlette:
    """Entry point for `uvicorn --factory`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    service = ScoringService.from_registry(
        os.getenv("FRAUDSHIELD_MODELS", "models"),
        state_db=os.getenv("FRAUDSHIELD_STATE_DB"),
    )
    return create_app(service, os.getenv("FRAUDSHIELD_API_KEY"))
