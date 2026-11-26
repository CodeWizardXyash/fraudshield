"""Black-box tests against a real uvicorn server."""

import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
import uvicorn

from conftest import make_txn
from fraudshield.serving.api import create_app
from fraudshield.serving.service import ScoringService

KEY = {"x-api-key": "test-key"}


@pytest.fixture(scope="module")
def server(trained):
    service = ScoringService.from_registry(trained[0] / "models")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = uvicorn.Config(create_app(service, "test-key"), port=port, log_level="error")
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not instance.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    instance.should_exit = True
    thread.join(timeout=5)


def post(url, txn, headers=KEY):
    return requests.post(f"{url}/v1/predict", data=txn.model_dump_json(), headers=headers,
                         timeout=10)


def test_health_and_ready(server):
    assert requests.get(f"{server}/health", timeout=5).json() == {"status": "ok"}
    assert requests.get(f"{server}/ready", timeout=5).json()["model_version"]


def test_predict_and_idempotent_retry(server):
    txn = make_txn(100, customer_id="api-1")
    first = post(server, txn).json()
    assert 0 <= first["fraud_probability"] <= 1
    assert first["decision"] in {"approve", "review"}
    retry = post(server, txn).json()
    assert retry["duplicate"] and retry["fraud_probability"] == first["fraud_probability"]
    changed = make_txn(100, customer_id="api-1", amount=999)
    assert post(server, changed).status_code == 409


def test_out_of_order(server):
    assert post(server, make_txn(200, 30, customer_id="api-2")).status_code == 200
    assert post(server, make_txn(201, 0, customer_id="api-2")).status_code == 409


def test_errors(server):
    url = f"{server}/v1/predict"
    assert post(server, make_txn(300), headers={}).status_code == 401
    assert requests.post(url, data=b"{bad", headers=KEY, timeout=5).status_code == 422
    assert requests.post(url, data=b"x" * 20_000, headers=KEY, timeout=5).status_code == 413
    future = make_txn(301, timestamp="2099-01-01T00:00:00Z")
    assert post(server, future).status_code == 422


def test_concurrent_requests(server):
    txns = [make_txn(1000 + i, customer_id=f"p{i}") for i in range(40)]
    with ThreadPoolExecutor(8) as pool:
        codes = [r.status_code for r in pool.map(lambda t: post(server, t), txns)]
    assert codes == [200] * 40


def test_metrics_and_drift(server):
    assert requests.get(f"{server}/metrics", timeout=5).status_code == 401
    text = requests.get(f"{server}/metrics", headers=KEY, timeout=5).text
    assert "fraudshield_latency_seconds_count" in text
    drift = requests.get(f"{server}/v1/drift", headers=KEY, timeout=5).json()
    assert drift["status"] == "insufficient_data"
