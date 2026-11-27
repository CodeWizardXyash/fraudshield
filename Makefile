.PHONY: setup data train test lint serve demo monitor tune-threshold docker

setup:
	pip install -r requirements-dev.txt && pip install --no-deps -e .

data:
	fraudshield simulate

train:
	fraudshield train

test:
	pytest -q

lint:
	ruff check src tests scripts

serve:
	fraudshield serve

serve-persist:
	fraudshield serve --state-db state/customers.db

demo:
	python scripts/demo.py

monitor:
	fraudshield monitor --data data/transactions.csv.gz --since 2026-06-01

tune-threshold:
	fraudshield tune-threshold

docker:
	docker compose up --build
