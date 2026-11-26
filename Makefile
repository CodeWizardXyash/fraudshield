.PHONY: setup data train test lint serve demo monitor docker

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

demo:
	python scripts/demo.py

monitor:
	fraudshield monitor --data data/transactions.csv.gz --since 2026-06-01

docker:
	docker compose up --build
