.PHONY: install test coverage lint format typecheck check-examples run-examples lock docker-build docker-run clean

VENV := .venv
PYTHON := $(VENV)/bin/python

install:
	uv venv --python 3.14 $(VENV)
	uv pip install --python $(PYTHON) -r config/requirements-dev.txt -e '.[python,sql]'

test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing --cov-fail-under=90

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

format:
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .

typecheck:
	$(VENV)/bin/mypy

check-examples:
	$(VENV)/bin/eigrel check examples/*.eig

run-examples:
	$(VENV)/bin/eigrel run examples/ml.eig
	$(VENV)/bin/eigrel run examples/regression.eig
	$(VENV)/bin/eigrel run examples/sql.eig

lock:
	uv pip compile config/requirements.txt --output-file=config/requirements.lock --universal

docker-build:
	docker build -f docker/Dockerfile -t eigrel .

docker-run:
	docker run --rm --user "$$(id -u):$$(id -g)" -v $(CURDIR)/examples:/work eigrel run ml.eig

clean:
	find . -type d -name __pycache__ -not -path './$(VENV)/*' -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage build dist src/*.egg-info
