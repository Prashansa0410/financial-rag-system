.PHONY: help install test serve docker-build docker-run clean

help:
	@echo "Available commands:"
	@echo "  make install      Install dependencies"
	@echo "  make test         Run unit and integration tests"
	@echo "  make serve        Run FastAPI development server"
	@echo "  make search Q=..  Execute CLI search for query Q"
	@echo "  make docker-build Build Docker container"
	@echo "  make docker-run   Run Docker container"
	@echo "  make clean        Remove build and cache artifacts"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest tests/ -v

serve:
	python main.py serve --port 8000 --reload

docker-build:
	docker compose build

docker-run:
	docker compose up

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
