.PHONY: setup data eda train test lint api docker-build docker-run mlflow

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

data:
	python src/data_acquisition.py

eda:
	python src/eda.py

train:
	python src/train.py

test:
	pytest -q

lint:
	ruff check src api tests

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

mlflow:
	mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000

docker-build:
	docker build -t heart-disease-mlops:latest .

docker-run:
	docker run --rm -p 8000:8000 heart-disease-mlops:latest
