# Commands Runbook - Raw UCI Data Only

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
.venv\Scripts\activate
```

## Data acquisition

```bash
python src/data_acquisition.py
```

This downloads the official raw UCI Heart Disease files into `data/raw/uci_full_archive/`, extracts the 14 documented ML variables from the raw 76-attribute files, and saves:

```text
data/processed/heart_disease_raw_76_selected14.csv
data/processed/heart_disease_raw_only_all_sites_cleaned.csv
```

The `processed.*` files are intentionally not used for model training.

## Training

```bash
python src/train.py
```

## Testing

```bash
pytest -q
ruff check src api tests
```

## API

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

## MLflow

```bash
mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
```

## Docker

```bash
docker build -t heart-disease-mlops:latest .
docker run --rm -p 8000:8000 heart-disease-mlops:latest
```

## Kubernetes

```bash
minikube start
minikube image load heart-disease-mlops:latest
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
kubectl get pods
kubectl port-forward service/heart-disease-service 8000:80
```
