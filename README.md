# MLOps Assignment 01 - Heart Disease Risk Prediction

**Course:** Machine Learning Operations AIMLCZG523  
**Topic:** End-to-End ML Model Development, CI/CD, and Production Deployment  
**Project:** Heart Disease Risk Prediction using the official UCI Heart Disease dataset

This repository is a production-style academic MLOps project. It contains raw data acquisition, EDA, preprocessing, model training, MLflow tracking, FastAPI serving, Docker containerization, CI/CD, Kubernetes deployment files, monitoring assets, reports, and video-recording guidance.

> **Raw-only data policy:** This version intentionally trains the model only from the original unprocessed UCI 76-attribute source files: `cleveland.data`, `hungarian.data`, `switzerland.data`, and `long-beach-va.data`. The `processed.*` files are not used for model development.

---

## 1. Architecture

```text
Official UCI Heart Disease raw 76-attribute files
  -> src/data_acquisition.py
  -> data/raw/uci_full_archive/ with raw files, metadata, and Costs folder
  -> extract 14 documented ML variables directly from raw 76-attribute records
  -> data/processed/heart_disease_raw_76_selected14.csv
  -> data/processed/heart_disease_raw_only_all_sites_cleaned.csv
  -> notebooks/01_data_acquisition_eda.ipynb
  -> sklearn preprocessing pipeline
  -> model training and tuning
  -> MLflow experiment tracking
  -> models/final_model.joblib
  -> FastAPI inference service
  -> Docker container
  -> GitHub Actions CI/CD
  -> Render/Railway public deployment or local Kubernetes deployment
  -> Logging and Prometheus metrics
```

---

## 2. Data Scope - Raw Data Only

The acquisition script downloads these official UCI raw/metadata files into `data/raw/uci_full_archive/`:

```text
heart-disease.names
Index
WARNING
ask-detrano
bak
cleve
cleveland.data
hungarian.data
switzerland.data
long-beach-va.data
new.data
Costs/heart-disease.cost
Costs/heart-disease.delay
Costs/heart-disease.expense
Costs/heart-disease.group
Costs/heart-disease.README
Costs/Index
```

The model uses only these four raw institutional databases:

```text
cleveland.data
hungarian.data
switzerland.data
long-beach-va.data
```

`new.data` is downloaded for traceability, but it is not used for training because it is not one of the four named institutional raw database files. The `Costs` folder is also downloaded for completeness but is not patient-level training data.

The raw files contain 76 attributes. The pipeline extracts the 14 documented ML variables directly from the raw records:

```text
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
oldpeak, slope, ca, thal, num
```

Then `num` is converted into the binary target:

```text
0 = no heart disease
1 = heart disease risk present, from original values 1, 2, 3, or 4
```

---

## 3. Quick Start - Local Python

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Download official UCI raw files and build raw-only cleaned dataset
python src/data_acquisition.py

# 4. Train and evaluate models with MLflow tracking
python src/train.py

# 5. Run the API locally
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## 4. Generated Data Files

After running `python src/data_acquisition.py`, check these outputs:

```text
data/raw/uci_full_archive/download_manifest.json
data/raw/uci_full_archive/EXPECTED_UCI_RAW_FILES.txt
data/processed/heart_disease_raw_76_selected14.csv
data/processed/heart_disease_raw_76_selected14.profile.json
data/processed/heart_disease_raw_only_all_sites_cleaned.csv
data/processed/heart_disease_raw_only_all_sites_cleaned.profile.json
```

---

## 5. API Test

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3
  }'
```

Expected response pattern:

```json
{
  "prediction": 1,
  "risk_label": "Heart disease risk present",
  "confidence": 0.87,
  "model_name": "RandomForestClassifier",
  "model_version": "v1.2-raw-76-only"
}
```

Actual probability depends on the final trained model.

---

## 6. MLflow

```bash
mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

Capture screenshots in `screenshots/mlflow/`:

- `mlflow_experiment_list.png`
- `mlflow_best_run_metrics.png`
- `mlflow_artifacts_confusion_matrix.png`
- `mlflow_artifacts_roc_curve.png`

---

## 7. Automated Tests

```bash
pytest -q
ruff check src api tests
```

CI/CD is configured in `.github/workflows/ci-cd.yml`.

---

## 8. Docker

Build and run:

```bash
docker build -t heart-disease-mlops:latest .
docker run --rm -p 8000:8000 heart-disease-mlops:latest
```

The Dockerfile runs raw-only data acquisition and training during image build, so Docker needs outbound internet access during build unless you pre-run data acquisition and training locally.

---

## 9. Render Deployment

Recommended academic public deployment option: **Render Web Service**.

1. Push this repository to GitHub.
2. Create a new Render Web Service from the GitHub repo.
3. Use the Docker environment.
4. Test `/health`, `/docs`, `/predict`, and `/metrics` after deployment.

---

## 10. Kubernetes Local Deployment

```bash
docker build -t heart-disease-mlops:latest .
minikube start
minikube image load heart-disease-mlops:latest
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
kubectl get pods
kubectl port-forward service/heart-disease-service 8000:80
```

Test:

```bash
curl http://127.0.0.1:8000/health
```

---

## 11. Evidence to Capture

Use screenshots from real execution only:

```text
screenshots/eda/
screenshots/mlflow/
screenshots/github_actions/
screenshots/docker/
screenshots/deployment/
screenshots/api_testing/
screenshots/monitoring/
```

Do not fabricate screenshots, metrics, deployment URLs, or MLflow runs.


## Downloader note

The raw-only downloader uses the official four raw 76-attribute institutional files for model training: `cleveland.data`, `hungarian.data`, `switzerland.data`, and `long-beach-va.data`. Test-cost metadata is downloaded from the lowercase UCI `costs/` path. The historical `cleve` MOD helper file is not required for training and is not part of the mandatory raw-only modeling workflow.

## Data Acquisition and EDA deliverables

Task 1 files are located here:

- `src/data_acquisition.py` downloads and prepares the raw-only UCI dataset.
- `src/eda.py` generates EDA tables and professional visualization files.
- `notebooks/01_data_acquisition_eda.ipynb` shows the full notebook workflow.
- `docs/data_acquisition_eda_guide.md` explains the Task 1 evidence mapping.

Run:

```powershell
python src/data_acquisition.py
python src/eda.py
```

Then check:

```text
artifacts/eda/
artifacts/figures/eda/
screenshots/eda/
```
