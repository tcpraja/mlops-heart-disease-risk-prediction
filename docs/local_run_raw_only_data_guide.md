# Local Run Guide - Raw UCI Data Files Only

This guide runs the project using only the original unprocessed UCI Heart Disease raw files.

## 1. Extract and open the project

```powershell
cd mlops-heart-disease
```

## 2. Create and activate environment

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Download raw UCI files and create cleaned raw-only dataset

```powershell
python src/data_acquisition.py
```

Expected raw archive:

```text
data/raw/uci_full_archive/cleveland.data
data/raw/uci_full_archive/hungarian.data
data/raw/uci_full_archive/switzerland.data
data/raw/uci_full_archive/long-beach-va.data
data/raw/uci_full_archive/new.data
data/raw/uci_full_archive/Costs/
```

Expected processed outputs:

```text
data/processed/heart_disease_raw_76_selected14.csv
data/processed/heart_disease_raw_only_all_sites_cleaned.csv
```

The `processed.*` UCI files are not used in this raw-only version.

## 4. Train the model

```powershell
python src/train.py
```

Expected artifacts:

```text
models/final_model.joblib
models/preprocessing_pipeline.joblib
models/model_metadata.json
models/model_comparison.csv
artifacts/figures/
mlruns/
```

## 5. Run tests

```powershell
pytest -q
ruff check src api tests
```

## 6. Run API locally

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
http://127.0.0.1:8000/metrics
```

## 7. Test prediction

Use this JSON in Swagger UI `/predict`:

```json
{
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
}
```

## 8. Run MLflow

Open a second terminal:

```powershell
.venv\Scripts\activate
mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

## 9. Docker

```powershell
docker build -t heart-disease-mlops:latest .
docker run --rm -p 8000:8000 heart-disease-mlops:latest
```

## 10. Screenshot evidence

Capture only real screenshots after execution:

```text
screenshots/eda/
screenshots/mlflow/
screenshots/api_testing/
screenshots/docker/
screenshots/deployment/
screenshots/monitoring/
```


## Downloader note

The raw-only downloader uses the official four raw 76-attribute institutional files for model training: `cleveland.data`, `hungarian.data`, `switzerland.data`, and `long-beach-va.data`. Test-cost metadata is downloaded from the lowercase UCI `costs/` path. The historical `cleve` MOD helper file is not required for training and is not part of the mandatory raw-only modeling workflow.
