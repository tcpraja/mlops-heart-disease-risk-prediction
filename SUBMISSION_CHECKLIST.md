# Submission Checklist - Raw UCI Data Only

Use this checklist before final submission.

## Data

- [ ] Ran `python src/data_acquisition.py`
- [ ] Confirmed raw files exist in `data/raw/uci_full_archive/`
- [ ] Confirmed `cleveland.data`, `hungarian.data`, `switzerland.data`, and `long-beach-va.data` are downloaded
- [ ] Confirmed `data/processed/heart_disease_raw_76_selected14.csv` exists
- [ ] Confirmed `data/processed/heart_disease_raw_only_all_sites_cleaned.csv` exists
- [ ] Confirmed model does not train on `processed.*` files

## Model and MLflow

- [ ] Ran `python src/train.py`
- [ ] Confirmed `models/final_model.joblib`
- [ ] Confirmed `models/preprocessing_pipeline.joblib`
- [ ] Confirmed `models/model_metadata.json`
- [ ] Captured MLflow screenshots

## Tests and CI/CD

- [ ] Ran `pytest -q`
- [ ] Ran `ruff check src api tests`
- [ ] Pushed project to GitHub
- [ ] Captured GitHub Actions success screenshot

## API

- [ ] Ran `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000`
- [ ] Tested `/health`
- [ ] Tested `/docs`
- [ ] Tested `/predict`
- [ ] Tested `/metrics`

## Docker and Deployment

- [ ] Ran `docker build -t heart-disease-mlops:latest .`
- [ ] Ran `docker run --rm -p 8000:8000 heart-disease-mlops:latest`
- [ ] Tested API from Docker
- [ ] Completed Render/Railway or Kubernetes proof

## Report

- [ ] Updated report with real screenshots
- [ ] Added GitHub repository URL
- [ ] Added local or public API URL
- [ ] Clearly stated raw-only UCI data policy
- [ ] Did not fabricate results, screenshots, URLs, or metrics


## Downloader note

The raw-only downloader uses the official four raw 76-attribute institutional files for model training: `cleveland.data`, `hungarian.data`, `switzerland.data`, and `long-beach-va.data`. Test-cost metadata is downloaded from the lowercase UCI `costs/` path. The historical `cleve` MOD helper file is not required for training and is not part of the mandatory raw-only modeling workflow.

## Task 1 Data Acquisition and EDA checklist

Run these commands before collecting screenshots:

```powershell
python src/data_acquisition.py
python src/eda.py
```

Confirm these files exist:

- [ ] `data/processed/heart_disease_raw_only_all_sites_cleaned.csv`
- [ ] `artifacts/eda/missing_values.csv`
- [ ] `artifacts/eda/data_types.csv`
- [ ] `artifacts/eda/class_balance.csv`
- [ ] `artifacts/eda/correlation_matrix.csv`
- [ ] `artifacts/figures/eda/01_class_balance.png`
- [ ] `artifacts/figures/eda/02_source_distribution.png`
- [ ] `artifacts/figures/eda/03_feature_histograms.png`
- [ ] `artifacts/figures/eda/04_correlation_heatmap.png`
- [ ] `artifacts/figures/eda/05_thalach_by_target.png`
- [ ] `artifacts/figures/eda/06_age_cholesterol_by_target.png`

Use `notebooks/01_data_acquisition_eda.ipynb` to show the complete EDA workflow in notebook form.
