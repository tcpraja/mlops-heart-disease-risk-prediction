# Implementation Plan Mapped to Marks

| Task | Marks | Implementation files | Validation |
|---|---:|---|---|
| Data acquisition and EDA | 5 | `src/data_acquisition.py`, `notebooks/01_data_acquisition_eda.ipynb` | Downloads official raw UCI files, extracts selected 14 variables directly from 76-attribute raw records, builds raw-only cleaned dataset |
| Feature engineering and modeling | 8 | `src/preprocessing.py`, `src/train.py` | `python src/train.py` |
| Experiment tracking | 5 | `src/train.py`, `mlruns/` | `mlflow ui --backend-store-uri mlruns` |
| Packaging and reproducibility | 7 | `requirements.txt`, `env.yml`, `config.yaml`, `models/` | fresh environment run |
| CI/CD and testing | 8 | `tests/`, `.github/workflows/ci-cd.yml` | `pytest -q`, GitHub Actions |
| Containerization | 5 | `api/`, `Dockerfile` | `docker build`, `docker run` |
| Production deployment | 7 | `render.yaml`, `railway.json`, `deployment/*.yaml` | public URL or Minikube |
| Monitoring and logging | 3 | `api/main.py`, `monitoring/` | `/metrics`, logs |
| Documentation and reporting | 2 | `reports/`, `README.md`, `docs/` | final report review |
