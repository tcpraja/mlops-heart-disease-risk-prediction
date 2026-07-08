# Final Report - Raw UCI Heart Disease MLOps Project

This report documents a production-style MLOps assignment for Heart Disease Risk Prediction using only the original unprocessed UCI 76-attribute raw files. The model development workflow deliberately excludes `processed.*` UCI files. The acquisition pipeline downloads the raw source files, extracts the 14 documented ML variables directly from the 76-attribute records, converts the original diagnosis field into a binary target, trains multiple sklearn classifiers, tracks experiments with MLflow, serves predictions through FastAPI, containerizes the service with Docker, and provides CI/CD and deployment assets.

## Raw-only data scope

Model training uses only:

- `cleveland.data`
- `hungarian.data`
- `switzerland.data`
- `long-beach-va.data`

The `new.data`, metadata files, and `Costs/` folder are downloaded for traceability but are not used as patient training records. The cleaned model dataset is saved as `data/processed/heart_disease_raw_only_all_sites_cleaned.csv`.

## Target conversion

The original UCI `num` diagnosis field is converted into binary classification:

- `0`: no heart disease
- `1`: heart disease risk present, based on original values `1`, `2`, `3`, or `4`

## MLOps components

The repository includes reusable source code, notebooks, MLflow experiment tracking, pytest tests, FastAPI endpoints, Dockerfile, GitHub Actions workflow, Kubernetes manifests, monitoring assets, and report/video guidance. Real screenshots and deployment URLs must be captured after local or cloud execution; they must not be fabricated.
