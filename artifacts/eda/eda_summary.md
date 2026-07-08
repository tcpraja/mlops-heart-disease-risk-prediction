# Data Acquisition and EDA Summary

## Dataset acquisition
The project uses a raw-only data policy. The model dataset is created from the original UCI 76-attribute raw files: `cleveland.data`, `hungarian.data`, `switzerland.data`, and `long-beach-va.data`. The script `src/data_acquisition.py` downloads the official files, extracts the 14 documented ML variables, handles missing markers, and converts the diagnosis target into binary classification.

## Cleaned dataset profile
- Cleaned dataset path: `C:\Users\ASUS\MLOPS\mlops-heart-disease-v3\data\processed\heart_disease_raw_only_all_sites_cleaned.csv`
- Shape: 907 rows x 15 columns
- Source distribution: {'hungarian': 294, 'cleveland': 290, 'long_beach_va': 200, 'switzerland': 123}
- Target distribution: {0: 405, 1: 502}
- Total missing values after cleaning: 1787

## EDA outputs generated
- `artifacts/eda/missing_values.csv`
- `artifacts/eda/data_types.csv`
- `artifacts/eda/class_balance.csv`
- `artifacts/eda/correlation_matrix.csv`
- `artifacts/figures/eda/01_class_balance.png`
- `artifacts/figures/eda/02_source_distribution.png`
- `artifacts/figures/eda/03_feature_histograms.png`
- `artifacts/figures/eda/04_correlation_heatmap.png`
- `artifacts/figures/eda/05_thalach_by_target.png`
- `artifacts/figures/eda/06_age_cholesterol_by_target.png`

The same PNG files are also copied to `screenshots/eda/` for direct insertion into the final report.
