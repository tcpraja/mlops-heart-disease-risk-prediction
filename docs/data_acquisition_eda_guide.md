# Data Acquisition and EDA Guide

This file explains where to find and how to run the Task 1 deliverables.

## Task 1 requirement

Data Acquisition and Exploratory Data Analysis requires:

- dataset download script or instructions,
- raw and cleaned dataset structure,
- missing value analysis,
- data type inspection,
- class balance analysis,
- histograms,
- correlation heatmap,
- feature relationship analysis,
- professional visualizations,
- short interpretation of EDA findings.

## Where the files are located

| Requirement | File or folder |
|---|---|
| Download script | `src/data_acquisition.py` |
| Raw-only EDA script | `src/eda.py` |
| EDA notebook | `notebooks/01_data_acquisition_eda.ipynb` |
| Raw UCI files after running | `data/raw/uci_full_archive/` |
| Cleaned modeling dataset | `data/processed/heart_disease_raw_only_all_sites_cleaned.csv` |
| Dataset profile | `data/processed/heart_disease_raw_only_all_sites_cleaned.profile.json` |
| EDA tables | `artifacts/eda/` |
| EDA figures | `artifacts/figures/eda/` |
| Report-ready screenshot copies | `screenshots/eda/` |

## Commands to run

From the project root:

```powershell
python src/data_acquisition.py
python src/eda.py
```

The first command downloads the official UCI raw data files and creates the cleaned raw-only modeling dataset.

The second command creates the EDA evidence files and figures.

## EDA outputs created

After running `python src/eda.py`, check these folders:

```text
artifacts/eda/
artifacts/figures/eda/
screenshots/eda/
```

Expected EDA figures:

```text
01_class_balance.png
02_source_distribution.png
03_feature_histograms.png
04_correlation_heatmap.png
05_thalach_by_target.png
06_age_cholesterol_by_target.png
```

Expected EDA tables:

```text
data_types.csv
missing_values.csv
class_balance.csv
source_distribution.csv
correlation_matrix.csv
descriptive_statistics.csv
eda_summary.md
```

## Notebook option

Open this notebook in VS Code or Jupyter:

```text
notebooks/01_data_acquisition_eda.ipynb
```

Run all cells from top to bottom. The notebook calls both `src/data_acquisition.py` and `src/eda.py`.

## What to screenshot for marks

Capture these for the final report:

1. successful `python src/data_acquisition.py` output,
2. cleaned dataset shape and target distribution,
3. missing values table,
4. class balance plot,
5. histograms,
6. correlation heatmap,
7. source database distribution,
8. feature relationship plots.

Save them in:

```text
screenshots/eda/
```
