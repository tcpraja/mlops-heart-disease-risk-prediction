# Data Instructions - Raw UCI Files Only

This project uses only the original unprocessed UCI Heart Disease raw data files for model development.

## Raw files downloaded

The script `src/data_acquisition.py` downloads raw UCI files into:

```text
data/raw/uci_full_archive/
```

Model training uses these four raw 76-attribute files:

```text
cleveland.data
hungarian.data
switzerland.data
long-beach-va.data
```

The script also downloads metadata and Costs files for traceability:

```text
heart-disease.names
Index
WARNING
ask-detrano
bak
cleve
new.data
Costs/
```

## Processed UCI files are not used

Files such as `processed.cleveland.data`, `processed.hungarian.data`, `processed.switzerland.data`, and `processed.va.data` are intentionally not used in this raw-only version.

## Generated files

After running:

```bash
python src/data_acquisition.py
```

the project creates:

```text
data/processed/heart_disease_raw_76_selected14.csv
data/processed/heart_disease_raw_only_all_sites_cleaned.csv
```

The first file contains the 14 documented ML variables extracted directly from the raw 76-attribute records. The second file is cleaned and ready for EDA, model training, testing, API deployment, and MLflow tracking.

## Target conversion

The original UCI `num` field is converted as follows:

```text
0 = no heart disease
1, 2, 3, 4 = heart disease risk present
```
