# Model Artifact Instructions

Model artifacts are intentionally not committed before execution.

Run:

```bash
python src/data_acquisition.py
python src/train.py
```

The pipeline uses only the original raw UCI 76-attribute files for training:

```text
cleveland.data
hungarian.data
switzerland.data
long-beach-va.data
```

After training, this folder should contain:

```text
final_model.joblib
preprocessing_pipeline.joblib
model_metadata.json
model_comparison.csv
```

The Dockerfile also runs raw-only data acquisition and training during image build so deployment artifacts are produced from the original raw UCI files.
