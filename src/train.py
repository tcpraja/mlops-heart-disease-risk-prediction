from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.config import (
    CLEAN_DATA_PATH,
    FEATURE_COLUMNS,
    FIGURE_DIR,
    METADATA_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_PATH,
    MODEL_VERSION,
    PREPROCESSOR_PATH,
    RANDOM_SEED,
    SOURCE_COLUMN,
    TARGET_COLUMN,
)
from src.data_acquisition import main as acquire_data
from src.preprocessing import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def safe_mlflow_import():
    try:
        import mlflow
        import mlflow.sklearn

        return mlflow
    except Exception as exc:  # pragma: no cover - depends on optional install
        LOGGER.warning("MLflow unavailable; training will continue without tracking: %s", exc)
        return None


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    if not CLEAN_DATA_PATH.exists():
        LOGGER.info("Cleaned dataset not found. Running data acquisition first.")
        acquire_data()
    df = pd.read_csv(CLEAN_DATA_PATH)
    return df[FEATURE_COLUMNS], df[TARGET_COLUMN]


def build_model_searches() -> dict[str, GridSearchCV]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    logistic_pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED),
            ),
        ]
    )

    rf_pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED),
            ),
        ]
    )

    svm_pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                SVC(probability=True, class_weight="balanced", random_state=RANDOM_SEED),
            ),
        ]
    )

    return {
        "LogisticRegression": GridSearchCV(
            logistic_pipe,
            param_grid={
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__solver": ["lbfgs"],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
        "RandomForestClassifier": GridSearchCV(
            rf_pipe,
            param_grid={
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 4, 6],
                "classifier__min_samples_leaf": [1, 3, 5],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
        "SupportVectorMachine": GridSearchCV(
            svm_pipe,
            param_grid={
                "classifier__C": [0.5, 1.0, 2.0],
                "classifier__kernel": ["rbf", "linear"],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
    }


def save_plots(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, prefix: str) -> dict[str, str]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]

    cm_path = FIGURE_DIR / f"{prefix}_confusion_matrix.png"
    ConfusionMatrixDisplay(confusion_matrix(y_test, preds)).plot(cmap="Blues")
    plt.title(f"{prefix} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=160)
    plt.close()

    roc_path = FIGURE_DIR / f"{prefix}_roc_curve.png"
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test, probs):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=160)
    plt.close()

    return {"confusion_matrix": str(cm_path), "roc_curve": str(roc_path)}


def save_eda_figures(df: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x=TARGET_COLUMN)
    plt.title("Heart Disease Target Class Balance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_class_balance.png", dpi=160)
    plt.close()

    df[FEATURE_COLUMNS].hist(figsize=(14, 10), bins=20)
    plt.suptitle("Feature Histograms", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_histograms.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr(numeric_only=True), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_correlation_heatmap.png", dpi=160)
    plt.close()


def main() -> None:
    x, y = load_training_data()
    source_summary = {}
    if CLEAN_DATA_PATH.exists():
        loaded_df = pd.read_csv(CLEAN_DATA_PATH)
        if SOURCE_COLUMN in loaded_df.columns:
            source_summary = {str(k): int(v) for k, v in loaded_df[SOURCE_COLUMN].value_counts().to_dict().items()}
    full_df = pd.concat([x, y], axis=1)
    save_eda_figures(full_df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    mlflow = safe_mlflow_import()
    if mlflow:
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    results = []
    searches = build_model_searches()
    best_model_name = None
    best_model = None
    best_priority_tuple = None

    for model_name, search in searches.items():
        LOGGER.info("Training %s", model_name)
        run_context = mlflow.start_run(run_name=model_name) if mlflow else None
        try:
            search.fit(x_train, y_train)
            model = search.best_estimator_
            metrics = evaluate_model(model, x_test, y_test)
            plots = save_plots(model, x_test, y_test, model_name)
            report_text = classification_report(y_test, model.predict(x_test), zero_division=0)
            report_path = FIGURE_DIR / f"{model_name}_classification_report.txt"
            report_path.write_text(report_text, encoding="utf-8")

            row = {
                "model": model_name,
                "best_params": json.dumps(search.best_params_),
                "best_cv_roc_auc": search.best_score_,
                **metrics,
            }
            results.append(row)

            if mlflow:
                mlflow.log_param("model_type", model_name)
                mlflow.log_params(search.best_params_)
                mlflow.log_metric("best_cv_roc_auc", float(search.best_score_))
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, float(metric_value))
                mlflow.log_artifact(plots["confusion_matrix"])
                mlflow.log_artifact(plots["roc_curve"])
                mlflow.log_artifact(str(report_path))
                mlflow.sklearn.log_model(model, artifact_path="model")

            priority_tuple = (metrics["recall"], metrics["roc_auc"], metrics["f1"])
            if best_priority_tuple is None or priority_tuple > best_priority_tuple:
                best_priority_tuple = priority_tuple
                best_model_name = model_name
                best_model = model
        finally:
            if run_context:
                mlflow.end_run()

    if best_model is None or best_model_name is None:
        raise RuntimeError("No model was trained successfully")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(best_model.named_steps["preprocessor"], PREPROCESSOR_PATH)

    results_df = pd.DataFrame(results).sort_values(
        by=["recall", "roc_auc", "f1"], ascending=False
    )
    results_df.to_csv(MODEL_PATH.parent / "model_comparison.csv", index=False)

    metadata = {
        "model_name": best_model_name,
        "model_version": MODEL_VERSION,
        "dataset_source": "Only the original raw 76-attribute UCI Heart Disease files: cleveland.data, hungarian.data, switzerland.data and long-beach-va.data.",
        "raw_archive_note": "The model-ready dataset is extracted directly from raw 76-attribute files into data/processed/heart_disease_raw_76_selected14.csv and cleaned into data/processed/heart_disease_raw_only_all_sites_cleaned.csv. Processed.* files are intentionally not used for training.",
        "source_distribution": source_summary,
        "selection_rule": "Prioritized recall, then ROC-AUC, then F1 due to healthcare-style false-negative risk.",
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(MODEL_PATH),
        "preprocessor_path": str(PREPROCESSOR_PATH),
        "metrics": results_df.iloc[0].drop(labels=["best_params"]).to_dict(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    LOGGER.info("Saved final model to %s", MODEL_PATH)
    LOGGER.info("Saved metadata to %s", METADATA_PATH)
    LOGGER.info("Best model: %s", best_model_name)


if __name__ == "__main__":
    main()
