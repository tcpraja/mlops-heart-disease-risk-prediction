"""Exploratory Data Analysis for the raw-only UCI Heart Disease MLOps project.

This script generates the evidence required for Task 1: Data Acquisition and EDA.
It assumes the raw-only data acquisition pipeline has created the cleaned modeling
file. If the cleaned file is missing, it runs data acquisition first.

Outputs are saved to:
- artifacts/figures/eda/
- screenshots/eda/
- artifacts/eda/eda_summary.md
- artifacts/eda/missing_values.csv
- artifacts/eda/class_balance.csv
- artifacts/eda/correlation_matrix.csv
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    ARTIFACT_DIR,
    CLEAN_DATA_PATH,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    SOURCE_COLUMN,
    TARGET_COLUMN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

EDA_DIR = ARTIFACT_DIR / "eda"
EDA_FIGURE_DIR = ARTIFACT_DIR / "figures" / "eda"
SCREENSHOT_EDA_DIR = PROJECT_ROOT / "screenshots" / "eda"


def ensure_clean_data_exists() -> None:
    """Run data acquisition if the cleaned raw-only dataset is missing."""
    if CLEAN_DATA_PATH.exists() and CLEAN_DATA_PATH.stat().st_size > 0:
        return

    LOGGER.info("Clean dataset not found. Running data acquisition first...")
    from src.data_acquisition import main as acquire_data

    acquire_data()


def save_figure(filename: str) -> Path:
    """Save the active Matplotlib figure to artifact and screenshot folders."""
    EDA_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_EDA_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = EDA_FIGURE_DIR / filename
    screenshot_path = SCREENSHOT_EDA_DIR / filename
    plt.tight_layout()
    plt.savefig(artifact_path, dpi=180, bbox_inches="tight")
    shutil.copy2(artifact_path, screenshot_path)
    plt.close()
    LOGGER.info("Saved EDA figure: %s", artifact_path)
    return artifact_path


def load_dataset() -> pd.DataFrame:
    """Load the cleaned raw-only all-site UCI modeling dataset."""
    ensure_clean_data_exists()
    df = pd.read_csv(CLEAN_DATA_PATH)
    LOGGER.info("Loaded EDA dataset from %s with shape %s", CLEAN_DATA_PATH, df.shape)
    return df


def create_profile_tables(df: pd.DataFrame) -> None:
    """Save EDA profile tables required for academic evidence."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    df.dtypes.rename("dtype").astype(str).to_csv(EDA_DIR / "data_types.csv")
    df.isna().sum().sort_values(ascending=False).rename("missing_count").to_csv(EDA_DIR / "missing_values.csv")
    df[TARGET_COLUMN].value_counts().rename("count").to_csv(EDA_DIR / "class_balance.csv")

    if SOURCE_COLUMN in df.columns:
        df[SOURCE_COLUMN].value_counts().rename("count").to_csv(EDA_DIR / "source_distribution.csv")

    numeric_df = df.select_dtypes(include="number")
    numeric_df.corr().to_csv(EDA_DIR / "correlation_matrix.csv")
    df.describe(include="all").transpose().to_csv(EDA_DIR / "descriptive_statistics.csv")


def plot_class_balance(df: pd.DataFrame) -> None:
    """Class balance plot for binary target distribution."""
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x=TARGET_COLUMN)
    ax.set_title("Heart disease target class balance")
    ax.set_xlabel("Target class: 0 = no disease, 1 = disease risk present")
    ax.set_ylabel("Patient count")

    for container in ax.containers:
        ax.bar_label(container)

    save_figure("01_class_balance.png")


def plot_source_distribution(df: pd.DataFrame) -> None:
    """Plot patient record distribution across the four raw UCI source databases."""
    if SOURCE_COLUMN not in df.columns:
        return

    source_counts = df[SOURCE_COLUMN].value_counts().reset_index()
    source_counts.columns = [SOURCE_COLUMN, "count"]

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=source_counts, y=SOURCE_COLUMN, x="count", orient="h")
    ax.set_title("Raw UCI source database distribution")
    ax.set_xlabel("Patient count")
    ax.set_ylabel("Source database")

    for container in ax.containers:
        ax.bar_label(container)

    save_figure("02_source_distribution.png")


def plot_histograms(df: pd.DataFrame) -> None:
    """Numeric feature histograms."""
    numeric_columns = [col for col in FEATURE_COLUMNS if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    axes = df[numeric_columns].hist(figsize=(14, 10), bins=20)
    for ax in axes.flatten():
        ax.set_ylabel("Frequency")
    plt.suptitle("Feature distributions for raw-only UCI Heart Disease dataset", y=1.02)
    save_figure("03_feature_histograms.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation heatmap across numeric ML variables and target."""
    corr_columns = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col in df.columns]
    corr_df = df[corr_columns].corr(numeric_only=True)

    plt.figure(figsize=(12, 9))
    ax = sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.4)
    ax.set_title("Correlation heatmap: raw-only selected 14 UCI variables")
    save_figure("04_correlation_heatmap.png")


def plot_feature_relationships(df: pd.DataFrame) -> None:
    """Feature relationship plots for medical/ML interpretation."""
    if "age" in df.columns and "thalach" in df.columns:
        plt.figure(figsize=(8, 5))
        ax = sns.boxplot(data=df, x=TARGET_COLUMN, y="thalach")
        ax.set_title("Maximum heart rate by target class")
        ax.set_xlabel("Target class")
        ax.set_ylabel("Maximum heart rate achieved")
        save_figure("05_thalach_by_target.png")

    if {"age", "chol", TARGET_COLUMN}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        ax = sns.scatterplot(data=df, x="age", y="chol", hue=TARGET_COLUMN, alpha=0.65)
        ax.set_title("Age vs cholesterol by heart disease target")
        ax.set_xlabel("Age")
        ax.set_ylabel("Serum cholesterol")
        save_figure("06_age_cholesterol_by_target.png")


def write_eda_summary(df: pd.DataFrame) -> None:
    """Create a concise Markdown EDA summary for the report/notebook."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    class_counts = df[TARGET_COLUMN].value_counts().sort_index().to_dict()
    missing_total = int(df.isna().sum().sum())
    source_summary = (
        df[SOURCE_COLUMN].value_counts().to_dict() if SOURCE_COLUMN in df.columns else {"not_available": len(df)}
    )

    summary = "# Data Acquisition and EDA Summary\n\n"
    summary += "## Dataset acquisition\n"
    summary += "The project uses a raw-only data policy. The model dataset is created from the original UCI 76-attribute raw files: `cleveland.data`, `hungarian.data`, `switzerland.data`, and `long-beach-va.data`. The script `src/data_acquisition.py` downloads the official files, extracts the 14 documented ML variables, handles missing markers, and converts the diagnosis target into binary classification.\n\n"
    summary += "## Cleaned dataset profile\n"
    summary += f"- Cleaned dataset path: `{CLEAN_DATA_PATH}`\n"
    summary += f"- Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
    summary += f"- Source distribution: {source_summary}\n"
    summary += f"- Target distribution: {class_counts}\n"
    summary += f"- Total missing values after cleaning: {missing_total}\n\n"
    summary += "## EDA outputs generated\n"
    summary += "- `artifacts/eda/missing_values.csv`\n"
    summary += "- `artifacts/eda/data_types.csv`\n"
    summary += "- `artifacts/eda/class_balance.csv`\n"
    summary += "- `artifacts/eda/correlation_matrix.csv`\n"
    summary += "- `artifacts/figures/eda/01_class_balance.png`\n"
    summary += "- `artifacts/figures/eda/02_source_distribution.png`\n"
    summary += "- `artifacts/figures/eda/03_feature_histograms.png`\n"
    summary += "- `artifacts/figures/eda/04_correlation_heatmap.png`\n"
    summary += "- `artifacts/figures/eda/05_thalach_by_target.png`\n"
    summary += "- `artifacts/figures/eda/06_age_cholesterol_by_target.png`\n\n"
    summary += "The same PNG files are also copied to `screenshots/eda/` for direct insertion into the final report.\n"

    (EDA_DIR / "eda_summary.md").write_text(summary, encoding="utf-8")
    LOGGER.info("Saved EDA summary: %s", EDA_DIR / "eda_summary.md")


def main() -> None:
    """Run full EDA workflow and save all required outputs."""
    df = load_dataset()
    create_profile_tables(df)
    plot_class_balance(df)
    plot_source_distribution(df)
    plot_histograms(df)
    plot_correlation_heatmap(df)
    plot_feature_relationships(df)
    write_eda_summary(df)
    LOGGER.info("EDA completed successfully. Figures are in %s", EDA_FIGURE_DIR)


if __name__ == "__main__":
    main()
