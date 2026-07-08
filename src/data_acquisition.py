"""Raw-only data acquisition for the UCI Heart Disease MLOps project.

This module intentionally uses the original unprocessed UCI Heart Disease data files:

- cleveland.data
- hungarian.data
- switzerland.data
- long-beach-va.data

Each raw source file follows the same 76-attribute record format. The UCI documentation
states that the published ML experiments use 14 attributes from those raw records. This
script extracts the 14 documented attributes directly from the raw 76-attribute files,
cleans missing markers such as -9 and ?, converts the original diagnosis field into a
binary target, and saves the final model-ready dataset.

The processed.* files are deliberately not used for model development in this version.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from src.config import (
    CLEAN_DATA_PATH,
    CLEVELAND_ONLY_CLEAN_PATH,
    FEATURE_COLUMNS,
    RAW_76_RECORD_LENGTH,
    RAW_76_SELECTED_ATTRIBUTE_NUMBERS,
    RAW_76_SELECTED_PATH,
    RAW_76_SOURCE_FILES,
    RAW_COLUMNS,
    SOURCE_COLUMN,
    TARGET_COLUMN,
    UCI_BASE_URL,
    UCI_DIRECTORY_FILES,
    UCI_FULL_ARCHIVE_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def download_file(filename: str, overwrite: bool = False) -> dict[str, str | bool]:
    """Download one official UCI file into the local raw archive folder."""
    UCI_FULL_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    destination = UCI_FULL_ARCHIVE_DIR / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, str | bool] = {
        "file": filename,
        "path": str(destination),
        "downloaded": False,
        "available": True,
    }

    if destination.exists() and destination.stat().st_size > 0 and not overwrite:
        result["status"] = "already_exists"
        return result

    try:
        url = f"{UCI_BASE_URL}/{filename}"
        LOGGER.info("Downloading UCI file: %s", url)
        urlretrieve(url, destination)
        result["downloaded"] = True
        result["status"] = "downloaded"
    except (HTTPError, URLError, TimeoutError, OSError) as exc:  # pragma: no cover - network dependent
        result["available"] = False
        result["status"] = f"failed: {exc}"
        LOGGER.warning("Could not download UCI file %s: %s", filename, exc)
    return result


def download_all_uci_raw_files(overwrite: bool = False) -> list[dict[str, str | bool]]:
    """Download the official raw UCI files, metadata files, and Costs folder files."""
    results = [download_file(filename, overwrite=overwrite) for filename in UCI_DIRECTORY_FILES]
    manifest_path = UCI_FULL_ARCHIVE_DIR / "download_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    expected_path = UCI_FULL_ARCHIVE_DIR / "EXPECTED_UCI_RAW_FILES.txt"
    expected_path.write_text("\n".join(UCI_DIRECTORY_FILES) + "\n", encoding="utf-8")
    return results


def _tokenize_raw_76_text(text: str) -> list[str]:
    """Tokenize raw UCI text supporting comma, whitespace, and line-wrapped records."""
    normalized = text.replace(",", " ")
    return [token.strip() for token in re.split(r"\s+", normalized) if token.strip()]


def parse_raw_76_file(path: Path, source_name: str) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Extract the 14 published ML attributes from one raw 76-attribute source file.

    The raw UCI files may be line-wrapped, so this parser tokenizes the full file and
    chunks it into 76-token patient records before extracting the documented attribute
    positions. Missing values are preserved at this stage and cleaned later.
    """
    columns = RAW_COLUMNS + [SOURCE_COLUMN]
    diagnostics: dict[str, int | str] = {
        "source": source_name,
        "file": str(path),
        "raw_tokens": 0,
        "full_records": 0,
        "trailing_tokens_skipped": 0,
        "rows_parsed": 0,
    }

    if not path.exists() or path.stat().st_size == 0:
        diagnostics["status"] = "missing_or_empty"
        return pd.DataFrame(columns=columns), diagnostics

    text = path.read_text(encoding="latin-1", errors="ignore")
    tokens = _tokenize_raw_76_text(text)
    record_count = len(tokens) // RAW_76_RECORD_LENGTH
    trailing = len(tokens) % RAW_76_RECORD_LENGTH
    zero_based_indices = [idx - 1 for idx in RAW_76_SELECTED_ATTRIBUTE_NUMBERS]

    rows: list[list[str]] = []
    for record_number in range(record_count):
        start = record_number * RAW_76_RECORD_LENGTH
        record = tokens[start : start + RAW_76_RECORD_LENGTH]
        if len(record) == RAW_76_RECORD_LENGTH:
            rows.append([record[index] for index in zero_based_indices] + [source_name])

    diagnostics.update(
        {
            "status": "parsed",
            "raw_tokens": len(tokens),
            "full_records": record_count,
            "trailing_tokens_skipped": trailing,
            "rows_parsed": len(rows),
        }
    )
    return pd.DataFrame(rows, columns=columns), diagnostics


def build_raw_76_selected_dataset() -> pd.DataFrame:
    """Build the selected-14 dataset from only the raw 76-attribute UCI files."""
    frames: list[pd.DataFrame] = []
    diagnostics = []

    for source_name, filename in RAW_76_SOURCE_FILES.items():
        path = UCI_FULL_ARCHIVE_DIR / filename
        if not path.exists():
            download_file(filename)
        parsed, source_diagnostics = parse_raw_76_file(path, source_name)
        diagnostics.append(source_diagnostics)
        if not parsed.empty:
            frames.append(parsed)
        else:
            LOGGER.warning("No raw records parsed for %s from %s", source_name, filename)

    if not frames:
        raise RuntimeError(
            "No raw UCI records were parsed. Check internet access or manually place the raw files in "
            f"{UCI_FULL_ARCHIVE_DIR}. Required files: {list(RAW_76_SOURCE_FILES.values())}"
        )

    combined = pd.concat(frames, ignore_index=True)
    RAW_76_SELECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(RAW_76_SELECTED_PATH, index=False)
    RAW_76_SELECTED_PATH.with_suffix(".profile.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return combined


def clean_heart_data(df: pd.DataFrame, keep_source: bool = True) -> pd.DataFrame:
    """Clean selected raw UCI attributes and convert the original target to binary."""
    missing_columns = [c for c in FEATURE_COLUMNS + ["num"] if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    output_columns = FEATURE_COLUMNS + ["num"] + ([SOURCE_COLUMN] if keep_source and SOURCE_COLUMN in df.columns else [])
    clean_df = df[output_columns].copy()

    # UCI raw files distinguish missing values using -9.0. Processed files often use ?.
    missing_markers = ["?", "-9", "-9.0", -9, -9.0]
    clean_df.replace(missing_markers, np.nan, inplace=True)

    for col in FEATURE_COLUMNS + ["num"]:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # Original UCI target: 0 = absence, 1-4 = presence.
    clean_df = clean_df.dropna(subset=["num"])
    clean_df[TARGET_COLUMN] = (clean_df["num"] > 0).astype(int)
    clean_df.drop(columns=["num"], inplace=True)

    ordered_columns = FEATURE_COLUMNS + ([SOURCE_COLUMN] if keep_source and SOURCE_COLUMN in clean_df.columns else []) + [TARGET_COLUMN]
    return clean_df[ordered_columns]


def save_data_profile(df: pd.DataFrame, output_path: Path, download_results: list[dict[str, str | bool]]) -> None:
    """Save a JSON audit profile for the cleaned raw-only modeling dataset."""
    profile = {
        "data_policy": "raw_76_attribute_files_only",
        "modeling_sources": list(RAW_76_SOURCE_FILES.values()),
        "excluded_from_training": {
            "processed_files": "All processed.* files are intentionally not used in this raw-only version.",
            "new.data": "Downloaded for traceability but not used for training because it is not one of the four named institutional raw databases.",
            "costs": "Downloaded for completeness but not patient-level model training data.",
        },
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "source_distribution": (
            {str(k): int(v) for k, v in df[SOURCE_COLUMN].value_counts(dropna=False).to_dict().items()}
            if SOURCE_COLUMN in df.columns
            else {}
        ),
        "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "target_distribution": {str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().to_dict().items()},
        "download_manifest": download_results,
        "modeling_note": (
            "The cleaned model dataset is created only from the original raw 76-attribute UCI files. "
            "The 14 documented ML variables are extracted directly from raw records, then the diagnosis field is "
            "converted to binary target: 0 = no heart disease, 1 = heart disease risk present."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")


def main() -> None:
    download_results = download_all_uci_raw_files(overwrite=False)

    raw_selected = build_raw_76_selected_dataset()
    clean_all = clean_heart_data(raw_selected, keep_source=True)
    clean_all.to_csv(CLEAN_DATA_PATH, index=False)
    save_data_profile(clean_all, CLEAN_DATA_PATH.with_suffix(".profile.json"), download_results=download_results)

    cleveland_only = raw_selected[raw_selected[SOURCE_COLUMN] == "cleveland"].copy()
    if not cleveland_only.empty:
        clean_cleveland = clean_heart_data(cleveland_only, keep_source=True)
        clean_cleveland.to_csv(CLEVELAND_ONLY_CLEAN_PATH, index=False)

    LOGGER.info("Saved raw selected-14 file to %s", RAW_76_SELECTED_PATH)
    LOGGER.info("Saved raw-only cleaned modeling dataset to %s", CLEAN_DATA_PATH)
    LOGGER.info("Raw-only cleaned shape: %s", clean_all.shape)
    LOGGER.info("Source distribution: %s", clean_all[SOURCE_COLUMN].value_counts(dropna=False).to_dict())
    LOGGER.info("Target distribution: %s", clean_all[TARGET_COLUMN].value_counts().to_dict())


if __name__ == "__main__":
    main()
