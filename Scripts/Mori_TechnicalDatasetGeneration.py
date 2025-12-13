# -*- coding: utf-8 -*-
"""
Merge multiple Mori JSON datasets into a single Parquet file.

This script builds Mori's base knowledge dataset by merging multiple JSON files.
It can be imported as a function (pipeline-friendly) or executed as a script.

Expected JSON structure: list[dict] with keys:
  - canonical_term
  - input
  - output
  - context
  - question_type
"""

# =============================================================================
# Imports
# =============================================================================
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from Scripts.config import JSONs_PATH, DATA_PATH as OUTPUT_PARQUET_PATH

# =============================================================================
# Configuration
# =============================================================================
REQUIRED_COLS = ["canonical_term", "input", "output", "context", "question_type"]

# Dedup strategy:
# - "strict": exact match on all required columns
# - "qa": dedup by (input, output) only (fast, but may merge different contexts/types)
# - None: no dedup
DEDUP_MODE: Optional[str] = "qa"

# Input directory containing JSON files
JSON_DIR: Path = JSONs_PATH

# Output parquet path
OUTPUT_PARQUET: Path = OUTPUT_PARQUET_PATH

# Optional: restrict allowed question types (helps keep the dataset clean)
# Set to None to disable validation.
ALLOWED_QUESTION_TYPES = None  # e.g. {"definicion", "procedimiento", "ejemplos"}


# =============================================================================
# Helpers
# =============================================================================
def find_json_files(data_dir: Path) -> List[Path]:
    """Recursively locate all JSON files within a directory."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    return sorted(data_dir.rglob("*.json"))


def load_json_records(path: Path) -> List[dict]:
    """Load a JSON file containing a non-empty list of dict records."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Expected a non-empty list of records in: {path}")

    if not isinstance(data[0], dict):
        raise ValueError(f"Expected list[dict] in {path}, got list[{type(data[0])}]")

    return data


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns and enforce the required schema.

    - Ensures REQUIRED_COLS exist
    - Strips whitespace
    - Removes empty/invalid rows
    """
    # Ensure all required columns exist
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = None

    # Keep only required columns
    df = df[REQUIRED_COLS].copy()

    # Normalize text fields
    for col in REQUIRED_COLS:
        df[col] = (
            df[col]
            .astype("string")
            .fillna("")
            .str.strip()
        )

    # Drop invalid rows
    df = df[(df["input"] != "") & (df["output"] != "") & (df["question_type"] != "")]

    # Optional: validate question_type values
    if ALLOWED_QUESTION_TYPES is not None:
        df = df[df["question_type"].isin(ALLOWED_QUESTION_TYPES)]

    return df


def deduplicate(df: pd.DataFrame, mode: Optional[str]) -> pd.DataFrame:
    """Deduplicate records using the selected strategy."""
    if mode is None:
        return df

    if mode == "strict":
        return df.drop_duplicates(subset=REQUIRED_COLS, keep="first")

    if mode == "qa":
        # Note: This may collapse records that differ only in context/question_type.
        return df.drop_duplicates(subset=["input", "output"], keep="first")

    raise ValueError(f"Unknown DEDUP_MODE: {mode}")


# =============================================================================
# Pipeline entry point
# =============================================================================
def generate_mori_knowledge_dataset() -> None:
    """
    Build the consolidated Mori knowledge Parquet dataset.

    Steps:
      1) Discover JSON knowledge files
      2) Load + normalize each file into a DataFrame
      3) Merge all frames
      4) Deduplicate
      5) Save to Parquet
    """
    json_files = find_json_files(JSON_DIR)

    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON files found under: {JSON_DIR}")

    frames: List[pd.DataFrame] = []

    for path in json_files:
        print(f"[INFO] Loading Mori training data from: {path}")
        records = load_json_records(path)
        frames.append(normalize_dataframe(pd.DataFrame(records)))

    merged = pd.concat(frames, ignore_index=True)
    merged = deduplicate(merged, DEDUP_MODE)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"[INFO] Saved Parquet: {OUTPUT_PARQUET}")
    print(f"[INFO] Rows: {len(merged)}")
    print(f"[INFO] Columns: {list(merged.columns)}")


def main() -> None:
    """CLI entry point."""
    generate_mori_knowledge_dataset()


if __name__ == "__main__":
    main()
