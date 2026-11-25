"""Validate and normalize UNSW-NB15 data before model fitting."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.utils import PROCESSED_DIR, RAW_DIR, REPORT_DIR


TRAIN_NAME = "UNSW_NB15_training-set.csv"
TEST_NAME = "UNSW_NB15_testing-set.csv"
EXPECTED_ROWS = {TRAIN_NAME: 175_341, TEST_NAME: 82_332}
CATEGORICAL = ["proto", "service", "state"]
NON_FEATURES = ["id", "attack_cat", "label"]


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = frame.columns.str.removeprefix("\ufeff").str.strip()
    required = set(CATEGORICAL + ["label", "attack_cat"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")
    for col in CATEGORICAL:
        frame[col] = frame[col].astype("string").fillna("unknown").str.strip().replace("", "unknown")
    frame["attack_cat"] = frame["attack_cat"].astype("string").fillna("Normal").str.strip()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("label must contain only 0 (normal) and 1 (attack)")
    for col in frame.columns.difference(CATEGORICAL + NON_FEATURES):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def inspect(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "columns": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "missing_values": {name: int(count) for name, count in frame.isna().sum().items()},
        "label_distribution": {str(k): int(v) for k, v in frame["label"].value_counts().items()},
        "attack_category_distribution": {
            str(k): int(v) for k, v in frame["attack_cat"].value_counts().items()
        },
    }


def prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    report: dict[str, dict] = {}
    for name, expected in EXPECTED_ROWS.items():
        path = RAW_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run python scripts/download_data.py")
        raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        if len(raw) != expected:
            raise ValueError(f"{name}: expected {expected} rows; found {len(raw)}")
        report[name] = inspect(raw)
        frame = normalize(raw)
        frames[name] = frame
        frame.to_csv(PROCESSED_DIR / name, index=False)
    if list(frames[TRAIN_NAME].columns) != list(frames[TEST_NAME].columns):
        raise ValueError("Training and test columns differ")
    if frames[TRAIN_NAME].columns.duplicated().any():
        raise ValueError("Duplicate dataset columns")
    (REPORT_DIR / "dataset_inspection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return frames[TRAIN_NAME], frames[TEST_NAME]


if __name__ == "__main__":
    train, test = prepare()
    print(f"Prepared {len(train):,} training and {len(test):,} test flows")
    print(f"Features: {len(train.columns) - len(NON_FEATURES)}")
