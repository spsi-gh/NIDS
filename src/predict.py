"""Predict binary attacks from UNSW-NB15-format network-flow records."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils import MODEL_DIR, severity_for


SIGNALS = {
    "rate": "high connection rate",
    "spkts": "many source packets",
    "dpkts": "many destination packets",
    "sbytes": "large source byte count",
    "dbytes": "large destination byte count",
    "dur": "long flow duration",
}


def load_model(model_dir: Path = MODEL_DIR):
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Model metadata missing. Run python -m src.train first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_path = model_dir / metadata["artifact"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact missing: {model_path}. Run python -m src.train first.")
    return joblib.load(model_path), metadata


def validate_input(records: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    missing = sorted(set(metadata["features"]) - set(records.columns))
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    if records.empty:
        raise ValueError("Input CSV has no flow records")
    data = records[metadata["features"]].copy()
    for col in metadata["numeric"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for col in metadata["categorical"]:
        data[col] = data[col].astype(object).where(pd.notna(data[col]), np.nan)
    return data


def observed_signals(row: pd.Series, quantiles: dict) -> str:
    signals = []
    for col, description in SIGNALS.items():
        value = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(value) and value > quantiles[col]:
            signals.append(description)
    return ", ".join(signals[:3]) if signals else "No single elevated signal identified"


def predict_records(records: pd.DataFrame, model=None, metadata: dict | None = None) -> pd.DataFrame:
    if model is None or metadata is None:
        model, metadata = load_model()
    records = records.reset_index(drop=True)
    data = validate_input(records, metadata)
    probabilities = model.predict_proba(data)[:, 1]
    predictions = (probabilities >= metadata["decision_threshold"]).astype(int)
    results = records.copy().reset_index(drop=True)
    results["analyzed_at"] = datetime.now(timezone.utc).isoformat()
    results["prediction"] = np.where(predictions == 1, "Attack", "Normal")
    results["attack_probability"] = probabilities
    results["confidence"] = np.where(predictions == 1, probabilities, 1 - probabilities)
    results["severity"] = [severity_for(int(pred), float(prob)) for pred, prob in zip(predictions, probabilities)]
    results["suggested_reason"] = [
        observed_signals(row, metadata["training_p95"]) if pred else "No attack flagged"
        for (_, row), pred in zip(data.iterrows(), predictions)
    ]
    results["source_label"] = pd.to_numeric(records["label"], errors="coerce") if "label" in records else np.nan
    results["attack_cat"] = records["attack_cat"].fillna("Unknown type") if "attack_cat" in records else "Unknown type"
    results["proto"] = records["proto"].fillna("unknown")
    results["service"] = records["service"].fillna("unknown")
    return results


def predict_record(record: dict) -> dict:
    """Public one-flow interface; returns the same fields as batch inference."""
    result = predict_records(pd.DataFrame([record])).iloc[0]
    return {
        "prediction": result["prediction"],
        "confidence": float(result["confidence"]),
        "attack_probability": float(result["attack_probability"]),
        "severity": result["severity"],
        "suggested_reason": result["suggested_reason"],
    }
