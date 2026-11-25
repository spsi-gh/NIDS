"""Paths, alert rules, and local event storage."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
DB_PATH = ROOT / "data" / "events.sqlite3"


def severity_for(prediction: int, attack_probability: float) -> str:
    if prediction == 0:
        return "Low"
    if attack_probability > 0.90:
        return "Critical"
    if attack_probability >= 0.75:
        return "High"
    return "Medium"


def init_db(path: Path = DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS flow_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                row_index INTEGER NOT NULL,
                analyzed_at TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                attack_probability REAL NOT NULL,
                severity TEXT NOT NULL,
                proto TEXT,
                service TEXT,
                attack_cat TEXT,
                source_label INTEGER,
                suggested_reason TEXT,
                UNIQUE(batch_id, row_index)
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flow_events_time ON flow_events(analyzed_at DESC)")


def save_events(events: pd.DataFrame, batch_id: str, path: Path = DB_PATH) -> None:
    init_db(path)
    cols = [
        "batch_id", "row_index", "analyzed_at", "prediction", "confidence",
        "attack_probability", "severity", "proto", "service", "attack_cat",
        "source_label", "suggested_reason",
    ]
    stored = events.copy()
    stored["batch_id"] = batch_id
    stored["row_index"] = range(len(stored))
    for col in cols:
        if col not in stored:
            stored[col] = None
    stored = stored[cols].astype(object).where(pd.notna(stored[cols]), None)
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO flow_events (" + ",".join(cols) + ") VALUES (" + ",".join("?" for _ in cols) + ")",
            stored.itertuples(index=False, name=None),
        )


def read_events(path: Path = DB_PATH) -> pd.DataFrame:
    init_db(path)
    with sqlite3.connect(path) as conn:
        return pd.read_sql_query("SELECT * FROM flow_events ORDER BY analyzed_at DESC, id DESC", conn)
