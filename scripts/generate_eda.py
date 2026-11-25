"""Generate a portable EDA report from the verified training partition."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "UNSW_NB15_training-set.csv"
OUT = ROOT / "reports"


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError("Run python -m src.preprocess first")
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA)
    figures = []
    labels = frame["label"].map({0: "Normal", 1: "Attack"})
    figures.append(px.histogram(x=labels, title="Normal vs attack traffic", labels={"x": "Class"}))
    categories = frame.loc[frame["label"] == 1, "attack_cat"].value_counts()
    figures.append(px.bar(x=categories.index, y=categories.values, title="Attack type distribution",
                          labels={"x": "Attack type", "y": "Flows"}))
    for field, title in [("proto", "Most common protocols"), ("service", "Most common services")]:
        counts = frame[field].value_counts().head(12)
        figures.append(px.bar(x=counts.index, y=counts.values, title=title,
                              labels={"x": field, "y": "Flows"}))
    sample = frame.sample(min(10000, len(frame)), random_state=42)
    figures.append(px.histogram(sample, x="dur", color="label", nbins=70, log_y=True,
                                title="Flow duration (10,000-row sample)"))
    numerical = ["dur", "rate", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl", "sload", "dload"]
    figures.append(px.imshow(frame[numerical].corr(), color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, title="Numerical feature correlation"))
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>UNSW-NB15 EDA</title>",
            "<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;background:#0b1724;color:#e6eef6}h1{margin-bottom:0}p{color:#9cb0c4}</style></head><body>",
            "<h1>UNSW-NB15 training data: exploratory analysis</h1>",
            "<p>Official 175,341-row training partition. Attack categories are ground truth, not model predictions. The partition has no port or flow timestamp columns.</p>"]
    for i, fig in enumerate(figures):
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1724", plot_bgcolor="#0b1724")
        html.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn" if i == 0 else False))
    html.append("</body></html>")
    (OUT / "eda_charts.html").write_text("\n".join(html), encoding="utf-8")
    inspection = json.loads((OUT / "dataset_inspection.json").read_text(encoding="utf-8"))
    train = inspection["UNSW_NB15_training-set.csv"]
    test = inspection["UNSW_NB15_testing-set.csv"]
    lines = [
        "# Dataset inspection", "",
        f"- Training records: {train['rows']:,}; test records: {test['rows']:,}.",
        f"- Columns: {len(train['columns'])}; usable model features: {len(train['columns']) - 3}.",
        f"- Training normal: {train['label_distribution'].get('0', 0):,}; attacks: {train['label_distribution'].get('1', 0):,}.",
        f"- Test normal: {test['label_distribution'].get('0', 0):,}; attacks: {test['label_distribution'].get('1', 0):,}.",
        f"- Missing values in original training CSV: {sum(train['missing_values'].values()):,}; test CSV: {sum(test['missing_values'].values()):,}.",
        "", "## Column types", "",
        "| Column | pandas dtype | Training missing |", "|---|---|---:|",
    ]
    lines += [f"| {name} | {dtype} | {train['missing_values'][name]:,} |" for name, dtype in train["columns"].items()]
    lines += ["", "## Training attack categories", "", "| Category | Rows |", "|---|---:|"]
    lines += [f"| {name} | {count:,} |" for name, count in train["attack_category_distribution"].items()]
    lines += ["", "Charts: [Open interactive EDA charts](eda_charts.html).", ""]
    (OUT / "dataset_inspection.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
