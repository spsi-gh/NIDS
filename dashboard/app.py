"""Streamlit interface for UNSW-NB15 flow classification and alert review."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import load_model, predict_records
from src.utils import PROCESSED_DIR, REPORT_DIR, read_events, save_events


st.set_page_config(page_title="NIDS | Network Intrusion Detection", page_icon="🛡️", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 1.5rem; max-width: 1450px;}
    div[data-testid="stMetric"] {background: #eef4fa; border: 1px solid #cbd9e7;
        border-radius: 14px; padding: 16px;}
    h1, h2, h3 {letter-spacing: -.025em;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def model_bundle():
    return load_model()


@st.cache_data
def training_data():
    path = PROCESSED_DIR / "UNSW_NB15_training-set.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def evaluation_data():
    path = REPORT_DIR / "evaluation.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def chart(fig, key: str | None = None):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=45, b=20),
        font=dict(color="#d8e5f0"),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def page_overview(events: pd.DataFrame):
    st.title("Network Intrusion Detection")
    st.caption("Flow classification and alert review · UNSW-NB15 model")
    total = len(events)
    attacks = int((events["prediction"] == "Attack").sum()) if total else 0
    normal = total - attacks
    labeled = events.dropna(subset=["source_label"]) if total else events
    actual_attacks = labeled[labeled["source_label"] == 1] if len(labeled) else labeled
    recall = (
        (actual_attacks["prediction"] == "Attack").mean()
        if len(actual_attacks) else None
    )
    cols = st.columns(4)
    for col, label, value in zip(
        cols,
        ["Analyzed flows", "Normal flows", "Detected attacks", "Attack detection rate"],
        [f"{total:,}", f"{normal:,}", f"{attacks:,}", f"{recall:.1%}" if recall is not None else "—"],
    ):
        col.metric(label, value)
    st.caption("Attack detection rate is recall on uploaded flows with a known label; — means no labeled attacks have been analyzed.")
    if not total:
        st.info("Upload a UNSW-NB15-format CSV in Batch Detection to start analyzing flows.")
        return
    left, right = st.columns([1, 1.5])
    with left:
        chart(px.pie(events, names="prediction", hole=0.62,
                     color="prediction", color_discrete_map={"Normal": "#3db9a3", "Attack": "#ef6274"},
                     title="Normal vs detected attack"))
    with right:
        by_severity = events[events["prediction"] == "Attack"]["severity"].value_counts()
        if len(by_severity):
            chart(px.bar(x=by_severity.index, y=by_severity.values,
                         labels={"x": "Severity", "y": "Alerts"}, title="Alerts by severity",
                         color=by_severity.index,
                         color_discrete_map={"Medium": "#eabf63", "High": "#f19362", "Critical": "#ef6274"}))
    st.subheader("Recent detections")
    st.dataframe(events[["analyzed_at", "prediction", "confidence", "severity", "proto", "service"]].head(12),
                 hide_index=True, use_container_width=True)


def page_detection():
    st.title("Batch Detection")
    st.caption("Upload flow features matching the official UNSW-NB15 partition columns. Labels are optional and never used as model inputs.")
    try:
        model, metadata = model_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    with st.expander("Required input columns"):
        st.write(", ".join(metadata["features"]))
    upload = st.file_uploader("Traffic CSV", type=["csv"])
    if upload is not None:
        try:
            records = pd.read_csv(upload, low_memory=False, encoding="utf-8-sig")
            st.caption(f"{len(records):,} rows · {len(records.columns)} columns")
            st.dataframe(records.head(5), hide_index=True, use_container_width=True)
            if st.button("Analyze flows", type="primary"):
                with st.spinner("Classifying flows..."):
                    result = predict_records(records, model, metadata)
                    save_events(result, str(uuid.uuid4()))
                    st.session_state["latest_result"] = result
                st.success(f"Analyzed {len(result):,} flows; {(result['prediction'] == 'Attack').sum():,} attacks flagged.")
        except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
            st.error(f"Could not analyze CSV: {exc}")
    result = st.session_state.get("latest_result")
    if result is not None:
        st.subheader("Suspicious flows")
        suspicious = result[result["prediction"] == "Attack"]
        display = ["analyzed_at", "prediction", "confidence", "severity", "proto", "service",
                   "attack_cat", "suggested_reason"]
        st.dataframe(suspicious[display], hide_index=True, use_container_width=True)
        st.download_button("Download all results", result.to_csv(index=False).encode("utf-8"),
                           "nids_predictions.csv", "text/csv")


def page_analytics(events: pd.DataFrame):
    st.title("Traffic Analytics")
    tab1, tab2 = st.tabs(["Analyzed traffic", "Training dataset EDA"])
    with tab1:
        if events.empty:
            st.info("Analyze a CSV to see traffic and alert trends.")
        else:
            left, right = st.columns(2)
            with left:
                counts = events["proto"].value_counts().head(12)
                chart(px.bar(x=counts.index, y=counts.values, labels={"x": "Protocol", "y": "Flows"},
                             title="Protocol distribution"))
            with right:
                counts = events["service"].value_counts().head(12)
                chart(px.bar(x=counts.index, y=counts.values, labels={"x": "Service", "y": "Flows"},
                             title="Service distribution"))
            chart(px.pie(events, names="prediction", hole=0.45, title="Normal vs malicious predictions"))
            timed = events.copy()
            timed["period"] = pd.to_datetime(timed["analyzed_at"], utc=True).dt.floor("h")
            trend = timed.groupby(["period", "prediction"]).size().reset_index(name="flows")
            chart(px.line(trend, x="period", y="flows", color="prediction", markers=True,
                          title="Analysis events over time (UTC)"))
            st.caption("Times record when the dashboard analyzed each upload, not when network traffic occurred.")
    with tab2:
        train = training_data()
        if train.empty:
            st.info("Run preprocessing to view training dataset charts.")
            return
        a, b = st.columns(2)
        with a:
            labels = train["label"].map({0: "Normal", 1: "Attack"})
            chart(px.histogram(x=labels, labels={"x": "Class"}, title="Training class balance"))
            categories = train.loc[train["label"] == 1, "attack_cat"].value_counts()
            chart(px.bar(x=categories.index, y=categories.values,
                         labels={"x": "Attack category", "y": "Flows"}, title="Known attack categories"))
        with b:
            protocols = train["proto"].value_counts().head(12)
            chart(px.bar(x=protocols.index, y=protocols.values,
                         labels={"x": "Protocol", "y": "Flows"}, title="Most common protocols"))
            services = train["service"].value_counts().head(12)
            chart(px.bar(x=services.index, y=services.values,
                         labels={"x": "Service", "y": "Flows"}, title="Most common services"))
        chart(px.histogram(train.sample(min(10000, len(train)), random_state=42), x="dur", color="label",
                           nbins=70, log_y=True, title="Flow duration distribution (sample)"))
        selected = ["dur", "rate", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl", "sload", "dload"]
        corr = train[selected].corr()
        chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Numerical feature correlations"))
        st.caption("The official pre-split CSV has service names but no source or destination port columns.")


def page_alerts(events: pd.DataFrame):
    st.title("Alerts")
    alerts = events[events["prediction"] == "Attack"].copy() if len(events) else events
    if alerts.empty:
        st.info("No attack alerts have been stored yet.")
        return
    a, b, c = st.columns(3)
    severities = a.multiselect("Severity", sorted(alerts["severity"].dropna().unique()))
    protocols = b.multiselect("Protocol", sorted(alerts["proto"].dropna().unique()))
    types = c.multiselect("Attack type", sorted(alerts["attack_cat"].dropna().unique()))
    if severities:
        alerts = alerts[alerts["severity"].isin(severities)]
    if protocols:
        alerts = alerts[alerts["proto"].isin(protocols)]
    if types:
        alerts = alerts[alerts["attack_cat"].isin(types)]
    st.caption(f"{len(alerts):,} alerts · most recent first. Attack type is known only when supplied in the uploaded CSV.")
    fields = ["analyzed_at", "severity", "confidence", "proto", "service", "attack_cat", "suggested_reason", "batch_id"]
    st.dataframe(alerts[fields], hide_index=True, use_container_width=True)


def page_performance():
    st.title("Model Performance")
    report = evaluation_data()
    if report is None:
        st.info("Run python -m src.train to generate the evaluation report.")
        return
    st.caption(f"Selected model: {report['selected_model']} · official held-out test set ({report['test_rows']:,} flows)")
    metrics = report["test_metrics"]
    cols = st.columns(5)
    for col, name, key in zip(cols, ["Accuracy", "Precision", "Attack recall", "F1", "ROC-AUC"],
                              ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        col.metric(name, f"{metrics[key]:.1%}")
    left, right = st.columns(2)
    with left:
        matrix = report["confusion_matrix"]
        fig = go.Figure(go.Heatmap(z=matrix, x=["Predicted normal", "Predicted attack"],
                                   y=["Actual normal", "Actual attack"], colorscale="Blues",
                                   text=matrix, texttemplate="%{text}"))
        fig.update_layout(title="Confusion matrix")
        chart(fig)
    with right:
        importance_path = REPORT_DIR / "feature_importance.csv"
        if importance_path.exists():
            importance = pd.read_csv(importance_path).head(12).sort_values("importance")
            chart(px.bar(importance, x="importance", y="feature", orientation="h",
                         title="Permutation importance for attack recall"))
    st.subheader("Validation comparison")
    st.dataframe(pd.DataFrame(report["validation_comparison"]), hide_index=True, use_container_width=True)
    st.caption(report["selection_rule"])


PAGES = ["Overview", "Batch Detection", "Traffic Analytics", "Alerts", "Model Performance"]
with st.sidebar:
    st.title("🛡️ NIDS")
    st.caption("Network flow monitoring")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("UNSW-NB15 · binary classifier")

events = read_events()
if page == "Overview":
    page_overview(events)
elif page == "Batch Detection":
    page_detection()
elif page == "Traffic Analytics":
    page_analytics(events)
elif page == "Alerts":
    page_alerts(events)
else:
    page_performance()
