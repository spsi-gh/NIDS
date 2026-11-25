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


st.set_page_config(
    page_title="NIDS | Network Intrusion Detection", page_icon="🛡️",
    layout="wide", initial_sidebar_state="expanded",
)
st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=Anton&family=Barlow+Condensed:wght@400;500;600;700&display=swap');
    :root {--lime:#D7FF3F;--ink:#0B0E10;--panel:#171D20;--line:#303B3E;--muted:#A7B0AC;--cream:#F5F3E8;--red:#FF5D71;}
    .stApp {background:radial-gradient(circle at 82% 2%,rgba(215,255,63,.10),transparent 30%),
        repeating-linear-gradient(135deg,transparent 0,transparent 74px,rgba(255,255,255,.012) 75px,transparent 76px),var(--ink);}
    .block-container {max-width:1500px;padding:1.6rem 2.5rem 4rem;}
    h1,h2,h3,.nids-brand,.nids-hero-title {font-family:'Anton',Impact,'Arial Narrow',sans-serif!important;
        text-transform:uppercase;letter-spacing:.035em;font-weight:400!important;}
    h2 {font-size:1.75rem!important;} h3 {font-size:1.35rem!important;}
    .nids-hero {position:relative;overflow:hidden;border:1px solid #384344;border-left:6px solid var(--lime);
        border-radius:4px;background:linear-gradient(105deg,#1D2627,#111719 64%,#1D2B25);
        padding:1.45rem 1.8rem 1.55rem;margin-bottom:1.35rem;box-shadow:0 15px 35px rgba(0,0,0,.17);}
    .nids-hero:after {content:'//';position:absolute;right:2%;top:-55px;color:rgba(215,255,63,.08);
        font:240px Impact,sans-serif;line-height:1;pointer-events:none;}
    .nids-kicker,.nids-eyebrow {color:var(--lime);font:700 .82rem 'Barlow Condensed',sans-serif;
        letter-spacing:.24em;text-transform:uppercase;}
    .nids-hero-title {font-size:clamp(2.2rem,4vw,4rem);line-height:1.06;color:var(--cream);margin:.3rem 0 .55rem;}
    .nids-hero-copy {color:#C2CBC5;font-size:1.05rem;max-width:760px;line-height:1.5;margin:0;}
    .nids-brand {font-size:2.25rem;line-height:1;color:var(--cream);margin:.2rem 0 .15rem;}
    .nids-brand span {color:var(--lime);}
    .nids-brand-sub {font:700 .76rem 'Barlow Condensed',sans-serif;letter-spacing:.22em;
        color:#AAB4AD;text-transform:uppercase;margin-bottom:1.25rem;}
    .nids-side-label {font:700 .76rem 'Barlow Condensed',sans-serif;letter-spacing:.2em;
        color:var(--lime);text-transform:uppercase;margin:1rem 0 .45rem;}
    .nids-side-foot {border-top:1px solid var(--line);padding-top:1rem;margin-top:1.25rem;
        color:#9CAA9F;font:500 .85rem 'Barlow Condensed',sans-serif;letter-spacing:.04em;}
    section[data-testid="stSidebar"] {border-right:1px solid #354243;}
    section[data-testid="stSidebar"] .block-container {padding:1.8rem 1.2rem;}
    section[data-testid="stSidebar"] div[role="radiogroup"] {gap:.28rem;width:100%;}
    section[data-testid="stSidebar"] div[role="radiogroup"] > div {width:100%;}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {padding:.75rem .9rem;
        border:1px solid transparent;border-radius:4px;background:#192123;transition:.15s ease;
        font-family:'Barlow Condensed',sans-serif;font-size:1.07rem;font-weight:600;letter-spacing:.035em;
        width:100%;display:flex;}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {border-color:#65705E;background:#222E2B;}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background:#293723;border-color:var(--lime);box-shadow:inset 3px 0 var(--lime);}
    div[data-testid="stMetric"] {min-height:112px;background:linear-gradient(135deg,#202B2A,#171D20);
        border:1px solid #354340;border-top:3px solid var(--lime);border-radius:4px;padding:1rem 1.1rem;
        box-shadow:0 10px 24px rgba(0,0,0,.13);}
    div[data-testid="stMetricLabel"] {color:#B8C3B7;font:700 .85rem 'Barlow Condensed',sans-serif;
        text-transform:uppercase;letter-spacing:.12em;}
    div[data-testid="stMetricValue"] {color:var(--cream);font-family:'Anton',Impact,sans-serif;font-size:2.1rem;}
    div[data-testid="stButton"] button[kind="primary"],div[data-testid="stDownloadButton"] button {
        background:var(--lime);border:1px solid var(--lime);color:#111719;font-weight:800;
        text-transform:uppercase;letter-spacing:.08em;border-radius:3px;}
    div[data-testid="stButton"] button[kind="primary"]:hover,div[data-testid="stDownloadButton"] button:hover {
        background:#E7FF91;color:#111719;border-color:#E7FF91;}
    div[data-testid="stFileUploader"] {border:1px dashed #758C68;border-radius:4px;padding:.65rem;background:#171F1D;}
    div[data-testid="stDataFrame"] {border:1px solid #384443;border-radius:4px;overflow:hidden;}
    div[data-testid="stAlert"] {border-radius:4px;}
    .nids-steps {display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem;margin:.55rem 0 1.25rem;}
    .nids-step {border:1px solid #34413C;background:#18211E;border-radius:4px;padding:.8rem 1rem;
        color:#C4CDC4;font-size:.92rem;}
    .nids-step b {color:var(--lime);font:700 1rem 'Barlow Condensed',sans-serif;
        letter-spacing:.1em;margin-right:.45rem;}
    @media(max-width:900px){.block-container{padding:1rem 1rem 3rem}.nids-steps{grid-template-columns:1fr}}
    </style>""",
    unsafe_allow_html=True,
)


def page_header(number: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="nids-hero"><div class="nids-kicker">NIDS / {number}</div>'
        f'<h1 class="nids-hero-title">{title}</h1><p class="nids-hero-copy">{description}</p></div>',
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
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=55, b=20),
        font=dict(color="#D9E2D8", family="Arial, sans-serif", size=13),
        title_font=dict(color="#F5F3E8", family="Barlow Condensed, Arial, sans-serif", size=22),
        colorway=["#D7FF3F", "#FF5D71", "#45D6D0", "#FFB44A", "#A99BFF"],
        xaxis=dict(gridcolor="#2E3938", zerolinecolor="#3D4845"),
        yaxis=dict(gridcolor="#2E3938", zerolinecolor="#3D4845"),
    )
    st.plotly_chart(fig, width="stretch", key=key)


def readable_events(events: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Format stored events for quick human review without changing SQLite values."""
    display = events[columns].copy()
    if "analyzed_at" in display:
        display["analyzed_at"] = pd.to_datetime(display["analyzed_at"], utc=True).dt.strftime("%d %b %Y · %H:%M UTC")
    if "confidence" in display:
        display["confidence"] = display["confidence"].map(lambda value: f"{value:.1%}")
    return display.rename(columns={
        "id": "Alert #", "analyzed_at": "Analyzed", "prediction": "Result",
        "confidence": "Confidence", "severity": "Severity", "proto": "Protocol",
        "service": "Service", "attack_cat": "Attack type", "suggested_reason": "Observed signals",
    })


def page_overview(events: pd.DataFrame):
    page_header("01 / OVERVIEW", "Network Intrusion Detection", "Flow classification, alert activity, and a clear view of what the detector has flagged.")
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
        st.info("Get started: open Batch Detection in the sidebar and upload a UNSW-NB15-format CSV.")
        return
    left, right = st.columns([1, 1.5])
    with left:
        chart(px.pie(events, names="prediction", hole=0.62,
                     color="prediction", color_discrete_map={"Normal": "#D7FF3F", "Attack": "#FF5D71"},
                     title="Normal vs detected attack"))
    with right:
        by_severity = events[events["prediction"] == "Attack"]["severity"].value_counts()
        if len(by_severity):
            chart(px.bar(x=by_severity.index, y=by_severity.values,
                         labels={"x": "Severity", "y": "Alerts"}, title="Alerts by severity",
                         color=by_severity.index,
                         color_discrete_map={"Medium": "#FFB44A", "High": "#FF875A", "Critical": "#FF5D71"}))
    st.subheader("Recent detections")
    st.dataframe(readable_events(events.head(12), ["analyzed_at", "prediction", "confidence", "severity", "proto", "service"]),
                 hide_index=True, width="stretch")


def page_detection():
    page_header("02 / SCAN", "Batch Detection", "Upload network flows, run the detector, and download a result for every row.")
    st.markdown(
        '<div class="nids-steps"><div class="nids-step"><b>01</b> Choose a flow CSV</div>'
        '<div class="nids-step"><b>02</b> Analyze the records</div>'
        '<div class="nids-step"><b>03</b> Review and export alerts</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("Use the official UNSW-NB15 feature columns. Labels are optional and never used to make predictions.")
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
            st.dataframe(records.head(5), hide_index=True, width="stretch")
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
        st.dataframe(readable_events(suspicious, display), hide_index=True, width="stretch")
        st.download_button("Download all results", result.to_csv(index=False).encode("utf-8"),
                           "nids_predictions.csv", "text/csv")


def page_analytics(events: pd.DataFrame):
    page_header("03 / INTEL", "Traffic Analytics", "Explore analyzed traffic and the dataset patterns behind the model.")
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
    page_header("04 / WATCHLIST", "Alerts", "Filter recent detections by severity, protocol, and known attack type.")
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
    fields = ["id", "analyzed_at", "severity", "confidence", "proto", "service", "attack_cat", "suggested_reason"]
    st.dataframe(readable_events(alerts, fields), hide_index=True, width="stretch")


def page_performance():
    page_header("05 / MODEL", "Model Performance", "See how the selected detector performed on traffic it did not train on.")
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
    st.dataframe(pd.DataFrame(report["validation_comparison"]), hide_index=True, width="stretch")
    st.caption(report["selection_rule"])


PAGES = ["Overview", "Batch Detection", "Traffic Analytics", "Alerts", "Model Performance"]
events = read_events()
with st.sidebar:
    st.markdown('<div class="nids-brand">NIDS<span>//</span></div><div class="nids-brand-sub">Network security console</div>', unsafe_allow_html=True)
    st.markdown('<div class="nids-side-label">Navigate</div>', unsafe_allow_html=True)
    page = st.radio("Navigate", PAGES, label_visibility="collapsed", key="navigation")
    st.markdown('<div class="nids-side-label">Session status</div>', unsafe_allow_html=True)
    st.caption(f"{len(events):,} analyzed flows · {int((events['prediction'] == 'Attack').sum()) if len(events) else 0:,} alerts")
    st.markdown('<div class="nids-side-foot">UNSW-NB15 / BINARY DETECTION<br>CSV ANALYSIS · LOCAL ALERT HISTORY</div>', unsafe_allow_html=True)

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
