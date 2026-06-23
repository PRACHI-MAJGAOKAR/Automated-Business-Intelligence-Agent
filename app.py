"""

Streamlit frontend for the BI Agent.

Features:
  - Natural language question input
  - Live agent status indicators while pipeline runs
  - Interactive Plotly chart
  - Anomaly flags display
  - Executive summary
  - One-click PDF download
  - Session history (last 10 queries, persisted in st.session_state)

Run:  streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from datetime import datetime
from pipeline import run_pipeline
import streamlit as st

# Path 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Page config
st.set_page_config(
    page_title="Prachi NSE Stock Analytics",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

# Session state initialisation
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []   # list of completed run dicts
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

MAX_HISTORY = 10

# Sidebar 
with st.sidebar:
    st.title("Prachi Majgaokar")
    st.caption("NSE Stock Market Analytics")
    st.divider()

    st.subheader("Session History")
    if not st.session_state.history:
        st.caption("No queries yet.")
    else:
        for i, entry in enumerate(reversed(st.session_state.history)):
            label = entry["question"][:48] + ("…" if len(entry["question"]) > 48 else "")
            ts    = entry.get("timestamp", "")
            with st.expander(f"[{len(st.session_state.history) - i}] {label}", expanded=False):
                st.caption(ts)
                st.write(f"**Chart:** {entry.get('chart_type', 'N/A')}")
                st.write(f"**Anomaly method:** {entry.get('anomaly_method', 'N/A')}")
                if entry.get("pdf_path") and os.path.exists(entry["pdf_path"]):
                    with open(entry["pdf_path"], "rb") as f:
                        st.download_button(
                            "⬇ Download PDF",
                            data=f.read(),
                            file_name=os.path.basename(entry["pdf_path"]),
                            mime="application/pdf",
                            key=f"hist_dl_{i}",
                        )

    st.divider()
    st.subheader("Example Questions")
    examples = [
        "Show me the closing price trend for RELIANCE over the last 6 months",
        "Which stock had the highest average volume in 2024?",
        "Compare the average closing price of all 5 stocks",
        "Show daily high-low range for TCS in January 2025",
        "Which month had the maximum trading volume for INFY?",
        "Show the percentage change in closing price for HDFCBANK by month",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state["prefill"] = ex

#  Main area container
st.title("Business Intelligence Agent")
st.caption(
    "Ask any question about NSE stock data in plain English. "
    "The agent writes the SQL, picks the right chart, detects anomalies, "
    "and generates a PDF report — automatically."
)
st.divider()

# Question input
default_q = st.session_state.pop("prefill", "")
question  = st.text_input(
    "Your question",
    value=default_q,
    placeholder="e.g. Show me the closing price trend for RELIANCE over the last 3 months",
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    run_btn = st.button("▶  Analyse", type="primary", use_container_width=True)
with col_clear:
    if st.button("✕  Clear history", use_container_width=False):
        st.session_state.history = []
        st.rerun()

# Run pipeline
if run_btn and question.strip():
    st.divider()

    # Agent status placeholders
    status_cols = st.columns(5)
    labels = [" SQL", " Chart", " Anomaly", " Summary", " PDF"]
    status_boxes = [col.empty() for col in status_cols]

    def set_status(idx: int, label: str, state: str = "running"):
        icons = {"running": "⏳", "done": "✅", "error": "❌"}
        status_boxes[idx].info(f"{icons[state]} {label}")

    for i, lbl in enumerate(labels):
        set_status(i, lbl, "running")

    result_placeholder = st.empty()

    with st.spinner("Running pipeline..."):
        try:
            result = run_pipeline(
                question=question.strip(),
                session_id=st.session_state.session_id,
            )
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

    # Update status indicators 
    sql_ok = not result.get("sql_error")
    set_status(0, labels[0], "done" if sql_ok else "error")
    set_status(1, labels[1], "done" if result.get("plotly_fig") else "error")
    set_status(2, labels[2], "done")
    set_status(3, labels[3], "done" if result.get("executive_summary") else "error")
    set_status(4, labels[4], "done" if result.get("pdf_path") else "error")

    # Error state
    if result.get("sql_error"):
        st.error(f"SQL generation failed: {result['sql_error']}")
        st.code(result.get("generated_sql", ""), language="sql")
        st.stop()

    st.divider()

    # Generated SQL
    with st.expander("Generated SQL", expanded=False):
        st.code(result.get("generated_sql", ""), language="sql")

    # Query result 
    df = result.get("query_result")
    if df is not None and not df.empty:
        with st.expander(f"Query Result — {len(df)} rows", expanded=False):
            st.dataframe(df, use_container_width=True)

    # Chart 
    st.subheader(f" {result.get('chart_title', 'Visualisation')}")
    st.caption(
        f"Chart type: **{result.get('chart_type', 'N/A')}** — "
        f"{result.get('chart_rationale', '')}"
    )
    plotly_fig = result.get("plotly_fig")
    if plotly_fig is not None:
        st.plotly_chart(plotly_fig, use_container_width=True)
    else:
        st.warning("No chart could be generated for this query result.")

    # Anomaly detection 
    st.subheader(" Anomaly Detection")
    st.caption(f"Method: **{result.get('anomaly_method', 'N/A').replace('_',' ').title()}**")
    st.write(result.get("anomaly_explanation", ""))

    anomaly_dates  = result.get("anomaly_dates",  [])
    anomaly_values = result.get("anomaly_values", [])
    if anomaly_dates:
        import pandas as pd
        anom_df = pd.DataFrame({"Date": anomaly_dates, "Value": anomaly_values})
        st.dataframe(anom_df, use_container_width=True)

    # Executive summary 
    st.subheader(" Executive Summary")
    summary = result.get("executive_summary", "")
    for line in summary.splitlines():
        line = line.strip()
        if line:
            st.markdown(line)

    # PDF download 
    st.divider()
    pdf_path = result.get("pdf_path", "")
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="⬇  Download Full PDF Report",
            data=pdf_bytes,
            file_name=os.path.basename(pdf_path),
            mime="application/pdf",
            type="primary",
            use_container_width=False,
        )
        st.caption(f"Saved to: `{pdf_path}`")
    else:
        st.warning("PDF could not be generated. Check logs for details.")

    # Save to history
    history_entry = {
        "question":       question.strip(),
        "timestamp":      datetime.now().strftime("%d %b %Y %H:%M"),
        "chart_type":     result.get("chart_type", ""),
        "anomaly_method": result.get("anomaly_method", ""),
        "pdf_path":       pdf_path,
    }
    st.session_state.history.append(history_entry)
    if len(st.session_state.history) > MAX_HISTORY:
        st.session_state.history = st.session_state.history[-MAX_HISTORY:]

elif run_btn and not question.strip():
    st.warning("Please enter a question before clicking Analyse.")
