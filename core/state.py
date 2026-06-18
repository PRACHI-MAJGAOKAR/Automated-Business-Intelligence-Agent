"""
core/state.py
-------------
Single TypedDict that flows through every LangGraph node.
Every agent reads from and writes to this object — nothing is
passed between agents directly.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    user_question: str          # raw natural-language question from the user

    # ── SQL agent outputs ─────────────────────────────────────────────────────
    generated_sql:  str         # SQL produced by the LLM
    sql_error:      str         # non-empty if execution failed
    query_result:   Any         # pd.DataFrame returned by the query

    # ── Visualisation agent outputs ───────────────────────────────────────────
    chart_type:     str         # bar | line | scatter | heatmap | pie | histogram
    chart_title:    str
    chart_rationale: str        # one sentence: why this chart type was chosen
    plotly_fig:     Any         # plotly Figure object (shown in Streamlit)
    chart_png:      bytes       # matplotlib PNG (embedded in PDF)

    # ── Anomaly detection outputs ─────────────────────────────────────────────
    anomaly_method:     str     # "prophet" | "isolation_forest" | "none"
    anomaly_dates:      list    # list of date strings where anomalies were flagged
    anomaly_values:     list    # corresponding values
    anomaly_explanation: str    # human-readable paragraph from narrative agent

    # ── Narrative agent outputs ───────────────────────────────────────────────
    executive_summary: str      # 3-line summary written by the LLM

    # ── PDF output ────────────────────────────────────────────────────────────
    pdf_path: str               # absolute path to the written PDF report

    # ── Session history (managed by Streamlit layer) ──────────────────────────
    session_id: str
    error:      str             # pipeline-level error, if any
