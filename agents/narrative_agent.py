"""
agents/narrative_agent.py
--------------------------
Agent 4 — Narrative / Executive Summary

Sends the full analysis context to Gemini Flash and asks it to write
a 3-line executive summary that a business stakeholder can act on.

Writes: executive_summary into state.
"""

from __future__ import annotations

import logging

import pandas as pd

from core.llm import call_llm
from core.state import AgentState

logger = logging.getLogger(__name__)

_NARRATIVE_PROMPT = """\
You are a senior business intelligence analyst writing for a C-suite audience.
Given the analysis results below, write EXACTLY 3 concise bullet points
(each one sentence, starting with "•") that summarise:
  1. What the data shows (the main finding)
  2. The most important anomaly or trend detected
  3. A concrete, actionable recommendation

Keep the language direct. No filler phrases like "it is important to note".
No markdown headers. Only the 3 bullet points.

--- Analysis Context ---
User question  : {question}
SQL executed   : {sql}
Rows returned  : {rows}
Columns        : {columns}
Data sample    : {sample}
Chart type     : {chart_type}
Chart rationale: {chart_rationale}
Anomaly method : {anomaly_method}
Anomaly summary: {anomaly_explanation}

3-line executive summary:"""


def narrative_agent(state: AgentState) -> AgentState:
    """LangGraph node: full state context → 3-line executive summary"""
    df: pd.DataFrame | None = state.get("query_result")

    rows    = len(df)    if df is not None else 0
    cols    = list(df.columns) if df is not None else []
    sample  = df.head(3).to_string(index=False) if df is not None and not df.empty else "No data"

    prompt = _NARRATIVE_PROMPT.format(
        question         = state.get("user_question", ""),
        sql              = state.get("generated_sql", ""),
        rows             = rows,
        columns          = cols,
        sample           = sample,
        chart_type       = state.get("chart_type", ""),
        chart_rationale  = state.get("chart_rationale", ""),
        anomaly_method   = state.get("anomaly_method", ""),
        anomaly_explanation = state.get("anomaly_explanation", ""),
    )

    try:
        summary = call_llm(prompt).strip()
    except Exception as exc:
        logger.error("[narrative_agent] LLM call failed: %s", exc)
        summary = (
            "• Data analysis completed successfully.\n"
            "• Anomaly detection was applied to the result set.\n"
            "• Review the chart and anomaly details above for actionable insights."
        )

    logger.info("[narrative_agent] Summary generated (%d chars).", len(summary))
    return {**state, "executive_summary": summary}
