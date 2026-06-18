"""
agents/viz_agent.py
-------------------
Agent 2 — Visualisation

Responsibilities:
  1. Send the DataFrame shape, column names, dtypes, and sample rows
     to Gemini Flash and ask it to select the best chart type
  2. Render an interactive Plotly figure (for Streamlit)
  3. Render a static matplotlib PNG (for the PDF report)
  4. Write chart_type, chart_title, chart_rationale, plotly_fig,
     chart_png into state
"""

from __future__ import annotations

import io
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.llm import call_structured
from core.state import AgentState

logger = logging.getLogger(__name__)

_VIZ_PROMPT = """\
You are a data visualisation expert. Given the DataFrame metadata below,
decide the single best chart type to answer the user's question visually.

Available chart types: bar, line, scatter, heatmap, pie, histogram

DataFrame metadata:
  Shape     : {rows} rows × {cols} columns
  Columns   : {columns}
  Dtypes    : {dtypes}
  Sample (first 3 rows):
{sample}

User question: {question}

Return a JSON object with exactly these keys:
{{
  "chart_type"   : "<one of: bar | line | scatter | heatmap | pie | histogram>",
  "x_column"     : "<column name for x-axis, or null if not applicable>",
  "y_column"     : "<column name for y-axis, or primary value column>",
  "color_column" : "<column to use for color grouping, or null>",
  "chart_title"  : "<concise descriptive title for the chart>",
  "rationale"    : "<one sentence explaining why this chart type was chosen>"
}}"""


# ── Plotly renderer ───────────────────────────────────────────────────────────

def _render_plotly(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str | None,
    y_col: str | None,
    color_col: str | None,
    title: str,
) -> go.Figure:
    """Build and return a Plotly Figure based on chart_type."""

    # Validate column names — fall back to first available col if LLM hallucinated
    all_cols = list(df.columns)

    def safe_col(name: str | None) -> str | None:
        if name and name in all_cols:
            return name
        return None

    x    = safe_col(x_col)
    y    = safe_col(y_col)
    clr  = safe_col(color_col)

    # If y is still None, pick first numeric column
    if y is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        y = numeric_cols[0] if numeric_cols else all_cols[0]

    # If x is still None, pick first non-y column
    if x is None:
        x = next((c for c in all_cols if c != y), None)

    common_kwargs = dict(title=title, template="plotly_white")

    try:
        if chart_type == "line":
            fig = px.line(df, x=x, y=y, color=clr, **common_kwargs)

        elif chart_type == "bar":
            fig = px.bar(df, x=x, y=y, color=clr, barmode="group", **common_kwargs)

        elif chart_type == "scatter":
            fig = px.scatter(df, x=x, y=y, color=clr, **common_kwargs)

        elif chart_type == "heatmap":
            # Build a pivot table for heatmap: needs at least 2 categorical columns
            cat_cols   = df.select_dtypes(exclude="number").columns.tolist()
            num_cols   = df.select_dtypes(include="number").columns.tolist()
            if len(cat_cols) >= 2 and num_cols:
                pivot = df.pivot_table(
                    index=cat_cols[0], columns=cat_cols[1],
                    values=num_cols[0], aggfunc="mean"
                )
                fig = px.imshow(pivot, **common_kwargs)
            else:
                # Fallback to correlation heatmap on numeric columns
                corr  = df.select_dtypes(include="number").corr()
                fig   = px.imshow(corr, text_auto=True, **common_kwargs)

        elif chart_type == "pie":
            fig = px.pie(df, names=x, values=y, **common_kwargs)

        elif chart_type == "histogram":
            fig = px.histogram(df, x=y, color=clr, **common_kwargs)

        else:
            # Safe default
            fig = px.bar(df, x=x, y=y, **common_kwargs)

    except Exception as exc:
        logger.warning("[viz_agent] Plotly render failed (%s), using fallback bar.", exc)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        y_fb = numeric_cols[0] if numeric_cols else all_cols[0]
        x_fb = next((c for c in all_cols if c != y_fb), None)
        fig  = px.bar(df, x=x_fb, y=y_fb, title=title, template="plotly_white")

    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig


# ── Matplotlib PNG renderer ───────────────────────────────────────────────────

def _render_matplotlib_png(fig_plotly: go.Figure, title: str) -> bytes:
    """
    Render a static PNG via matplotlib by re-reading the plotly figure's data.
    Used only for PDF embedding — Streamlit uses the Plotly figure directly.
    """
    mpl_fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax.set_title(title, fontsize=13, pad=12)

    rendered = False
    for trace in fig_plotly.data:
        try:
            trace_type = trace.type

            if trace_type in ("bar", "histogram"):
                x_vals = list(trace.x) if trace.x is not None else []
                y_vals = list(trace.y) if trace.y is not None else []
                if x_vals and y_vals:
                    ax.bar(range(len(x_vals)), y_vals, tick_label=x_vals)
                    plt.xticks(rotation=45, ha="right")
                    rendered = True

            elif trace_type in ("scatter", "scattergl"):
                x_vals = list(trace.x) if trace.x is not None else []
                y_vals = list(trace.y) if trace.y is not None else []
                mode   = getattr(trace, "mode", "lines")
                if x_vals and y_vals:
                    if "lines" in str(mode):
                        ax.plot(x_vals, y_vals, linewidth=1.8)
                    else:
                        ax.scatter(x_vals, y_vals, s=20)
                    rendered = True

            elif trace_type == "heatmap":
                import numpy as np
                z = trace.z
                if z is not None:
                    im = ax.imshow(z, aspect="auto", cmap="Blues")
                    mpl_fig.colorbar(im, ax=ax)
                    rendered = True

            elif trace_type == "pie":
                labels = list(trace.labels) if trace.labels is not None else []
                values = list(trace.values) if trace.values is not None else []
                if labels and values:
                    ax.pie(values, labels=labels, autopct="%1.1f%%")
                    rendered = True

        except Exception:
            continue

    if not rendered:
        ax.text(0.5, 0.5, "Chart data unavailable",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)

    ax.grid(True, alpha=0.3, linestyle="--")
    mpl_fig.tight_layout()

    buf = io.BytesIO()
    mpl_fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    png_bytes = buf.read()
    plt.close(mpl_fig)
    return png_bytes


# ── LangGraph node ────────────────────────────────────────────────────────────

def viz_agent(state: AgentState) -> AgentState:
    """LangGraph node: DataFrame → chart type decision → Plotly fig + PNG"""
    df = state.get("query_result")
    if df is None or (hasattr(df, "empty") and df.empty):
        logger.warning("[viz_agent] No data to visualise.")
        return {**state, "chart_type": "none", "plotly_fig": None, "chart_png": b""}

    question = state.get("user_question", "")

    # ── Ask LLM to choose chart type ─────────────────────────────────────────
    sample_str = df.head(3).to_string(index=False)
    dtypes_str = ", ".join(f"{c}: {t}" for c, t in df.dtypes.items())

    try:
        decision = call_structured(
            _VIZ_PROMPT.format(
                rows=len(df), cols=len(df.columns),
                columns=list(df.columns),
                dtypes=dtypes_str,
                sample=sample_str,
                question=question,
            )
        )
    except ValueError as exc:
        logger.error("[viz_agent] LLM structured call failed: %s", exc)
        # Safe fallback values
        decision = {
            "chart_type":    "bar",
            "x_column":      df.columns[0] if len(df.columns) > 0 else None,
            "y_column":      df.columns[1] if len(df.columns) > 1 else df.columns[0],
            "color_column":  None,
            "chart_title":   question[:80],
            "rationale":     "Bar chart selected as safe default after LLM error.",
        }

    chart_type  = decision.get("chart_type",   "bar")
    x_col       = decision.get("x_column")
    y_col       = decision.get("y_column")
    color_col   = decision.get("color_column")
    title       = decision.get("chart_title",  question[:80])
    rationale   = decision.get("rationale",    "")

    logger.info("[viz_agent] Chose chart_type='%s' — %s", chart_type, rationale)

    # ── Render ────────────────────────────────────────────────────────────────
    plotly_fig = _render_plotly(df, chart_type, x_col, y_col, color_col, title)
    chart_png  = _render_matplotlib_png(plotly_fig, title)

    return {
        **state,
        "chart_type":      chart_type,
        "chart_title":     title,
        "chart_rationale": rationale,
        "plotly_fig":      plotly_fig,
        "chart_png":       chart_png,
    }
