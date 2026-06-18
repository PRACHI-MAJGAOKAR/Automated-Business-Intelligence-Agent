"""
agents/anomaly_agent.py
-----------------------
Agent 3 — Anomaly Detection

Decision logic:
  - If the DataFrame has a date/datetime column AND a numeric column
    → use Facebook Prophet (time-series anomaly detection via residuals)
  - Otherwise
    → use IsolationForest on all numeric columns
  - If fewer than 10 rows: skip (not enough data)

Writes into state:
  anomaly_method, anomaly_dates, anomaly_values, anomaly_explanation
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from core.config import ANOMALY_CONTAMINATION, PROPHET_CHANGEPOINT_SCALE
from core.state import AgentState

logger = logging.getLogger(__name__)

# Suppress Prophet / cmdstanpy verbose logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_date_col(df: pd.DataFrame) -> str | None:
    """Return the first column that can be parsed as dates, or None.
    Supports object, string (StringDtype), and native datetime64 columns.
    Note: infer_datetime_format was removed in pandas 2.2 — not used here.
    """
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        # Handle object, pandas StringDtype ('string'), and pandas 3 str dtype ('str')
        col_dtype = str(df[col].dtype)
        if col_dtype in ("object", "string", "str") or col_dtype.startswith("string"):
            try:
                pd.to_datetime(df[col])
                return col
            except Exception:
                pass
    return None


def _find_numeric_col(df: pd.DataFrame, exclude: str | None = None) -> str | None:
    """Return the first numeric column other than the excluded one."""
    for col in df.select_dtypes(include="number").columns:
        if col != exclude:
            return col
    return None


# ── Prophet path ──────────────────────────────────────────────────────────────

def _prophet_anomalies(
    df: pd.DataFrame, date_col: str, value_col: str
) -> tuple[list[str], list[float]]:
    """
    Fit Prophet on (date, value), compute residuals, flag points where
    |residual| > 2 * residual standard deviation as anomalies.
    Returns (list of date strings, list of values).
    """
    from prophet import Prophet

    ts = df[[date_col, value_col]].copy()
    ts.columns = ["ds", "y"]
    ts["ds"] = pd.to_datetime(ts["ds"])
    ts = ts.dropna().sort_values("ds")

    if len(ts) < 10:
        return [], []

    m = Prophet(
        changepoint_prior_scale=PROPHET_CHANGEPOINT_SCALE,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality=False,
        interval_width=0.95,
    )
    m.fit(ts)

    forecast  = m.predict(ts[["ds"]])
    residuals = ts["y"].values - forecast["yhat"].values
    std_res   = residuals.std()

    if std_res == 0:
        return [], []

    mask = np.abs(residuals) > 2.0 * std_res
    anomaly_df = ts[mask].copy()
    anomaly_df["ds"] = anomaly_df["ds"].dt.strftime("%Y-%m-%d")

    return anomaly_df["ds"].tolist(), anomaly_df["y"].tolist()


# ── IsolationForest path ──────────────────────────────────────────────────────

def _isolation_forest_anomalies(
    df: pd.DataFrame,
) -> tuple[list[int], list[dict]]:
    """
    Run IsolationForest on all numeric columns.
    Returns (list of row indices, list of {column: value} dicts for flagged rows).
    """
    numeric_df = df.select_dtypes(include="number").dropna(axis=1)
    if numeric_df.empty or len(numeric_df) < 10:
        return [], []

    iso    = IsolationForest(
        contamination=ANOMALY_CONTAMINATION,
        random_state=42,
        n_estimators=100,
    )
    labels = iso.fit_predict(numeric_df)
    mask   = labels == -1
    flagged_indices = list(numeric_df.index[mask])
    flagged_rows    = numeric_df[mask].to_dict(orient="records")

    return flagged_indices, flagged_rows


# ── Explanation builder ───────────────────────────────────────────────────────

def _build_explanation(
    method: str,
    anomaly_dates: list,
    anomaly_values: list,
    flagged_rows: list[dict] | None,
    value_col: str | None,
) -> str:
    if method == "none":
        return "No anomalies detected — the data appears consistent with expected patterns."

    if method == "prophet" and anomaly_dates:
        top_n = min(5, len(anomaly_dates))
        items = [
            f"{anomaly_dates[i]} (value: {anomaly_values[i]:.2f})"
            for i in range(top_n)
        ]
        return (
            f"Prophet time-series analysis flagged {len(anomaly_dates)} anomalous "
            f"data point(s) where observed values deviated more than 2 standard "
            f"deviations from the fitted trend. "
            f"Most notable: {', '.join(items)}. "
            f"These may indicate unusual market events, data entry errors, or "
            f"genuine structural breaks in the series."
        )

    if method == "isolation_forest" and flagged_rows:
        return (
            f"Isolation Forest flagged {len(flagged_rows)} row(s) as anomalous "
            f"across the numeric feature space. "
            f"These records are statistically isolated from the majority of the "
            f"data and may warrant further investigation."
        )

    return "Anomaly detection ran but found no significant outliers in this dataset."


# ── LangGraph node ────────────────────────────────────────────────────────────

def anomaly_agent(state: AgentState) -> AgentState:
    """LangGraph node: DataFrame → anomaly flags + explanation"""
    df: pd.DataFrame | None = state.get("query_result")

    if df is None or (hasattr(df, "empty") and df.empty) or len(df) < 10:
        logger.info("[anomaly_agent] Skipping — insufficient data.")
        return {
            **state,
            "anomaly_method":      "none",
            "anomaly_dates":       [],
            "anomaly_values":      [],
            "anomaly_explanation": "Insufficient data for anomaly detection (fewer than 10 rows).",
        }

    date_col  = _find_date_col(df)
    value_col = _find_numeric_col(df, exclude=None)

    anomaly_dates:  list[Any] = []
    anomaly_values: list[Any] = []
    flagged_rows:   list[dict] | None = None
    method = "none"

    # ── Route: Prophet for time series, IsolationForest otherwise ────────────
    if date_col and value_col:
        logger.info(
            "[anomaly_agent] Using Prophet on (%s, %s).", date_col, value_col
        )
        try:
            anomaly_dates, anomaly_values = _prophet_anomalies(df, date_col, value_col)
            method = "prophet"
        except Exception as exc:
            logger.error("[anomaly_agent] Prophet failed: %s — falling back to IsolationForest.", exc)
            _, flagged_rows_raw = _isolation_forest_anomalies(df)
            flagged_rows   = flagged_rows_raw if flagged_rows_raw else []
            method         = "isolation_forest"
    else:
        logger.info("[anomaly_agent] Using IsolationForest (no date column found).")
        _, flagged_rows_raw = _isolation_forest_anomalies(df)
        flagged_rows = flagged_rows_raw if flagged_rows_raw else []
        method       = "isolation_forest"

    if method == "prophet":
        logger.info("[anomaly_agent] Prophet found %d anomalies.", len(anomaly_dates))
    else:
        n = len(flagged_rows) if flagged_rows else 0
        logger.info("[anomaly_agent] IsolationForest found %d anomalies.", n)

    explanation = _build_explanation(
        method, anomaly_dates, anomaly_values, flagged_rows, value_col
    )

    return {
        **state,
        "anomaly_method":      method,
        "anomaly_dates":       anomaly_dates,
        "anomaly_values":      anomaly_values,
        "anomaly_explanation": explanation,
    }
