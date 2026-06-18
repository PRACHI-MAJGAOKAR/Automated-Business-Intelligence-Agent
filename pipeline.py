"""
pipeline.py
-----------
LangGraph pipeline — wires all 5 agents into a directed state graph.

Graph structure:

  START
    │
    ▼
  sql_agent          (Text → SQL → DataFrame)
    │
    ├─ [sql failed] ──────────────────────────► END  (error state)
    │
    ▼
  viz_agent          (DataFrame → chart type → Plotly fig + PNG)
    │
    ▼
  anomaly_agent      (DataFrame → Prophet / IsolationForest → flags)
    │
    ▼
  narrative_agent    (full context → 3-line executive summary)
    │
    ▼
  pdf_agent          (full state → PDF written to disk)
    │
    ▼
  END
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END

from agents.sql_agent       import sql_agent
from agents.viz_agent       import viz_agent
from agents.anomaly_agent   import anomaly_agent
from agents.narrative_agent import narrative_agent
from agents.pdf_agent       import pdf_agent
from core.state             import AgentState

logger = logging.getLogger(__name__)


# ── Routing function ──────────────────────────────────────────────────────────

def _route_after_sql(state: AgentState) -> Literal["viz_agent", "__end__"]:
    """
    If the SQL agent returned an error, short-circuit to END.
    Otherwise continue to visualisation.
    """
    if state.get("sql_error"):
        logger.warning(
            "[pipeline] SQL agent failed — short-circuiting. Error: %s",
            state["sql_error"],
        )
        return END
    return "viz_agent"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("sql_agent",       sql_agent)
    graph.add_node("viz_agent",       viz_agent)
    graph.add_node("anomaly_agent",   anomaly_agent)
    graph.add_node("narrative_agent", narrative_agent)
    graph.add_node("pdf_agent",       pdf_agent)

    # Entry point
    graph.add_edge(START, "sql_agent")

    # Conditional routing after SQL
    graph.add_conditional_edges(
        "sql_agent",
        _route_after_sql,
        {"viz_agent": "viz_agent", END: END},
    )

    # Linear chain after visualisation
    graph.add_edge("viz_agent",       "anomaly_agent")
    graph.add_edge("anomaly_agent",   "narrative_agent")
    graph.add_edge("narrative_agent", "pdf_agent")
    graph.add_edge("pdf_agent",       END)

    return graph.compile()


# ── Public singleton ──────────────────────────────────────────────────────────
pipeline = build_pipeline()


def run_pipeline(question: str, session_id: str = "default") -> AgentState:
    """
    Entry point called by the Streamlit app.
    Returns the final AgentState dict.
    """
    initial_state: AgentState = {
        "user_question": question,
        "session_id":    session_id,
    }
    logger.info("[pipeline] Starting run for question: %r", question)
    result: AgentState = pipeline.invoke(initial_state)
    logger.info("[pipeline] Run complete. PDF at: %s", result.get("pdf_path", "N/A"))
    return result
