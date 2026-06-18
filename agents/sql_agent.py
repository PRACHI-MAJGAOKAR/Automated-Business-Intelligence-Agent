"""
agents/sql_agent.py

Agent 1 : Text-to-SQL

Responsibilities:
  1. Introspect the database schema once per call
  2. Ask Gemini Flash to generate a SELECT statement
  3. Execute the SQL via core/db.py (read-only enforced)
  4. On failure: re-prompt the LLM with the error and retry once
  5. Write generated_sql, query_result, sql_error into state
"""

from __future__ import annotations

import logging

from core.db import execute_query, get_schema_description
from core.llm import call_llm
from core.state import AgentState

logger = logging.getLogger(__name__)

_SQL_PROMPT = """\
You are a SQL expert. Given the database schema below and a user question,
write a single valid SQLite SELECT statement that answers the question.

Rules:
- Use only tables and columns that exist in the schema.
- Always alias aggregated columns with meaningful names (e.g. AVG(close) AS avg_close).
- For date filtering use: date >= '2024-01-01' format.
- Return AT MOST 500 rows unless the question specifically asks for all data.
- Do NOT use DML (INSERT, UPDATE, DELETE, DROP). SELECT only.
- Output ONLY the SQL statement, nothing else. No markdown, no explanation.

{schema}

User question: {question}

SQL:"""

_RETRY_PROMPT = """\
The SQL you generated caused an error. Fix it and return only the corrected SQL.

Original SQL:
{sql}

Error:
{error}

Schema:
{schema}

Corrected SQL:"""


def sql_agent(state: AgentState) -> AgentState:
    """LangGraph node: Text → SQL → DataFrame"""
    question = state.get("user_question", "").strip()
    if not question:
        return {**state, "sql_error": "No question provided.", "query_result": None}

    schema = get_schema_description()

    # First attempt
    sql = call_llm(_SQL_PROMPT.format(schema=schema, question=question)).strip()
    logger.info("[sql_agent] Generated SQL:\n%s", sql)

    df, error = execute_query(sql)

    # Retry once on error 
    if error:
        logger.warning("[sql_agent] Execution error: %s — retrying.", error)
        sql = call_llm(
            _RETRY_PROMPT.format(sql=sql, error=error, schema=schema)
        ).strip()
        logger.info("[sql_agent] Retry SQL:\n%s", sql)
        df, error = execute_query(sql)

    if error:
        logger.error("[sql_agent] Retry also failed: %s", error)
        return {
            **state,
            "generated_sql": sql,
            "sql_error":     error,
            "query_result":  None,
        }

    logger.info("[sql_agent] Query returned %d rows × %d cols.", *df.shape)
    return {
        **state,
        "generated_sql": sql,
        "sql_error":     "",
        "query_result":  df,
    }
