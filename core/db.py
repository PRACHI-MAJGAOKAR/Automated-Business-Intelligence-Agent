"""
core/db.py
----------
Database utilities: engine singleton, schema introspection, safe query
execution.  All queries go through execute_query() which enforces
read-only semantics by rejecting DML keywords.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, inspect as sa_inspect, text
from sqlalchemy.engine import Engine

from core.config import DB_URL

# ── Engine singleton ──────────────────────────────────────────────────────────
_engine: Optional[Engine] = None

_DML_PATTERN = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE)\b",
    re.IGNORECASE | re.MULTILINE,
)


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL, echo=False)
    return _engine


# ── Schema ────────────────────────────────────────────────────────────────────
def get_schema_description() -> str:
    """
    Returns a concise natural-language schema description that is injected
    into the SQL-generation prompt so the LLM knows exactly what tables,
    columns, and data types are available.
    """
    engine = get_engine()
    inspector = sa_inspect(engine)
    lines: list[str] = ["Database schema:\n"]

    for table_name in inspector.get_table_names():
        cols = inspector.get_columns(table_name)
        col_parts = [f"{c['name']} ({str(c['type'])})" for c in cols]
        lines.append(f"  Table '{table_name}': {', '.join(col_parts)}")

        # Include sample values for categorical / low-cardinality text columns
        for col in cols:
            if str(col["type"]).startswith("TEXT") or str(col["type"]).startswith("VARCHAR"):
                try:
                    with engine.connect() as conn:
                        result = conn.execute(
                            text(
                                f"SELECT DISTINCT {col['name']} FROM {table_name} "
                                f"LIMIT 8"
                            )
                        )
                        vals = [str(r[0]) for r in result if r[0] is not None]
                    if vals:
                        lines.append(
                            f"    '{col['name']}' sample values: {', '.join(vals)}"
                        )
                except Exception:
                    pass

    return "\n".join(lines)


# ── Query execution ───────────────────────────────────────────────────────────
def execute_query(sql: str) -> tuple[pd.DataFrame, str]:
    """
    Execute a SELECT statement and return (DataFrame, error_message).
    Rejects any statement containing DML/DDL keywords.
    Returns (empty DataFrame, error_message) on failure.
    """
    # Safety: reject DML
    if _DML_PATTERN.search(sql):
        return pd.DataFrame(), "Rejected: only SELECT statements are permitted."

    # Strip markdown code fences if the LLM wrapped the SQL
    sql = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE).strip().strip("`").strip()

    engine = get_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        return df, ""
    except Exception as exc:
        return pd.DataFrame(), str(exc)
