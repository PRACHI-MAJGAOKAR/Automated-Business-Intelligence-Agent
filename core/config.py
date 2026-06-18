"""
core/config.py
--------------
Central configuration.  Change DB_URL to your PostgreSQL DSN in production:
    postgresql+psycopg2://user:password@localhost:5432/bi_agent
"""
from dotenv import load_dotenv
load_dotenv()
import os

# ── Database ──────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_URL: str = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(_BASE, 'bi_agent.db')}",
)

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# ── PDF output ────────────────────────────────────────────────────────────────
OUTPUT_DIR: str = os.path.join(_BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Agent confidence thresholds ───────────────────────────────────────────────
ANOMALY_CONTAMINATION: float = 0.05   # IsolationForest expected anomaly fraction
PROPHET_CHANGEPOINT_SCALE: float = 0.05
