"""
seed_db.py
----------
Downloads 2 years of OHLC data for 5 NSE stocks via yfinance and seeds
the SQLite database (same schema works with PostgreSQL — swap the
connection string in config.py).

Run once:  python data/seed_db.py
"""

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import os, sys

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH   = os.path.join(os.path.dirname(__file__), "..", "bi_agent.db")
DB_URL    = f"sqlite:///{os.path.abspath(DB_PATH)}"
SYMBOLS   = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS"]
END_DATE  = datetime.today()
START_DATE = END_DATE - timedelta(days=730)  # 2 years

CREATE_STOCK_TABLE = """
CREATE TABLE IF NOT EXISTS stock_data (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    date     DATE    NOT NULL,
    symbol   TEXT    NOT NULL,
    open     REAL    NOT NULL,
    high     REAL    NOT NULL,
    low      REAL    NOT NULL,
    close    REAL    NOT NULL,
    volume   INTEGER NOT NULL,
    UNIQUE(date, symbol)
)
"""

CREATE_SUMMARY_VIEW = """
CREATE VIEW IF NOT EXISTS stock_summary AS
SELECT
    symbol,
    COUNT(*)                        AS trading_days,
    ROUND(MIN(low),   2)            AS all_time_low,
    ROUND(MAX(high),  2)            AS all_time_high,
    ROUND(AVG(close), 2)            AS avg_close,
    ROUND(AVG(volume),0)            AS avg_volume,
    MIN(date)                       AS from_date,
    MAX(date)                       AS to_date
FROM stock_data
GROUP BY symbol
"""


def seed():
    engine = create_engine(DB_URL, echo=False)
    with engine.connect() as conn:
        conn.execute(text(CREATE_STOCK_TABLE))
        try:
            conn.execute(text(CREATE_SUMMARY_VIEW))
        except Exception:
            pass  # view already exists
        conn.commit()

    rows_inserted = 0
    for raw_sym in SYMBOLS:
        display_sym = raw_sym.replace(".NS", "")
        print(f"  Fetching {display_sym} ...", end=" ", flush=True)
        try:
            df = yf.download(
                raw_sym,
                start=START_DATE.strftime("%Y-%m-%d"),
                end=END_DATE.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                print("no data returned, skipping.")
                continue

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df["symbol"] = display_sym
            df["date"]   = pd.to_datetime(df["date"]).dt.date

            df = df[["date", "symbol", "open", "high", "low", "close", "volume"]]
            df = df.dropna()

            with engine.connect() as conn:
                for _, row in df.iterrows():
                    try:
                        conn.execute(
                            text("""
                                INSERT OR IGNORE INTO stock_data
                                (date, symbol, open, high, low, close, volume)
                                VALUES (:date, :symbol, :open, :high, :low, :close, :volume)
                            """),
                            {
                                "date":   str(row["date"]),
                                "symbol": row["symbol"],
                                "open":   float(row["open"]),
                                "high":   float(row["high"]),
                                "low":    float(row["low"]),
                                "close":  float(row["close"]),
                                "volume": int(row["volume"]),
                            },
                        )
                    except Exception:
                        pass
                conn.commit()

            rows_inserted += len(df)
            print(f"{len(df)} rows OK")

        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nDone. Total rows inserted: {rows_inserted}")
    print(f"Database: {os.path.abspath(DB_PATH)}")


if __name__ == "__main__":
    print("Seeding NSE stock data into SQLite...\n")
    seed()
