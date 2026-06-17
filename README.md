# Business Intelligence Agent

An AI-powered BI system that answers plain-English questions about NSE stock data,
selects chart types autonomously, detects anomalies, and generates PDF reports.

## Architecture

5-agent LangGraph pipeline:
1. **SQL Agent** - Text-to-SQL with retry on execution failure
2. **Viz Agent** - LLM selects chart type based on data shape (bar/line/scatter/heatmap/pie)
3. **Anomaly Agent** - Prophet for time series, IsolationForest for tabular data
4. **Narrative Agent** - 3-line executive summary via Gemini Flash
5. **PDF Agent** - Full ReportLab report with chart, anomaly table, and summary

## Stack

LangGraph · Gemini Flash · Prophet · Scikit-learn · Plotly · Streamlit · ReportLab · SQLite/PostgreSQL · Docker

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GEMINI_API_KEY=your_key_here" > .env

# Seed the database
python data/seed_db.py

# Run
streamlit run app.py
```

## Usage

Ask any business question in plain English:
- *Show me the closing price trend for RELIANCE over the last 6 months*
- *Which stock had the highest average volume in 2024?*
- *Compare average closing prices of all 5 stocks*
