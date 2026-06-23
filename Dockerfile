#Build 
FROM python:3.11-slim AS base

WORKDIR /app

# System deps: Prophet needs libgomp; ReportLab needs libfreetype
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libfreetype6-dev \
        && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p /app/output

# Runtime
EXPOSE 8501

# Environment variables (override at runtime with -e or docker-compose)
ENV GEMINI_API_KEY=""
ENV DATABASE_URL="sqlite:////app/bi_agent.db"
ENV GEMINI_MODEL="gemini-2.0-flash"

# Seed the database on first run, then start Streamlit
CMD ["sh", "-c", "python data/seed_db.py && streamlit run app.py \
     --server.port=8501 \
     --server.address=0.0.0.0 \
     --server.headless=true \
     --browser.gatherUsageStats=false"]
