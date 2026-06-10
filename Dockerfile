# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System deps needed by MNE ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /adhd-api

# ── Install Python deps first (layer-cached) ──────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY app/     ./app/
COPY model/   ./model/

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Run with uvicorn ──────────────────────────────────────────────────────────
# PORT env var is set automatically by Cloud Run & Render
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
