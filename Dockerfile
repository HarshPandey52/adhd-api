FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /adhd-api

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/     ./app/
COPY model/   ./model/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
