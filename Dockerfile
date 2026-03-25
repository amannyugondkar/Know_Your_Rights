FROM python:3.11-slim

WORKDIR /app

# System deps for pdfplumber (pdfminer.six) and general SSL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cloud platforms set PORT; default to 8000 locally
ENV PORT=8000

CMD ["bash", "-lc", "uvicorn knowrights_pipeline:app --host 0.0.0.0 --port ${PORT}"]
