FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY main.py .

EXPOSE 8080

# HEALTHCHECK (container level)
HEALTHCHECK --interval=20s --timeout=3s --retries=3 CMD curl -f http://localhost:8080/healthz || exit 1

CMD ["python", "main.py"]
