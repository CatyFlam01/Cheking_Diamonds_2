# Multi-stage Dockerfile для оптимизации размера образа
# Автор: [Ваше Имя]

FROM python:3.9-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.9-slim as production

LABEL maintainer="[your.email@example.com]"
LABEL version="1.0"
LABEL description="Diamonds Price Prediction API"

RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

COPY --from=builder /root/.local /home/appuser/.local

COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

ENV PATH=/home/appuser/.local/bin:$PATH

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
