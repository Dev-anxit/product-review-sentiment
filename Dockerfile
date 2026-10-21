FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements-prod.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY models/ models/
COPY data/reviews_sentiment.db data/reviews_sentiment.db
COPY api/ api/
COPY dashboard/ dashboard/
COPY src/ src/
COPY run_pipeline.py .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
