"""Unit tests for FastAPI endpoints."""
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Verify system health status endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["database"] == "connected"

def test_predict_endpoint_positive():
    """Verify single prediction endpoint for positive review."""
    payload = {"text": "Excellent and wonderful experience. Would buy again!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "Positive"
    assert data["confidence"] > 0.70
    assert "latency_ms" in data

def test_predict_endpoint_negative():
    """Verify single prediction endpoint for negative review."""
    payload = {"text": "Terrible product, completely defective and disappointing."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "Negative"
    assert data["confidence"] > 0.70

def test_batch_predict_endpoint():
    """Verify batch prediction endpoint."""
    payload = {
        "reviews": [
            "I love this so much, wonderful!",
            "Horrible experience, never again."
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_reviews"] == 2
    assert data["predictions"][0]["sentiment"] == "Positive"
    assert data["predictions"][1]["sentiment"] == "Negative"

def test_metrics_endpoint():
    """Verify model metrics endpoint returns valid metadata."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "latest_metrics" in data
    assert "accuracy" in data["latest_metrics"]
    assert data["latest_metrics"]["accuracy"] > 0.90

def test_db_stats_endpoint():
    """Verify database stats endpoint."""
    response = client.get("/db/stats")
    assert response.status_code == 200
    data = response.json()
    assert "raw_reviews_count" in data
    assert "processed_reviews_count" in data
