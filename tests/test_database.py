"""Unit tests for SQLite Database Manager."""
import os
import pytest
import pandas as pd
from src.database.db_manager import ReviewDatabase

@pytest.fixture
def temp_db(tmp_path):
    """Fixture providing an isolated temporary SQLite database."""
    test_db_path = tmp_path / "test_sentiment.db"
    db = ReviewDatabase(str(test_db_path))
    return db

def test_db_init(temp_db):
    """Test database tables are initialized properly."""
    stats = temp_db.get_db_stats()
    assert stats["raw_reviews_count"] == 0
    assert stats["processed_reviews_count"] == 0
    assert stats["experiments_count"] == 0
    assert stats["prediction_logs_count"] == 0

def test_raw_reviews_insert_and_fetch(temp_db):
    """Test inserting and fetching raw review records."""
    dummy_data = pd.DataFrame([
        {
            "review_id": "rev_01",
            "product_id": "prod_A",
            "review_title": "Great product",
            "review_text": "I loved it so much!",
            "rating": 5.0,
            "timestamp": 1600000000,
            "helpful_vote": 2,
            "verified_purchase": 1
        },
        {
            "review_id": "rev_02",
            "product_id": "prod_B",
            "review_title": "Terrible",
            "review_text": "Broke immediately",
            "rating": 1.0,
            "timestamp": 1600000001,
            "helpful_vote": 0,
            "verified_purchase": 1
        }
    ])

    inserted = temp_db.insert_raw_reviews(dummy_data)
    assert inserted == 2

    fetched = temp_db.fetch_raw_reviews()
    assert len(fetched) == 2
    assert "review_id" in fetched.columns
    assert fetched.iloc[0]["review_id"] == "rev_01"

def test_experiment_logging(temp_db):
    """Test MLOps experiment tracking in SQLite."""
    exp_payload = {
        "model_name": "sentiment_test",
        "algorithm": "logistic_regression",
        "dataset_size": 1000,
        "train_size": 800,
        "test_size": 200,
        "accuracy": 0.945,
        "precision_macro": 0.92,
        "precision_positive": 0.97,
        "precision_negative": 0.85,
        "recall_macro": 0.91,
        "f1_macro": 0.915,
        "roc_auc": 0.98,
        "hyperparameters": {"C": 2.0},
        "model_path": "models/test.joblib"
    }

    exp_id = temp_db.log_experiment(exp_payload)
    assert exp_id == 1

    best_exp = temp_db.get_best_experiment()
    assert best_exp is not None
    assert best_exp["accuracy"] == 0.945
    assert best_exp["algorithm"] == "logistic_regression"

def test_prediction_logging(temp_db):
    """Test real-time prediction logging in SQLite."""
    log_id = temp_db.log_prediction(
        review_text="Best purchase ever!",
        predicted_sentiment="Positive",
        predicted_class=1,
        confidence=0.99,
        prob_positive=0.99,
        prob_negative=0.01,
        latency_ms=1.2
    )

    assert log_id == 1
    logs = temp_db.get_recent_predictions(limit=5)
    assert len(logs) == 1
    assert logs[0]["predicted_sentiment"] == "Positive"
    assert logs[0]["confidence"] == 0.99
