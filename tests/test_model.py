"""Unit tests for model inference and artifacts."""
import pytest
from src.models.predict import SentimentPredictor

def test_sentiment_predictor_inference():
    """Verify that SentimentPredictor produces accurate classifications."""
    predictor = SentimentPredictor()

    pos_text = "I absolutely love this serum! It is amazing and works like a charm."
    neg_text = "Horrible waste of money. Completely useless, broke immediately."

    res_pos = predictor.predict(pos_text, log_to_db=False)
    res_neg = predictor.predict(neg_text, log_to_db=False)

    assert res_pos["sentiment"] == "Positive"
    assert res_pos["label"] == 1
    assert res_pos["confidence"] > 0.80

    assert res_neg["sentiment"] == "Negative"
    assert res_neg["label"] == 0
    assert res_neg["confidence"] > 0.80

def test_batch_prediction():
    """Verify batch prediction functionality."""
    predictor = SentimentPredictor()
    batch = [
        "Great quality and fast shipping!",
        "Poor quality, terrible customer service."
    ]
    results = predictor.predict_batch(batch, log_to_db=False)
    assert len(results) == 2
    assert results[0]["sentiment"] == "Positive"
    assert results[1]["sentiment"] == "Negative"
