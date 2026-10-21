"""
Inference Engine and Prediction Service.

Loads the trained model and vectorizer, executes real-time or batch sentiment
predictions with probability scores and latency measurement, and automatically
records predictions into the SQLite database for monitoring.
"""

import time
import joblib
import argparse
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.utils import load_config, setup_logger, get_project_root
from src.data.preprocess import clean_text
from src.database.db_manager import ReviewDatabase

logger = setup_logger("InferenceEngine")

class SentimentPredictor:
    """
    Production-grade inference engine for sentiment analysis.
    """

    def __init__(self, model_path: Optional[str] = None, vectorizer_path: Optional[str] = None):
        self.config = load_config()
        self.root = get_project_root()
        
        default_model = self.root / self.config.get("model", {}).get("save_dir", "models") / self.config.get("model", {}).get("model_filename", "sentiment_model.joblib")
        default_vec = self.root / self.config.get("model", {}).get("save_dir", "models") / self.config.get("model", {}).get("vectorizer_filename", "vectorizer.joblib")
        
        self.model_path = Path(model_path) if model_path else default_model
        self.vectorizer_path = Path(vectorizer_path) if vectorizer_path else default_vec
        
        self.db_path = self.config.get("database", {}).get("db_path", "data/reviews_sentiment.db")
        self.db = ReviewDatabase(self.db_path)
        
        self.model = None
        self.vectorizer = None
        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Load serialized model and vectorizer into memory."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}. Train model first!")
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found at: {self.vectorizer_path}. Train model first!")

        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        logger.info("Model and vectorizer artifacts successfully loaded into memory.")

    def predict(self, text: str, log_to_db: bool = True) -> Dict[str, Any]:
        """
        Predict sentiment for a single review string.
        
        Args:
            text: Raw review text or title.
            log_to_db: Whether to record prediction in SQLite for monitoring.
            
        Returns:
            Dict containing sentiment, probabilities, confidence, and latency.
        """
        start_time = time.perf_counter()

        cleaned = clean_text(text)
        if not cleaned:
            cleaned = "neutral"

        vec = self.vectorizer.transform([cleaned])

        pred_label = int(self.model.predict(vec)[0])
        
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec)[0]
            prob_neg = float(probs[0])
            prob_pos = float(probs[1])
            confidence = float(max(prob_pos, prob_neg))
        else:
            prob_pos = 1.0 if pred_label == 1 else 0.0
            prob_neg = 1.0 - prob_pos
            confidence = 1.0

        sentiment = "Positive" if pred_label == 1 else "Negative"
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        if log_to_db:
            try:
                self.db.log_prediction(
                    review_text=text,
                    predicted_sentiment=sentiment,
                    predicted_class=pred_label,
                    confidence=confidence,
                    prob_positive=prob_pos,
                    prob_negative=prob_neg,
                    latency_ms=latency_ms
                )
            except Exception as e:
                logger.warning(f"Could not log prediction to database: {e}")

        return {
            "text": text,
            "cleaned_text": cleaned,
            "sentiment": sentiment,
            "label": pred_label,
            "confidence": round(confidence, 4),
            "prob_positive": round(prob_pos, 4),
            "prob_negative": round(prob_neg, 4),
            "latency_ms": latency_ms
        }

    def predict_batch(self, texts: List[str], log_to_db: bool = False) -> List[Dict[str, Any]]:
        """Run batch predictions efficiently."""
        return [self.predict(t, log_to_db=log_to_db) for t in texts]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment of a product review")
    parser.add_argument("--text", type=str, required=True, help="Review text to analyze")
    args = parser.parse_args()

    predictor = SentimentPredictor()
    result = predictor.predict(args.text)
    
    print("\n" + "=" * 50)
    print("REVIEW SENTIMENT PREDICTION")
    print("=" * 50)
    print(f"Review:      \"{result['text']}\"")
    print(f"Sentiment:   {result['sentiment']} (Confidence: {result['confidence'] * 100:.1f}%)")
    print(f"Probability: Positive={result['prob_positive']:.4f} | Negative={result['prob_negative']:.4f}")
    print(f"Latency:     {result['latency_ms']} ms")
    print("=" * 50 + "\n")
