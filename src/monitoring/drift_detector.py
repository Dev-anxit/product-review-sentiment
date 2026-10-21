"""
MLOps Drift Detection and Production Monitoring.

Analyzes live prediction logs in the SQLite database to monitor:
1. Sentiment distribution drift (vs baseline training set)
2. Average prediction confidence
3. Latency benchmarks
4. Anomaly alerts
"""

from typing import Dict, Any, List
from src.utils import load_config, setup_logger
from src.database.db_manager import ReviewDatabase

logger = setup_logger("DriftDetector")

class DriftDetector:
    """
    Monitors live inference trends and flags statistical drift in reviews.
    """

    def __init__(self, db_path: str = "data/reviews_sentiment.db"):
        self.config = load_config()
        self.db = ReviewDatabase(db_path)

    def analyze_drift(self, sample_size: int = 100) -> Dict[str, Any]:
        """
        Analyze recent live prediction logs from the SQLite database.
        
        Args:
            sample_size: Number of recent predictions to evaluate.
            
        Returns:
            Dictionary containing drift statistics and alert statuses.
        """
        preds = self.db.get_recent_predictions(limit=sample_size)
        total_preds = len(preds)

        if total_preds == 0:
            return {
                "status": "INSUFFICIENT_DATA",
                "message": "No live predictions recorded in database yet.",
                "total_predictions_analyzed": 0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "avg_confidence": 0.0,
                "avg_latency_ms": 0.0,
                "drift_alert": False
            }

        pos_count = sum(1 for p in preds if p["predicted_sentiment"] == "Positive")
        neg_count = sum(1 for p in preds if p["predicted_sentiment"] == "Negative")
        confidences = [p["confidence"] for p in preds if p.get("confidence") is not None]
        latencies = [p["latency_ms"] for p in preds if p.get("latency_ms") is not None]

        pos_ratio = round(pos_count / total_preds, 4)
        neg_ratio = round(neg_count / total_preds, 4)
        avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

        drift_alert = False
        alert_reasons = []

        if avg_conf < 0.70:
            drift_alert = True
            alert_reasons.append(f"Low average model confidence ({avg_conf * 100:.1f}%). Possible out-of-distribution text.")

        if total_preds >= 20 and (pos_ratio > 0.98 or neg_ratio > 0.90):
            drift_alert = True
            alert_reasons.append(f"Extreme sentiment skew detected (Pos: {pos_ratio*100:.1f}%, Neg: {neg_ratio*100:.1f}%).")

        return {
            "status": "DRIFT_DETECTED" if drift_alert else "NORMAL",
            "total_predictions_analyzed": total_preds,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "positive_ratio": pos_ratio,
            "negative_ratio": neg_ratio,
            "avg_confidence": avg_conf,
            "avg_latency_ms": avg_lat,
            "drift_alert": drift_alert,
            "alert_reasons": alert_reasons
        }
