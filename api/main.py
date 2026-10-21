"""
FastAPI Production Application for Product Review Sentiment.

Provides RESTful endpoints for:
1. Real-time single & batch review sentiment predictions
2. Database statistics & live inference monitoring
3. Model performance metrics & experiment tracking
4. Model retraining triggers
5. Serving the interactive Web Dashboard
"""

import os
import json
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.utils import load_config, setup_logger, get_project_root
from src.database.db_manager import ReviewDatabase
from src.models.predict import SentimentPredictor
from src.monitoring.drift_detector import DriftDetector

logger = setup_logger("FastAPI")

config = load_config()
root = get_project_root()
db_path = config.get("database", {}).get("db_path", "data/reviews_sentiment.db")

app = FastAPI(
    title=config.get("api", {}).get("title", "Product Review Sentiment API"),
    version=config.get("api", {}).get("version", "1.0.0"),
    description="MLOps API for sentiment analysis with SQLite database tracking and monitoring."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = ReviewDatabase(db_path)

try:
    predictor = SentimentPredictor()
except Exception as e:
    logger.warning(f"SentimentPredictor initialization delayed: {e}")
    predictor = None

drift_detector = DriftDetector(db_path)

models_dir = root / config.get("model", {}).get("save_dir", "models")
if models_dir.exists():
    app.mount("/static/models", StaticFiles(directory=str(models_dir)), name="models_static")

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, example="This moisturizer is fantastic! Absorbs quickly and keeps skin hydrated.")
    log_to_db: Optional[bool] = Field(default=True, description="Whether to log this prediction to SQLite")

class PredictResponse(BaseModel):
    text: str
    cleaned_text: str
    sentiment: str
    label: int
    confidence: float
    prob_positive: float
    prob_negative: float
    latency_ms: float

class BatchPredictRequest(BaseModel):
    reviews: List[str] = Field(..., min_items=1, example=[
        "Great quality and fast delivery!",
        "Defective unit, stop working after 2 days."
    ])

class BatchPredictResponse(BaseModel):
    total_reviews: int
    predictions: List[PredictResponse]

@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint validating API and model status."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "database": "connected",
        "database_path": str(db_path)
    }

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_sentiment(req: PredictRequest):
    """Predict sentiment of a single product review."""
    global predictor
    if predictor is None:
        try:
            predictor = SentimentPredictor()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not trained or available yet: {str(e)}")

    try:
        result = predictor.predict(req.text, log_to_db=req.log_to_db)
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch_sentiment(req: BatchPredictRequest):
    """Predict sentiment for a batch of reviews."""
    global predictor
    if predictor is None:
        try:
            predictor = SentimentPredictor()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")

    results = predictor.predict_batch(req.reviews, log_to_db=True)
    return {
        "total_reviews": len(results),
        "predictions": results
    }

@app.get("/metrics", tags=["MLOps"])
def get_metrics():
    """Get latest model performance metrics and experiment history."""
    metrics_path = models_dir / config.get("model", {}).get("metrics_filename", "metrics.json")
    
    metrics_data = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    keywords_path = models_dir / "top_keywords.json"
    keywords_data = {}
    if keywords_path.exists():
        with open(keywords_path, "r", encoding="utf-8") as f:
            keywords_data = json.load(f)

    recent_experiments = db.get_recent_experiments(limit=5)

    return {
        "latest_metrics": metrics_data,
        "top_keywords": keywords_data,
        "recent_experiments": recent_experiments
    }

@app.get("/db/stats", tags=["Database"])
def get_database_stats():
    """Get comprehensive statistics of records stored in the SQLite database."""
    stats = db.get_db_stats()
    return stats

@app.get("/predictions/recent", tags=["Monitoring"])
def get_recent_predictions(limit: int = 25):
    """Fetch recent live predictions recorded in SQLite prediction_logs."""
    logs = db.get_recent_predictions(limit=limit)
    return {"count": len(logs), "logs": logs}

@app.get("/monitoring/drift", tags=["Monitoring"])
def get_drift_analysis(sample_size: int = 100):
    """Analyze recent predictions to detect data drift or sentiment anomalies."""
    report = drift_detector.analyze_drift(sample_size=sample_size)
    return report

@app.post("/pipeline/train", tags=["MLOps"])
def trigger_training(background_tasks: BackgroundTasks, algo: str = "logistic_regression"):
    """Trigger model retraining directly from the SQLite database."""
    from src.models.train import train_model

    def train_task():
        global predictor
        train_model(algorithm_override=algo)
        predictor = SentimentPredictor()

    background_tasks.add_task(train_task)
    return {"status": "TRAINING_TRIGGERED", "algorithm": algo, "message": "Training running in background"}

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def serve_dashboard():
    """Serves the interactive MLOps web dashboard."""
    dashboard_file = root / "dashboard" / "index.html"
    if dashboard_file.exists():
        with open(dashboard_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Dashboard file not found. Please verify dashboard/index.html</h2>")
