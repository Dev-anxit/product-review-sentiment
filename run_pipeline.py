"""
MLOps Master Pipeline Runner - Product Review Sentiment Analysis.

A beginner-friendly, end-to-end command-center that orchestrates:
1. Data Ingestion into SQLite
2. Text Preprocessing & Labeling in SQLite
3. Database-driven Model Training & Experiment Tracking
4. Evaluation & Explainability
5. Inference Engine Verification
6. API & Dashboard Serving
"""

import sys
import argparse
from pathlib import Path
from src.utils import setup_logger, load_config
from src.data.ingest import ingest_data
from src.data.preprocess import preprocess_and_store
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.models.predict import SentimentPredictor

logger = setup_logger("PipelineRunner")

def print_banner(step_num: int, title: str):
    print("\n" + "=" * 70)
    print(f" STEP {step_num}: {title.upper()}")
    print("=" * 70)

def run_full_pipeline(limit: int = 35000, algo: str = "logistic_regression"):
    """Execute complete end-to-end MLOps pipeline."""
    print("\n" + "#" * 70)
    print(" STARTING END-TO-END PRODUCT REVIEW SENTIMENT MLOPS PIPELINE")
    print("#" * 70)

    print_banner(1, "Ingesting Raw Reviews into SQLite Database")
    count_ingested = ingest_data(limit=limit)
    print(f"-> Ingested {count_ingested} records into database.")

    print_banner(2, "Preprocessing Reviews and Populating Feature Store")
    count_processed = preprocess_and_store()
    print(f"-> Preprocessed {count_processed} records ready for training.")

    print_banner(3, "Training Sentiment Model from Database")
    train_results = train_model(algorithm_override=algo)
    print(f"-> Model Trained: {train_results['model_name']}")
    print(f"-> Test Accuracy:  {train_results['accuracy'] * 100:.2f}%")
    print(f"-> Test Precision: {train_results['precision_macro'] * 100:.2f}%")

    print_banner(4, "Evaluating Model & Extracting Sentiment Explainability")
    eval_results = evaluate_model()
    print("-> Model evaluation and keywords generated.")

    print_banner(5, "Verifying Real-time Inference Engine")
    predictor = SentimentPredictor()
    sample_positive = "I absolutely love this serum! It transformed my skin completely and smells fantastic."
    sample_negative = "Horrible product! Do not waste your money, broke after one use and caused irritation."

    res_pos = predictor.predict(sample_positive)
    res_neg = predictor.predict(sample_negative)

    print(f"\n[Test Review 1]: \"{sample_positive}\"")
    print(f" -> Predicted: {res_pos['sentiment']} (Confidence: {res_pos['confidence']*100:.1f}%, Latency: {res_pos['latency_ms']}ms)")

    print(f"\n[Test Review 2]: \"{sample_negative}\"")
    print(f" -> Predicted: {res_neg['sentiment']} (Confidence: {res_neg['confidence']*100:.1f}%, Latency: {res_neg['latency_ms']}ms)")

    print("\n" + "#" * 70)
    print(" ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print(f" SQLite Database: data/reviews_sentiment.db")
    print(f" Model Artifacts: models/sentiment_model.joblib, models/vectorizer.joblib")
    print(f" Test Accuracy:   {train_results['accuracy'] * 100:.2f}%")
    print(f" Test Precision:  {train_results['precision_macro'] * 100:.2f}%")
    print("#" * 70 + "\n")

def serve_app():
    """Start the FastAPI backend and interactive web dashboard."""
    import uvicorn
    config = load_config()
    host = config.get("api", {}).get("host", "127.0.0.1")
    port = config.get("api", {}).get("port", 8000)
    print(f"\n[Serving API & Dashboard at http://{host}:{port}]")
    uvicorn.run("api.main:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Product Review Sentiment MLOps Pipeline Runner")
    parser.add_argument("--all", action="store_true", help="Run entire pipeline from scratch")
    parser.add_argument("--ingest", action="store_true", help="Run data ingestion to SQLite")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing and label generation")
    parser.add_argument("--train", action="store_true", help="Train sentiment model using SQLite data")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model performance")
    parser.add_argument("--predict", type=str, default=None, help="Predict sentiment of given text")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI server and UI dashboard")
    parser.add_argument("--limit", type=int, default=35000, help="Row limit for ingestion")
    parser.add_argument("--algo", type=str, default="logistic_regression", help="Model algorithm")

    args = parser.parse_args()

    if not any([args.all, args.ingest, args.preprocess, args.train, args.evaluate, args.predict, args.serve]):
        args.all = True

    if args.all:
        run_full_pipeline(limit=args.limit, algo=args.algo)
    else:
        if args.ingest:
            ingest_data(limit=args.limit)
        if args.preprocess:
            preprocess_and_store()
        if args.train:
            train_model(algorithm_override=args.algo)
        if args.evaluate:
            evaluate_model()
        if args.predict:
            p = SentimentPredictor()
            r = p.predict(args.predict)
            print(f"\nSentiment: {r['sentiment']} | Confidence: {r['confidence']*100:.2f}% | Latency: {r['latency_ms']}ms\n")
        if args.serve:
            serve_app()
