"""
In-depth Model Evaluation and Explainability.

Extracts feature importance weights (top positive and negative keywords),
computes comprehensive classification metrics against the database test set,
and provides clear explainability reports for beginners.
"""

import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from src.utils import load_config, setup_logger, get_project_root
from src.database.db_manager import ReviewDatabase

logger = setup_logger("ModelEvaluation")

def get_top_keywords(model, vectorizer, top_n: int = 15) -> Dict[str, List[Dict[str, Any]]]:
    """Extract top positive and negative predictive words based on model coefficients."""
    if not hasattr(model, "coef_"):
        if hasattr(model, "calibrated_classifiers_"):
            base = model.calibrated_classifiers_[0].estimator
            if hasattr(base, "coef_"):
                coefficients = base.coef_[0]
            else:
                return {"positive_words": [], "negative_words": []}
        else:
            return {"positive_words": [], "negative_words": []}
    else:
        coefficients = model.coef_[0]

    feature_names = np.array(vectorizer.get_feature_names_out())
    
    top_pos_idx = np.argsort(coefficients)[-top_n:][::-1]
    top_pos = [
        {"word": str(feature_names[i]), "weight": round(float(coefficients[i]), 4)}
        for i in top_pos_idx
    ]

    top_neg_idx = np.argsort(coefficients)[:top_n]
    top_neg = [
        {"word": str(feature_names[i]), "weight": round(float(coefficients[i]), 4)}
        for i in top_neg_idx
    ]

    return {"positive_words": top_pos, "negative_words": top_neg}

def evaluate_model() -> Dict[str, Any]:
    """
    Evaluate the saved sentiment model on the test split from the SQLite database.
    """
    config = load_config()
    root = get_project_root()
    db_path = config.get("database", {}).get("db_path", "data/reviews_sentiment.db")
    db = ReviewDatabase(db_path)

    test_df = db.fetch_processed_reviews(split="test")
    if test_df.empty:
        raise ValueError("No test data found in SQLite database! Run preprocessing first.")

    model_dir = root / config.get("model", {}).get("save_dir", "models")
    model_path = model_dir / config.get("model", {}).get("model_filename", "sentiment_model.joblib")
    vec_path = model_dir / config.get("model", {}).get("vectorizer_filename", "vectorizer.joblib")

    if not model_path.exists() or not vec_path.exists():
        raise FileNotFoundError("Model or vectorizer artifact missing. Please train model first.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)

    X_test_vec = vectorizer.transform(test_df["cleaned_text"].fillna(""))
    y_test = test_df["sentiment"].values
    y_pred = model.predict(X_test_vec)

    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro"))
    prec_pos = float(precision_score(y_test, y_pred, pos_label=1))
    prec_neg = float(precision_score(y_test, y_pred, pos_label=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro"))
    f1_m = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])

    keywords = get_top_keywords(model, vectorizer, top_n=12)

    top_words_path = model_dir / "top_keywords.json"
    with open(top_words_path, "w", encoding="utf-8") as f:
        json.dump(keywords, f, indent=2)

    logger.info("=" * 65)
    logger.info("MODEL EVALUATION & EXPLAINABILITY REPORT")
    logger.info("=" * 65)
    logger.info(f"Database Test Samples: {len(test_df)}")
    logger.info(f"Accuracy:            {acc * 100:.2f}%")
    logger.info(f"Macro Precision:     {prec_macro * 100:.2f}%")
    logger.info(f"Positive Precision:  {prec_pos * 100:.2f}%")
    logger.info(f"Negative Precision:  {prec_neg * 100:.2f}%")
    logger.info(f"Macro F1-Score:      {f1_m * 100:.2f}%")
    logger.info("-" * 65)
    logger.info("\nTOP POSITIVE PREDICTIVE WORDS:")
    for item in keywords["positive_words"]:
        logger.info(f"  + {item['word']:<20} (weight: {item['weight']})")
    logger.info("\nTOP NEGATIVE PREDICTIVE WORDS:")
    for item in keywords["negative_words"]:
        logger.info(f"  - {item['word']:<20} (weight: {item['weight']})")
    logger.info("=" * 65)

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_positive": prec_pos,
        "precision_negative": prec_neg,
        "recall_macro": rec_macro,
        "f1_macro": f1_m,
        "keywords": keywords,
        "classification_report": report
    }

if __name__ == "__main__":
    evaluate_model()
