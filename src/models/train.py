"""
Model Training and Experiment Tracking Pipeline.

Directly queries training data from the SQLite database, extracts TF-IDF features,
trains a high-precision sentiment classification model, evaluates against the
database test split, persists serialized model artifacts, and logs experiment
results to the SQLite database.
"""

import os
import json
import joblib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.utils import load_config, setup_logger, get_project_root
from src.database.db_manager import ReviewDatabase
from src.features.build_features import build_vectorizer, save_vectorizer

logger = setup_logger("ModelTraining")

def get_classifier(algorithm: str, config: dict):
    """Factory to instantiate the specified classification model."""
    model_cfg = config.get("model", {})
    random_state = model_cfg.get("random_state", 42)
    c_param = model_cfg.get("C", 2.0)
    max_iter = model_cfg.get("max_iter", 1000)

    if algorithm == "logistic_regression":
        return LogisticRegression(
            C=c_param,
            max_iter=max_iter,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs"
        )
    elif algorithm == "linear_svc":
        base_svc = LinearSVC(
            C=c_param,
            max_iter=max_iter,
            class_weight="balanced",
            random_state=random_state
        )
        return CalibratedClassifierCV(estimator=base_svc, cv=3)
    elif algorithm == "naive_bayes":
        return MultinomialNB(alpha=0.5)
    else:
        logger.warning(f"Unknown algorithm '{algorithm}', falling back to LogisticRegression")
        return LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=random_state)

def save_confusion_matrix_plot(cm: np.ndarray, labels: list, output_path: Path):
    """Render and save confusion matrix figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title="Sentiment Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info(f"Confusion matrix plot saved to {output_path}")

def train_model(algorithm_override: str = None) -> Dict[str, Any]:
    """
    Train model using database data, evaluate performance, and track in SQLite.
    
    Returns:
        Dictionary of computed performance metrics and artifact paths.
    """
    config = load_config()
    root = get_project_root()
    db_path = config.get("database", {}).get("db_path", "data/reviews_sentiment.db")
    db = ReviewDatabase(db_path)

    logger.info("Extracting training and testing splits directly from SQLite database...")
    train_df = db.fetch_processed_reviews(split="train")
    test_df = db.fetch_processed_reviews(split="test")

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Processed reviews table is empty or missing train/test splits! "
            "Please run: python -m src.data.preprocess first."
        )

    logger.info(f"Loaded from database: {len(train_df)} training samples, {len(test_df)} testing samples.")

    X_train_text = train_df["cleaned_text"].fillna("")
    y_train = train_df["sentiment"].values

    X_test_text = test_df["cleaned_text"].fillna("")
    y_test = test_df["sentiment"].values

    logger.info("Fitting TF-IDF Vectorizer on database training text...")
    vectorizer = build_vectorizer(config)
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)
    logger.info(f"Feature matrix shape: Train={X_train_vec.shape}, Test={X_test_vec.shape}")

    algo = algorithm_override or config.get("model", {}).get("algorithm", "logistic_regression")
    logger.info(f"Training sentiment classification model ({algo})...")
    clf = get_classifier(algo, config)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    y_prob = clf.predict_proba(X_test_vec)[:, 1] if hasattr(clf, "predict_proba") else None

    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro"))
    prec_pos = float(precision_score(y_test, y_pred, pos_label=1))
    prec_neg = float(precision_score(y_test, y_pred, pos_label=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro"))
    f1_m = float(f1_score(y_test, y_pred, average="macro"))
    roc_auc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else 0.0
    cm = confusion_matrix(y_test, y_pred)

    logger.info("=" * 60)
    logger.info("MODEL EVALUATION RESULTS (Database Test Set)")
    logger.info("=" * 60)
    logger.info(f"Accuracy:           {acc * 100:.2f}%")
    logger.info(f"Macro Precision:    {prec_macro * 100:.2f}%")
    logger.info(f"Positive Precision: {prec_pos * 100:.2f}%")
    logger.info(f"Negative Precision: {prec_neg * 100:.2f}%")
    logger.info(f"Macro Recall:       {rec_macro * 100:.2f}%")
    logger.info(f"Macro F1-Score:     {f1_m * 100:.2f}%")
    if roc_auc:
        logger.info(f"ROC-AUC:            {roc_auc:.4f}")
    logger.info("-" * 60)
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
    logger.info("=" * 60)

    model_dir = root / config.get("model", {}).get("save_dir", "models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / config.get("model", {}).get("model_filename", "sentiment_model.joblib")
    vec_path = model_dir / config.get("model", {}).get("vectorizer_filename", "vectorizer.joblib")
    metrics_path = model_dir / config.get("model", {}).get("metrics_filename", "metrics.json")
    cm_path = model_dir / "confusion_matrix.png"

    joblib.dump(clf, model_path)
    save_vectorizer(vectorizer, str(vec_path))
    save_confusion_matrix_plot(cm, ["Negative", "Positive"], cm_path)

    metrics_payload = {
        "model_name": f"sentiment_{algo}",
        "algorithm": algo,
        "dataset_size": int(len(train_df) + len(test_df)),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_positive": prec_pos,
        "precision_negative": prec_neg,
        "recall_macro": rec_macro,
        "f1_macro": f1_m,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "model_path": str(model_path.relative_to(root)),
        "vectorizer_path": str(vec_path.relative_to(root)),
        "hyperparameters": config.get("model", {})
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info(f"Metrics saved to {metrics_path}")

    exp_id = db.log_experiment(metrics_payload)
    logger.info(f"Experiment successfully recorded in SQLite database table 'experiments' (ID: {exp_id})")

    metrics_payload["experiment_id"] = exp_id
    return metrics_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment model using SQLite database")
    parser.add_argument("--algo", type=str, default=None, choices=["logistic_regression", "linear_svc", "naive_bayes"],
                        help="Model algorithm to train")
    args = parser.parse_args()
    train_model(algorithm_override=args.algo)
