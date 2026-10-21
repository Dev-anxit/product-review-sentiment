"""
Feature Engineering for Sentiment Analysis.

Extracts TF-IDF features with n-grams (1, 2) and sentiment-aware stop word handling
to ensure high accuracy and precision in detecting positive and negative nuances.
"""

import joblib
from pathlib import Path
from typing import Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from src.utils import load_config, setup_logger

logger = setup_logger("FeatureEngineering")

SENTIMENT_KEEP_WORDS = {
    "not", "no", "never", "none", "neither", "nor", "nothing", 
    "nowhere", "hardly", "scarcely", "barely", "doesn", "isn", "wasn",
    "won", "didn", "aren", "couldn", "shouldn", "wouldn"
}

CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS - SENTIMENT_KEEP_WORDS)

def build_vectorizer(config: Optional[dict] = None) -> TfidfVectorizer:
    """
    Construct a TF-IDF vectorizer optimized for review sentiment classification.
    
    Args:
        config: Optional configuration dictionary.
        
    Returns:
        Configured TfidfVectorizer instance.
    """
    if config is None:
        config = load_config()

    feat_cfg = config.get("features", {})
    ngram_range = tuple(feat_cfg.get("ngram_range", [1, 2]))
    max_features = feat_cfg.get("max_features", 20000)
    min_df = feat_cfg.get("min_df", 3)
    sublinear_tf = feat_cfg.get("sublinear_tf", True)
    use_idf = feat_cfg.get("use_idf", True)

    logger.info(
        f"Initializing TfidfVectorizer: ngram_range={ngram_range}, "
        f"max_features={max_features}, sublinear_tf={sublinear_tf}"
    )

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        use_idf=use_idf,
        stop_words=CUSTOM_STOP_WORDS,
        token_pattern=r"(?u)\b\w+\b"
    )

    return vectorizer

def save_vectorizer(vectorizer: TfidfVectorizer, file_path: str) -> None:
    """Persist fitted vectorizer to disk."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)
    logger.info(f"Vectorizer artifact saved to: {path}")

def load_vectorizer(file_path: str) -> TfidfVectorizer:
    """Load persisted vectorizer from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Vectorizer file not found at: {path}")
    vectorizer = joblib.load(path)
    logger.info(f"Loaded vectorizer from: {path}")
    return vectorizer
