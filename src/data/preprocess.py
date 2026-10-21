"""
Data Preprocessing and Feature Preparation for Sentiment Analysis.

Reads raw reviews from SQLite, cleans and normalizes text (handling contractions,
HTML, URLs, and negations), assigns sentiment labels based on ratings,
performs stratified train/test split, and stores the processed dataset back into SQLite.
"""

import re
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import load_config, setup_logger
from src.database.db_manager import ReviewDatabase

logger = setup_logger("DataPreprocessing")

CONTRACTIONS = {
    r"\bwon't\b": "will not",
    r"\bcan't\b": "cannot",
    r"\bshan't\b": "shall not",
    r"n't\b": " not",
    r"'re\b": " are",
    r"'s\b": " is",
    r"'d\b": " would",
    r"'ll\b": " will",
    r"'ve\b": " have",
    r"'m\b": " am"
}

def clean_text(text: str) -> str:
    """
    Clean raw review text for NLP sentiment modeling.
    
    1. Lowercase text
    2. Expand contractions (e.g., didn't -> did not)
    3. Remove HTML tags and entities
    4. Remove web URLs
    5. Remove non-alphabetical noise while keeping words and spaces
    6. Collapse redundant whitespaces
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(contraction, expansion, text)

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

def preprocess_and_store() -> int:
    """
    Load raw reviews from SQLite database, clean text, label sentiment,
    split into train/test sets, and save into SQLite processed_reviews table.
    
    Returns:
        Total number of processed reviews stored.
    """
    config = load_config()
    db_path = config.get("database", {}).get("db_path", "data/reviews_sentiment.db")
    db = ReviewDatabase(db_path)

    logger.info("Fetching raw reviews from SQLite database...")
    df_raw = db.fetch_raw_reviews()

    if df_raw.empty:
        raise ValueError("No raw reviews found in database! Please run ingestion first: python -m src.data.ingest")

    logger.info(f"Loaded {len(df_raw)} raw reviews from database.")

    df_raw["review_title"] = df_raw["review_title"].fillna("")
    df_raw["review_text"] = df_raw["review_text"].fillna("")
    df_raw["full_text"] = df_raw["review_title"] + " " + df_raw["review_text"]

    sentiment_mode = config.get("data", {}).get("sentiment_mode", "binary")

    if sentiment_mode == "binary":
        df = df_raw[df_raw["rating"] != 3.0].copy()
        df["sentiment"] = (df["rating"] >= 4.0).astype(int)
        df["sentiment_label"] = df["sentiment"].map({1: "Positive", 0: "Negative"})
    else:
        df = df_raw.copy()
        conditions = [df["rating"] >= 4.0, df["rating"] == 3.0, df["rating"] <= 2.0]
        choices = [1, 2, 0]
        labels = ["Positive", "Neutral", "Negative"]
        import numpy as np
        df["sentiment"] = np.select(conditions, choices, default=1)
        df["sentiment_label"] = np.select(conditions, labels, default="Positive")

    logger.info("Cleaning review text (handling negations, HTML, whitespaces)...")
    df["cleaned_text"] = df["full_text"].apply(clean_text)

    df["text_length"] = df["cleaned_text"].str.len()
    df = df[df["text_length"] > 3].copy()

    test_size = config.get("data", {}).get("test_size", 0.20)
    random_state = config.get("data", {}).get("random_state", 42)

    logger.info(f"Splitting dataset: {1 - test_size:.0%} Train, {test_size:.0%} Test (Stratified)...")
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df["sentiment"]
    )

    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    processed_df = df[[
        "review_id",
        "cleaned_text",
        "rating",
        "sentiment",
        "sentiment_label",
        "split",
        "text_length"
    ]].copy()

    inserted = db.insert_processed_reviews(processed_df, clear_existing=True)

    train_count = (processed_df["split"] == "train").sum()
    test_count = (processed_df["split"] == "test").sum()
    pos_count = (processed_df["sentiment"] == 1).sum()
    neg_count = (processed_df["sentiment"] == 0).sum()

    logger.info(f"Preprocessing completed! Stored {inserted} records in SQLite table 'processed_reviews'.")
    logger.info(f"Summary: Train={train_count}, Test={test_count} | Positive={pos_count}, Negative={neg_count}")

    return inserted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess reviews and store in database")
    parser.parse_args()
    preprocess_and_store()
