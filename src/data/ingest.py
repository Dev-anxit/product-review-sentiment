"""
Data Ingestion Pipeline for Product Review Sentiment.

Loads raw review data from CSV, validates schemas, handles missing values,
and populates the SQLite database table `raw_reviews`.
"""

import sys
import argparse
import pandas as pd
from typing import Optional
from src.utils import load_config, setup_logger, get_project_root
from src.database.db_manager import ReviewDatabase

logger = setup_logger("DataIngestion")

def ingest_data(csv_path: Optional[str] = None, limit: Optional[int] = None) -> int:
    """
    Ingest raw CSV reviews into SQLite database.
    
    Args:
        csv_path: Optional override path to raw CSV file.
        limit: Optional override for number of rows to ingest.
        
    Returns:
        Number of reviews ingested.
    """
    config = load_config()
    root = get_project_root()
    
    raw_csv = csv_path or config.get("data", {}).get("raw_csv_path", "data/raw/reviews.csv")
    csv_file = root / raw_csv if not str(raw_csv).startswith("/") else raw_csv
    
    if not csv_file.exists():
        logger.error(f"CSV file not found at: {csv_file}")
        raise FileNotFoundError(f"Raw CSV file does not exist: {csv_file}")
        
    row_limit = limit if limit is not None else config.get("data", {}).get("ingest_limit", 35000)
    
    logger.info(f"Reading raw CSV data from {csv_file} (limit={row_limit})...")
    
    if row_limit and row_limit > 0:
        df = pd.read_csv(csv_file, nrows=row_limit)
    else:
        df = pd.read_csv(csv_file)
        
    logger.info(f"Loaded {len(df)} records from CSV.")
    
    df["review_title"] = df["review_title"].fillna("")
    df["review_text"] = df["review_text"].fillna("")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3.0)
    df["verified_purchase"] = df["verified_purchase"].astype(int)
    
    db_path = config.get("database", {}).get("db_path", "data/reviews_sentiment.db")
    db = ReviewDatabase(db_path)
    
    inserted_count = db.insert_raw_reviews(df)
    logger.info(f"Successfully ingested {inserted_count} reviews into SQLite database '{db_path}'.")
    return inserted_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest raw reviews into SQLite database")
    parser.add_argument("--csv", type=str, default=None, help="Path to raw CSV file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to ingest (0 for all)")
    args = parser.parse_args()
    
    ingest_limit = None if args.limit == 0 else args.limit
    ingest_data(csv_path=args.csv, limit=ingest_limit)
