"""
Database Manager for Product Review Sentiment MLOps Pipeline.

Uses SQLite to provide a zero-setup, lightweight, production-style database
that stores:
1. Raw reviews (ingested from CSV/streaming source)
2. Processed features & labels for model training
3. Experiment tracking metrics (accuracy, precision, recall, F1, model artifacts)
4. Live inference prediction logs for model monitoring and drift detection
"""

import os
import json
import sqlite3
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
from src.utils import setup_logger, get_project_root

logger = setup_logger("DatabaseManager")

class ReviewDatabase:
    """
    Manages SQLite database connections, tables, and CRUD operations for MLOps.
    """

    def __init__(self, db_path: str = "data/reviews_sentiment.db"):
        root = get_project_root()
        self.db_path = root / db_path if not os.path.isabs(db_path) else Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a new SQLite database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Create database tables if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_reviews (
                review_id TEXT PRIMARY KEY,
                product_id TEXT,
                review_title TEXT,
                review_text TEXT,
                rating REAL,
                timestamp INTEGER,
                helpful_vote INTEGER,
                verified_purchase INTEGER,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id TEXT,
                cleaned_text TEXT,
                rating REAL,
                sentiment INTEGER,
                sentiment_label TEXT,
                split TEXT,
                text_length INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                algorithm TEXT,
                dataset_size INTEGER,
                train_size INTEGER,
                test_size INTEGER,
                accuracy REAL,
                precision_macro REAL,
                precision_positive REAL,
                precision_negative REAL,
                recall_macro REAL,
                f1_macro REAL,
                roc_auc REAL,
                hyperparameters TEXT,
                model_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_text TEXT,
                predicted_sentiment TEXT,
                predicted_class INTEGER,
                confidence REAL,
                prob_positive REAL,
                prob_negative REAL,
                latency_ms REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_split ON processed_reviews(split);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_rating ON raw_reviews(rating);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_time ON prediction_logs(timestamp);")

            conn.commit()
            logger.info(f"Database initialized successfully at: {self.db_path}")

    def insert_raw_reviews(self, df: pd.DataFrame, batch_size: int = 5000) -> int:
        """
        Insert raw reviews dataframe into raw_reviews table.
        Uses INSERT OR REPLACE to avoid duplicate key issues.
        """
        required_cols = ["review_id", "product_id", "review_title", "review_text", 
                         "rating", "timestamp", "helpful_vote", "verified_purchase"]
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        records = df[required_cols].to_dict(orient="records")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
            INSERT OR REPLACE INTO raw_reviews 
            (review_id, product_id, review_title, review_text, rating, timestamp, helpful_vote, verified_purchase)
            VALUES (:review_id, :product_id, :review_title, :review_text, :rating, :timestamp, :helpful_vote, :verified_purchase)
            """
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                cursor.executemany(query, batch)
            conn.commit()

        logger.info(f"Inserted {len(records)} raw reviews into database.")
        return len(records)

    def fetch_raw_reviews(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Fetch raw reviews from database as a pandas DataFrame."""
        with self.get_connection() as conn:
            query = "SELECT * FROM raw_reviews"
            if limit:
                query += f" LIMIT {int(limit)}"
            df = pd.read_sql_query(query, conn)
        return df

    def insert_processed_reviews(self, df: pd.DataFrame, clear_existing: bool = True) -> int:
        """Insert cleaned reviews ready for model training."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if clear_existing:
                cursor.execute("DELETE FROM processed_reviews;")
            
            df.to_sql("processed_reviews", conn, if_exists="append", index=False)
            conn.commit()

        logger.info(f"Inserted {len(df)} processed reviews into database.")
        return len(df)

    def fetch_processed_reviews(self, split: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch processed reviews for model training or evaluation.
        
        Args:
            split: 'train', 'test', or None (fetches all).
        """
        with self.get_connection() as conn:
            query = "SELECT * FROM processed_reviews"
            if split:
                query += f" WHERE split = '{split}'"
            df = pd.read_sql_query(query, conn)
        return df

    def log_experiment(self, experiment_data: Dict[str, Any]) -> int:
        """
        Log model training metrics, parameters, and metadata to SQLite.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
            INSERT INTO experiments (
                model_name, algorithm, dataset_size, train_size, test_size,
                accuracy, precision_macro, precision_positive, precision_negative,
                recall_macro, f1_macro, roc_auc, hyperparameters, model_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(query, (
                experiment_data.get("model_name", "sentiment_model"),
                experiment_data.get("algorithm", "logistic_regression"),
                experiment_data.get("dataset_size", 0),
                experiment_data.get("train_size", 0),
                experiment_data.get("test_size", 0),
                float(experiment_data.get("accuracy", 0.0)),
                float(experiment_data.get("precision_macro", 0.0)),
                float(experiment_data.get("precision_positive", 0.0)),
                float(experiment_data.get("precision_negative", 0.0)),
                float(experiment_data.get("recall_macro", 0.0)),
                float(experiment_data.get("f1_macro", 0.0)),
                float(experiment_data.get("roc_auc", 0.0)),
                json.dumps(experiment_data.get("hyperparameters", {})),
                experiment_data.get("model_path", "")
            ))
            conn.commit()
            exp_id = cursor.lastrowid

        logger.info(f"Logged experiment #{exp_id} with Accuracy: {experiment_data.get('accuracy'):.4f}")
        return exp_id

    def get_recent_experiments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent model training experiments."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments ORDER BY experiment_id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_best_experiment(self) -> Optional[Dict[str, Any]]:
        """Retrieve experiment with highest accuracy and precision."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments ORDER BY accuracy DESC, precision_macro DESC LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else None

    def log_prediction(
        self,
        review_text: str,
        predicted_sentiment: str,
        predicted_class: int,
        confidence: float,
        prob_positive: float,
        prob_negative: float,
        latency_ms: float
    ) -> int:
        """Log a real-time prediction request and outcome for monitoring."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
            INSERT INTO prediction_logs 
            (review_text, predicted_sentiment, predicted_class, confidence, prob_positive, prob_negative, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(query, (
                review_text[:500],
                predicted_sentiment,
                predicted_class,
                float(confidence),
                float(prob_positive),
                float(prob_negative),
                float(latency_ms)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_predictions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent prediction logs."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM prediction_logs ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_db_stats(self) -> Dict[str, Any]:
        """Return comprehensive overview statistics of the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM raw_reviews")
            raw_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM processed_reviews")
            processed_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM processed_reviews WHERE split = 'train'")
            train_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM processed_reviews WHERE split = 'test'")
            test_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM processed_reviews WHERE sentiment = 1")
            positive_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM processed_reviews WHERE sentiment = 0")
            negative_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM experiments")
            experiments_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM prediction_logs")
            predictions_count = cursor.fetchone()[0]

        return {
            "database_path": str(self.db_path),
            "raw_reviews_count": raw_count,
            "processed_reviews_count": processed_count,
            "train_count": train_count,
            "test_count": test_count,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "experiments_count": experiments_count,
            "prediction_logs_count": predictions_count
        }
