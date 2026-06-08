"""Model training pipeline for flight delay prediction."""
from __future__ import annotations
import json, logging, os, pickle
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
FEATURE_COLS = ["day_of_week","month","is_weekend","quarter","dep_hour",
                "is_peak_hour","route_avg_delay","distance","dep_delay"]
TARGET_COL   = "is_delayed"
MODEL_PATH   = "models/flight_delay_model.pkl"
METRICS_PATH = "models/metrics.json"

def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Select features and return (X, y) arrays."""
    X = df[[c for c in FEATURE_COLS if c in df.columns]].fillna(0).values
    y = df[TARGET_COL].values
    return X, y

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    """Compute classification metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    m = dict(model=name,
             accuracy=round(accuracy_score(y_test, y_pred), 4),
             precision=round(precision_score(y_test, y_pred, zero_division=0), 4),
             recall=round(recall_score(y_test, y_pred, zero_division=0), 4),
             f1=round(f1_score(y_test, y_pred, zero_division=0), 4),
             auc_roc=round(roc_auc_score(y_test, y_prob), 4))
    logger.info("[%s] %s", name, m)
    logger.info(classification_report(y_test, y_pred, target_names=["On-time","Delayed"]))
    return m

def train(df: pd.DataFrame) -> None:
    """Train, evaluate, and save the best flight delay model."""
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info("Train=%d  Test=%d", len(X_train), len(X_test))
    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    }
    results = []
    for name, model in candidates.items():
        logger.info("Training %s...", name)
        model.fit(X_train, y_train)
        results.append((evaluate(model, X_test, y_test, name)["f1"], name, model))
    results.sort(reverse=True)
    best_f1, best_name, best_model = results[0]
    logger.info("Best: %s  F1=%.4f", best_name, best_f1)
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f: pickle.dump(best_model, f)
    logger.info("Model saved to %s", MODEL_PATH)

if __name__ == "__main__":
    from feature_engineering import load_and_engineer
    train(load_and_engineer())
