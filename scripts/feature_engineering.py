"""Feature engineering for flight delay prediction."""
from __future__ import annotations
import logging, os
from typing import Optional
import pandas as pd
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
PROJECT_ID    = os.getenv("GCP_PROJECT_ID", "your-gcp-project")
DATASET       = os.getenv("BIGQUERY_DATASET", "flights")
SOURCE_TABLE  = f"{PROJECT_ID}.{DATASET}.cleaned_flights"
FEATURE_TABLE = f"{PROJECT_ID}.{DATASET}.flight_features"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features from cleaned flight records."""
    df = df.copy()
    df["fl_date"]      = pd.to_datetime(df["fl_date"])
    df["day_of_week"]  = df["fl_date"].dt.dayofweek
    df["month"]        = df["fl_date"].dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["quarter"]      = df["fl_date"].dt.quarter
    df["dep_hour"]     = (df["crs_dep_time"] // 100).clip(0, 23)
    df["is_peak_hour"] = (df["dep_hour"].between(7,9) | df["dep_hour"].between(16,19)).astype(int)
    df["route"]        = df["carrier"] + "_" + df["origin"] + "_" + df["dest"]
    df["route_avg_delay"] = df.groupby("route")["arr_delay"].transform("mean").fillna(0)
    df["distance_bucket"] = pd.cut(df["distance"],bins=[0,500,1500,float("inf")],labels=["short","medium","long"])
    df["is_delayed"]   = (df["arr_delay"] > 15).astype(int)
    logger.info("Engineered %d features for %d records", df.shape[1], len(df))
    return df

def load_and_engineer(bq_client: Optional[bigquery.Client] = None) -> pd.DataFrame:
    """Load cleaned flights from BigQuery, engineer features, write back."""
    client = bq_client or bigquery.Client(project=PROJECT_ID)
    query  = f"SELECT fl_date,carrier,origin,dest,dep_delay,arr_delay,distance,crs_dep_time,crs_arr_time FROM `{SOURCE_TABLE}` WHERE arr_delay IS NOT NULL"
    df     = client.query(query).to_dataframe()
    df     = engineer_features(df)
    client.load_table_from_dataframe(df, FEATURE_TABLE,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")).result()
    logger.info("Features written to %s", FEATURE_TABLE)
    return df

if __name__ == "__main__":
    load_and_engineer()
