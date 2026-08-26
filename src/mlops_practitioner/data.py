"""Data loading and train/validation splitting."""
import pandas as pd
from mlops_practitioner.config import settings
from mlops_practitioner.logging_conf import get_logger

logger = get_logger(__name__)


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load the raw parquet trip data."""
    path = path or settings.data_path
    logger.info(f"Loading data from {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows")
    return df


def compute_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Add a duration-in-minutes column and drop unrealistic trips."""
    df = df.copy()
    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    before = len(df)
    df = df[
        (df["duration"] >= settings.min_duration_minutes)
        & (df["duration"] <= settings.max_duration_minutes)
    ]
    logger.info(f"Dropped {before - len(df)} rows with unrealistic duration")
    return df


def train_val_split(df: pd.DataFrame, ratio: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple chronological split (no shuffling, since this is time-ordered trip data)."""
    ratio = ratio if ratio is not None else settings.train_split_ratio
    split_idx = int(len(df) * ratio)
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]
    logger.info(f"Split: {len(df_train)} train / {len(df_val)} val")
    return df_train, df_val