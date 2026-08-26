"""Feature engineering."""
import pandas as pd

CATEGORICAL = ["PU_DO"]
NUMERICAL = ["trip_distance"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the PU_DO categorical feature and cast types."""
    df = df.copy()
    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)
    df[CATEGORICAL] = df[CATEGORICAL].astype(str)
    return df


def to_dicts(df: pd.DataFrame) -> list[dict]:
    """Convert the feature columns into a list of dicts for DictVectorizer."""
    return df[CATEGORICAL + NUMERICAL].to_dict(orient="records")