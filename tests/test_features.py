"""Tests for feature engineering edge cases."""
import pandas as pd
from mlops_practitioner.features import add_features, to_dicts, CATEGORICAL, NUMERICAL


def test_add_features_creates_pu_do_column():
    df = pd.DataFrame({
        "PULocationID": [43, 100],
        "DOLocationID": [151, 200],
        "trip_distance": [2.5, 1.0],
    })
    result = add_features(df)
    assert "PU_DO" in result.columns
    assert result["PU_DO"].tolist() == ["43_151", "100_200"]


def test_add_features_casts_categorical_to_string():
    df = pd.DataFrame({
        "PULocationID": [43],
        "DOLocationID": [151],
        "trip_distance": [2.5],
    })
    result = add_features(df)
    assert pd.api.types.is_string_dtype(result["PU_DO"])


def test_add_features_with_zero_distance():
    df = pd.DataFrame({
        "PULocationID": [1],
        "DOLocationID": [1],
        "trip_distance": [0.0],
    })
    result = add_features(df)
    assert result["trip_distance"].iloc[0] == 0.0


def test_to_dicts_returns_correct_keys():
    df = pd.DataFrame({
        "PU_DO": ["43_151"],
        "trip_distance": [2.5],
    })
    dicts = to_dicts(df)
    assert len(dicts) == 1
    assert set(dicts[0].keys()) == set(CATEGORICAL + NUMERICAL)


def test_add_features_with_missing_pu_do_pair():
    """A never-before-seen PU_DO pair should still produce a valid string."""
    df = pd.DataFrame({
        "PULocationID": [9999],
        "DOLocationID": [9999],
        "trip_distance": [3.0],
    })
    result = add_features(df)
    assert result["PU_DO"].iloc[0] == "9999_9999"