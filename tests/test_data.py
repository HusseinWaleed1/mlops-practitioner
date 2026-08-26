"""Tests for data loading and preprocessing."""
import pandas as pd
from mlops_practitioner.data import compute_duration, train_val_split


def test_compute_duration_calculates_minutes():
    df = pd.DataFrame({
        "lpep_pickup_datetime": pd.to_datetime(["2023-01-01 10:00:00"]),
        "lpep_dropoff_datetime": pd.to_datetime(["2023-01-01 10:15:00"]),
    })
    result = compute_duration(df)
    assert result["duration"].iloc[0] == 15.0


def test_compute_duration_drops_zero_duration_trips():
    df = pd.DataFrame({
        "lpep_pickup_datetime": pd.to_datetime(["2023-01-01 10:00:00", "2023-01-01 11:00:00"]),
        "lpep_dropoff_datetime": pd.to_datetime(["2023-01-01 10:00:00", "2023-01-01 11:15:00"]),
    })
    result = compute_duration(df)
    assert len(result) == 1
    assert result["duration"].iloc[0] == 15.0


def test_compute_duration_drops_trips_over_max():
    df = pd.DataFrame({
        "lpep_pickup_datetime": pd.to_datetime(["2023-01-01 10:00:00", "2023-01-01 11:00:00"]),
        "lpep_dropoff_datetime": pd.to_datetime(["2023-01-01 10:10:00", "2023-01-01 13:00:00"]),
    })
    result = compute_duration(df)
    assert len(result) == 1
    assert result["duration"].iloc[0] == 10.0


def test_train_val_split_default_ratio():
    df = pd.DataFrame({"x": range(100)})
    train, val = train_val_split(df)
    assert len(train) == 80
    assert len(val) == 20


def test_train_val_split_custom_ratio():
    df = pd.DataFrame({"x": range(100)})
    train, val = train_val_split(df, ratio=0.7)
    assert len(train) == 70
    assert len(val) == 30


def test_train_val_split_preserves_order():
    df = pd.DataFrame({"x": range(10)})
    train, val = train_val_split(df, ratio=0.8)
    assert train["x"].tolist() == list(range(8))
    assert val["x"].tolist() == [8, 9]