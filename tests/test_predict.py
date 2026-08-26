"""Tests for the DurationPredictor class."""
import pytest
from mlops_practitioner.predict import DurationPredictor


def test_predict_one_returns_float(trained_model, sample_features):
    result = trained_model.predict_one(sample_features)
    assert isinstance(result, float)


def test_predict_one_is_positive(trained_model, sample_features):
    result = trained_model.predict_one(sample_features)
    assert result > 0


def test_predict_one_is_deterministic(trained_model, sample_features):
    result1 = trained_model.predict_one(sample_features)
    result2 = trained_model.predict_one(sample_features)
    assert result1 == pytest.approx(result2, rel=1e-6)


def test_predict_batch_returns_list(trained_model, sample_features):
    result = trained_model.predict_batch([sample_features, sample_features])
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(x, float) for x in result)


def test_predict_without_loading_raises_error():
    predictor = DurationPredictor()
    with pytest.raises(RuntimeError):
        predictor.predict_one({"PU_DO": "1_1", "trip_distance": 1.0})


@pytest.mark.parametrize("distance", [0.1, 5.0, 50.0])
def test_predict_various_distances(trained_model, distance):
    result = trained_model.predict_one({"PU_DO": "43_151", "trip_distance": distance})
    assert result > 0