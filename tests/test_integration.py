import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from mlops_practitioner.api.main import app, get_predictor
from mlops_practitioner.predict import DurationPredictor


@pytest.fixture(scope="module")
def client():
    mock = MagicMock(spec=DurationPredictor)
    mock.predict_one.return_value = 15.5

    app.dependency_overrides[get_predictor] = lambda: mock

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


def test_predict_success(client):
    response = client.post(
        "/predict",
        json={"PU_DO": "43_151", "trip_distance": 5.0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_duration_minutes"] == 15.5


def test_predict_validation_error_negative_distance(client):
    # trip_distance must be > 0 (Field(gt=0)) -> pydantic rejects it
    # before the request even reaches the mocked model.
    response = client.post(
        "/predict",
        json={"PU_DO": "43_151", "trip_distance": -1.0},
    )
    assert response.status_code == 422


def test_predict_validation_error_distance_too_large(client):
    # trip_distance must be < 200 (Field(lt=200))
    response = client.post(
        "/predict",
        json={"PU_DO": "43_151", "trip_distance": 500.0},
    )
    assert response.status_code == 422


def test_predict_missing_field(client):
    # PU_DO is required (Field(...))
    response = client.post(
        "/predict",
        json={"trip_distance": 5.0},
    )
    assert response.status_code == 422


def test_predict_batch_success(client):
    response = client.post(
        "/predict/batch",
        json={
            "trips": [
                {"PU_DO": "43_151", "trip_distance": 5.0},
                {"PU_DO": "10_20", "trip_distance": 12.5},
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"