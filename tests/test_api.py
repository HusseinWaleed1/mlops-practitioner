"""Tests for the FastAPI endpoints."""


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metadata_returns_features(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    assert "features" in response.json()


def test_predict_happy_path(client, sample_features):
    response = client.post("/predict", json=sample_features)
    assert response.status_code == 200
    assert "predicted_duration_minutes" in response.json()


def test_predict_invalid_distance_returns_422(client):
    response = client.post("/predict", json={"PU_DO": "43_151", "trip_distance": -5})
    assert response.status_code == 422


def test_predict_missing_field_returns_422(client):
    response = client.post("/predict", json={"PU_DO": "43_151"})
    assert response.status_code == 422


def test_predict_batch_happy_path(client, sample_features):
    response = client.post("/predict/batch", json={"trips": [sample_features, sample_features]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2