"""Shared pytest fixtures."""
import pytest
from fastapi.testclient import TestClient
from mlops_practitioner.predict import DurationPredictor


@pytest.fixture(scope="session")
def trained_model() -> DurationPredictor:
    """Load the real trained model once per test session."""
    return DurationPredictor().load()


@pytest.fixture
def sample_features() -> dict:
    return {"PU_DO": "43_151", "trip_distance": 2.5}


@pytest.fixture
def client():
    from mlops_practitioner.api.main import app
    with TestClient(app) as c:
        yield c