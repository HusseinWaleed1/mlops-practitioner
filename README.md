# MLOps Practitioner — Trip Duration Predictor

A production-ready ML service predicting NYC green taxi trip duration, built as part of the MLOps Practitioner course (Module 1: Packaging & Containerization).

## Quick Start (3 commands)

```bash
docker pull husseinwaleed/mlops-practitioner-api:latest
docker run -p 8000:8000 husseinwaleed/mlops-practitioner-api:latest
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"PU_DO": "43_151", "trip_distance": 2.5}'
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/metadata` | GET | Model version and feature list |
| `/predict` | POST | Single trip prediction |
| `/predict/batch` | POST | Batch trip predictions |

Interactive docs available at `http://localhost:8000/docs` once running.

## Local Development

```bash
uv sync --extra dev
uv run python -m mlops_practitioner.train      # train the model
uv run uvicorn mlops_practitioner.api.main:app --reload  # run the API
uv run pytest -v                                # run tests
```

## Project Structure

```
src/mlops_practitioner/   # core package (data, features, train, predict, export, api)
tests/                    # test suite (70%+ coverage)
docker/                   # Dockerfile and docker-compose
notebooks/                # exploratory baseline notebook
reports/                  # module reports and metrics
```

## Model

RandomForestRegressor trained on NYC Green Taxi trip data (Jan 2023), predicting trip duration in minutes from pickup/dropoff location pair and trip distance.

## Check Train Output and MLflow Result

```bash
uv run python -m mlops_practitioner.train
uv run mlflow ui
```

Open this link in your browser: http://127.0.0.1:5000