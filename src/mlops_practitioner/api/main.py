"""FastAPI application exposing the duration prediction model."""
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mlops_practitioner.config import settings
from mlops_practitioner.logging_conf import setup_logging, get_logger, correlation_id_var
from mlops_practitioner.predict import DurationPredictor
from mlops_practitioner.features import CATEGORICAL, NUMERICAL
from mlops_practitioner.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    MetadataResponse,
)

logger = get_logger(__name__)

predictor: DurationPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    global predictor
    logger.info("Starting up: loading model")
    predictor = DurationPredictor().load()
    yield
    logger.info("Shutting down")


app = FastAPI(title="Trip Duration Predictor", lifespan=lifespan)


@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    correlation_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token = correlation_id_var.set(correlation_id)
    try:
        response = await call_next(request)
    finally:
        correlation_id_var.reset(token)
    response.headers["X-Request-ID"] = correlation_id
    return response


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthResponse)
async def health():
    if predictor is None or predictor.model is None:
        return JSONResponse(status_code=503, content={"status": "model not loaded"})
    return HealthResponse(status="ok")


@app.get("/metadata", response_model=MetadataResponse)
async def metadata():
    return MetadataResponse(
        model_version=settings.model_path,
        features=CATEGORICAL + NUMERICAL,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    logger.info(f"Predicting for PU_DO={request.PU_DO} distance={request.trip_distance}")
    pred = predictor.predict_one(request.model_dump())
    return PredictionResponse(predicted_duration_minutes=pred)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    logger.info(f"Batch predicting {len(request.trips)} trips")
    dicts = [t.model_dump() for t in request.trips]
    preds = predictor.predict_batch(dicts)
    return BatchPredictionResponse(predictions=preds)