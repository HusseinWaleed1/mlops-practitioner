"""Pydantic request/response schemas for the API."""
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    PU_DO: str = Field(..., description="Pickup_Dropoff location pair, e.g. '43_151'")
    trip_distance: float = Field(..., gt=0, lt=200, description="Trip distance in miles")


class PredictionResponse(BaseModel):
    predicted_duration_minutes: float


class BatchPredictionRequest(BaseModel):
    trips: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: list[float]


class HealthResponse(BaseModel):
    status: str


class MetadataResponse(BaseModel):
    model_version: str
    features: list[str]