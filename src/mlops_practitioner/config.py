"""Centralized configuration loaded from environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PRODML_", env_file=".env", extra="ignore")

    # Paths
    model_path: str = "../models/baseline.pkl"
    data_path: str = "../data/green_tripdata_2023-01.parquet"

    # Model training
    n_estimators: int = 100
    max_depth: int = 10
    random_state: int = 42
    train_split_ratio: float = 0.8
    min_duration_minutes: float = 1.0
    max_duration_minutes: float = 60.0

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"


settings = Settings()