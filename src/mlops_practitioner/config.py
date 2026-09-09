"""Centralized configuration loaded from environment variables and params.yaml."""
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRODML_",
        env_file=".env",
        extra="ignore",
        yaml_file="params.yaml",
    )

    # Paths
    model_path: str = "models/baseline.pkl"
    data_path: str = "data/green_tripdata_2023-01.parquet"

    # Model training (defaults come from params.yaml, overridable via PRODML_* env vars)
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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        # Priority (highest first): explicit init args > env vars > .env file
        # > params.yaml > secret files. This means params.yaml only fills in
        # values that aren't already set via environment, keeping existing
        # env-based overrides working exactly as before.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


settings = Settings()