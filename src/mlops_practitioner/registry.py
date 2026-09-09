"""Model registry management: promote and load models by alias."""
import sys

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from mlops_practitioner.logging_conf import setup_logging, get_logger
from mlops_practitioner.config import settings

logger = get_logger(__name__)

MODEL_NAME = "RandomForestModel"
TRACKING_URI = "http://localhost:5000"


def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient(TRACKING_URI)


def list_versions(client: MlflowClient, model_name: str) -> None:
    """List every registered version of a model, with its current aliases."""
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        logger.warning(f"No versions found for model '{model_name}'")
        return
    for v in versions:
        logger.info(
            f"Version: {v.version}, Aliases: {v.aliases}, "
            f"Run ID: {v.run_id}, Status: {v.status}"
        )


def promote_version(
    client: MlflowClient, model_name: str, version: int, alias: str
) -> None:
    """Assign an alias (e.g. 'champion') to a specific model version."""
    try:
        client.set_registered_model_alias(
            name=model_name, alias=alias, version=str(version)
        )
        logger.info(f"Alias '{alias}' -> version {version} of '{model_name}'")
    except MlflowException as e:
        logger.error(f"Failed to set alias: {e}")
        raise


def load_champion_model(model_name: str, alias: str = "champion"):
    """Load the model currently pointed to by an alias.
    Serving code should ALWAYS load by alias, never by hardcoded version number,
    so that promoting a new version doesn't require a code change.
    """
    model_uri = f"models:/{model_name}@{alias}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model from {model_uri}")
        return model
    except MlflowException as e:
        logger.error(
            f"Could not load model at '{model_uri}'. "
            f"Has an alias '{alias}' been set for '{model_name}'? Error: {e}"
        )
        raise


def main() -> None:
    setup_logging(settings.log_level)
    client = get_client()

    list_versions(client, MODEL_NAME)

    # Promote version 1 to the 'champion' alias.
    # In a real pipeline this would be a deliberate, reviewed step
    # (e.g. after comparing metrics against the current champion),
    # not run unconditionally on every script execution.
    promote_version(client, MODEL_NAME, version=1, alias="champion")

    # Serving code loads by alias, never by hardcoded version number.
    model = load_champion_model(MODEL_NAME, alias="champion")

    logger.info(f"Model ready: {type(model).__name__}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Registry script failed")
        sys.exit(1)