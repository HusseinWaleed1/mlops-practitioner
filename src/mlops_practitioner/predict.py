"""Prediction logic: loads the trained model and serves predictions."""
import pickle
import time
from mlops_practitioner.config import settings
from mlops_practitioner.features import CATEGORICAL, NUMERICAL
from mlops_practitioner.logging_conf import get_logger

logger = get_logger(__name__)


class DurationPredictor:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or settings.model_path
        self.dv = None
        self.model = None

    def load(self) -> "DurationPredictor":
        logger.info(f"Loading model from {self.model_path}")
        with open(self.model_path, "rb") as f_in:
            self.dv, self.model = pickle.load(f_in)
        logger.info("Model loaded successfully")
        return self

    def _ensure_loaded(self) -> None:
        if self.model is None or self.dv is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

    def predict_one(self, features: dict) -> float:
        self._ensure_loaded()
        record = {k: features[k] for k in CATEGORICAL + NUMERICAL}
        X = self.dv.transform([record])
        pred = self.model.predict(X)[0]
        return float(pred)

    def predict_batch(self, features_list: list[dict]) -> list[float]:
        self._ensure_loaded()
        records = [{k: f[k] for k in CATEGORICAL + NUMERICAL} for f in features_list]
        X = self.dv.transform(records)
        preds = self.model.predict(X)
        return [float(p) for p in preds]