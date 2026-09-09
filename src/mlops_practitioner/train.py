"""Model training entry point."""
import pickle
import time
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow

from mlops_practitioner.config import settings
from mlops_practitioner.data import load_raw_data, compute_duration, train_val_split
from mlops_practitioner.features import add_features, to_dicts
from mlops_practitioner.logging_conf import setup_logging, get_logger

logger = get_logger(__name__)


def timed(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


@timed
def train_model() -> dict:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nyc-taxi-duration")
    df = load_raw_data()
    df = compute_duration(df)
    df = add_features(df)

    df_train, df_val = train_val_split(df)

    dv = DictVectorizer()
    X_train = dv.fit_transform(to_dicts(df_train))
    X_val = dv.transform(to_dicts(df_val))

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    params = {
        "n_estimators": settings.n_estimators,
        "max_depth": settings.max_depth,
        "random_state": settings.random_state,
        "n_jobs": -1
    }

    with mlflow.start_run(run_name="random_forest_baseline"):
        mlflow.log_params(params)

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        logger.info(f"MAE={mae:.3f} RMSE={rmse:.3f}")

        with open(settings.model_path, "wb") as f_out:
            pickle.dump((dv, model), f_out)
        logger.info(f"Model saved to {settings.model_path}")
        mlflow.sklearn.log_model(model,
                                  artifact_path="model",registered_model_name="RandomForestModel")

        

    return {"mae": mae, "rmse": rmse}


def main() -> None:
    setup_logging(settings.log_level)
    metrics = train_model()
    print(f"MAE: {metrics['mae']:.3f} | RMSE: {metrics['rmse']:.3f}")


if __name__ == "__main__":
    main()