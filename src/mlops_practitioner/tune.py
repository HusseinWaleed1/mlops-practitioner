"""Hyperparameter search entry point using Optuna, tracked in MLflow."""
import optuna
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from mlops_practitioner.config import settings
from mlops_practitioner.data import load_raw_data, compute_duration, train_val_split
from mlops_practitioner.features import add_features, to_dicts
from mlops_practitioner.logging_conf import setup_logging, get_logger

logger = get_logger(__name__)


def prepare_data():
    """Same preprocessing pipeline used in train.py, reused here."""
    df = load_raw_data()
    df = compute_duration(df)
    df = add_features(df)
    df_train, df_val = train_val_split(df)

    dv = DictVectorizer()
    X_train = dv.fit_transform(to_dicts(df_train))
    X_val = dv.transform(to_dicts(df_val))
    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    return X_train, X_val, y_train, y_val


def make_objective(X_train, X_val, y_train, y_val):
    """Returns an Optuna objective closure with the data baked in."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [50, 100, 200]),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "random_state": settings.random_state,
            "n_jobs": -1,
        }

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)

            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)

            mae = mean_absolute_error(y_val, model.predict(X_val))
            mlflow.log_metric("mae", mae)

        logger.info(f"Trial {trial.number}: params={params} mae={mae:.3f}")
        return mae

    return objective


def run_search(n_trials: int = 20) -> optuna.Study:
    mlflow.set_experiment("nyc-taxi-duration")
    X_train, X_val, y_train, y_val = prepare_data()
    objective = make_objective(X_train, X_val, y_train, y_val)

    # Parent run groups all trials together in the MLflow UI.
    with mlflow.start_run(run_name="optuna_search"):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_mae", study.best_value)

    logger.info(f"Best params: {study.best_params}")
    logger.info(f"Best MAE: {study.best_value:.3f}")
    return study


def main() -> None:
    setup_logging(settings.log_level)
    run_search(n_trials=20)


if __name__ == "__main__":
    main()