# Module 1 Report — From Notebook to Production Service

**Project:** MLOps Practitioner — Trip Duration Predictor
**Author:** Hussein Waleed
**Branch:** `module-1-packaging`
**Tag:** `v0.1.0`

## 1. Model Performance

Baseline model: `RandomForestRegressor` (n_estimators=100, max_depth=10) trained on NYC Green Taxi trip data (January 2023), using `trip_distance` and `PU_DO` (pickup-dropoff location pair) as features.

| Metric | Value |
|---|---|
| MAE  | 3.933 minutes |
| RMSE | 6.069 minutes |

**Note:** RMSE is notably higher than MAE (~54% higher), indicating a small number of trips with large prediction errors (outliers), while most trips are predicted with reasonable accuracy. This is expected given the model uses only two simple features and no hyperparameter tuning — it serves as a baseline for future iterations, not a final accuracy target.

Result was reproduced identically after refactoring the notebook into the `mlops_practitioner` package (`python -m mlops_practitioner.train`), confirming the packaging preserved the original logic exactly.

## 2. Serialization Comparison: Pickle vs ONNX

| Format | Avg Latency | p95 Latency |
|---|---|---|
| Pickle | 35.619 ms | 63.872 ms |
| ONNX   | 0.126 ms  | 0.155 ms  |

**Parity check:** Passed (`np.allclose`, atol=1e-3) — ONNX predictions match Pickle predictions exactly within tolerance.

**Speedup:** ONNX is ~283x faster than Pickle for single-record inference.

**Trade-off notes:**
- Pickle: human-readable Python object, easy to debug, but carries deserialization security risk (arbitrary code execution) and is Python-only.
- ONNX: cross-language, schema-enforced, much faster runtime (backed by C++ ONNX Runtime), but the model graph itself is not human-readable and harder to debug directly.
- Decision: ONNX is used for the production API endpoint due to the significant latency advantage; Pickle remains the training-time artifact format.

## 3. Docker Image Size Comparison

| Build Type | Image Size |
|---|---|
| Multi-stage (production) | 240 MB |
| Single-stage | 242 MB |
| Difference | ~2 MB (~0.8%) |

**Analysis:** The size difference is small in this case because all Python dependencies (`pandas`, `scikit-learn`, `fastapi`, etc.) are installed from pre-built wheels for `linux/amd64` — no compilation toolchain (e.g. `gcc`, `build-essential`) was ever needed in the builder stage, so there was little build bloat to strip out. The main benefit of the multi-stage build here is therefore not size but **reduced attack surface**: the final runtime image contains no `uv`/`pip` build tooling, only the installed packages and application code, running as a non-root user (`appuser`).

## 4. Test Suite & Coverage

| Metric | Value |
|---|---|
| Total tests | 22 |
| Passed | 22 |
| Coverage | 70.62% |
| Coverage gate | ≥ 70% (passed) |

Test files: `test_data.py`, `test_features.py`, `test_predict.py`, `test_api.py`, `test_serialization.py`.

Coverage by module (before adding `test_data.py`, for reference):

| Module | Coverage |
|---|---|
| `api/main.py` | 94% |
| `api/schemas.py` | 100% |
| `config.py` | 100% |
| `data.py` | 0% → covered after adding `test_data.py` |
| `export.py` | 58% |
| `features.py` | 100% |
| `logging_conf.py` | 95% |
| `predict.py` | 100% |
| `train.py` | 0% (not directly unit-tested; exercised indirectly via `predict`/`export` using the already-trained artifact) |

## 5. Docker Hub Deployment

- Image published to: `husseinwaleed/mlops-practitioner-api` (tags: `0.1.0`, `latest`)
- Verified end-to-end: removed local image, pulled fresh from Docker Hub, ran container, confirmed `/health` returns `200 OK` from a clean pull.
- Container runs as non-root user (`appuser`, uid 1000).
- `HEALTHCHECK` configured (30s interval, 3 retries) against `/health`.

## 6. Self-Assessment — MLOps Maturity After Module 1

At this stage, the project has moved from a "manual, ad-hoc" notebook workflow to a **packaged, tested, containerized, and deployable service** with structured logging and request tracing (correlation IDs). What is still missing to reach a fully mature MLOps setup (to be addressed in later modules): experiment tracking (MLflow), data/model versioning (DVC), CI/CD automation, and production monitoring/alerting. Module 1 established the engineering foundation (packaging, API, containerization, testing) that these later layers will build on top of, without needing to rewrite the core service.

## 7. Key Failure Modes Encountered & Resolved

- Relative paths (`../models/...`) copied over from the notebook context caused `FileNotFoundError` once the code ran from the project root instead of `notebooks/`; fixed by making all config paths relative to the project root.
- Stale `pyproject.toml` references to an earlier package name (`src/prodml`) caused `coverage` to report 0% despite all tests passing, since it was pointed at a nonexistent module path; fixed by aligning `--cov=`, `packages=`, and the script entry point with the actual package name (`mlops_practitioner`).
- A floating-point determinism test failed on an exact `==` comparison between two model predictions differing in the last decimal digits; fixed by using `pytest.approx` instead of strict equality.
- Working across two active environments (conda `mlops` layered with the `uv` `.venv`) caused Python to resolve some standard-library modules from the wrong installation; resolved by deactivating conda and running exclusively inside the `uv`-managed environment.