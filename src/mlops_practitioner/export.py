"""Export the trained model to ONNX and verify parity with the pickle version."""
import pickle
import time
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

from mlops_practitioner.config import settings
from mlops_practitioner.logging_conf import get_logger

logger = get_logger(__name__)


def load_pickle_model(path: str | None = None):
    path = path or settings.model_path
    with open(path, "rb") as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def export_to_onnx(dv, model, output_path: str = "models/baseline.onnx") -> str:
    """Convert the sklearn pipeline (DictVectorizer output -> RandomForest) to ONNX."""
    n_features = len(dv.get_feature_names_out())
    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=15)

    with open(output_path, "wb") as f_out:
        f_out.write(onnx_model.SerializeToString())

    logger.info(f"ONNX model exported to {output_path}")
    return output_path


def check_parity(dv, model, onnx_path: str, sample_dicts: list[dict], atol: float = 1e-3) -> bool:
    """Compare pickle predictions vs ONNX predictions on the same input."""
    X = dv.transform(sample_dicts).toarray().astype(np.float32)

    pkl_preds = model.predict(X)

    sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_preds = sess.run(None, {input_name: X})[0].flatten()

    match = np.allclose(pkl_preds, onnx_preds, atol=atol)
    max_diff = np.max(np.abs(pkl_preds - onnx_preds))
    logger.info(f"Parity check: match={match}, max_diff={max_diff:.6f}")
    return match


def benchmark_latency(predict_fn, sample_dicts: list[dict], n_runs: int = 500) -> dict:
    """Measure average and p95 latency in milliseconds."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn(sample_dicts[0])
        times.append((time.perf_counter() - start) * 1000)

    times.sort()
    return {
        "avg_ms": sum(times) / len(times),
        "p95_ms": times[int(len(times) * 0.95)],
    }


def main() -> None:
    dv, model = load_pickle_model()

    sample_dicts = [
        {"PU_DO": "43_151", "trip_distance": 2.5},
        {"PU_DO": "166_239", "trip_distance": 1.2},
        {"PU_DO": "41_42", "trip_distance": 5.8},
    ]

    onnx_path = export_to_onnx(dv, model)
    ok = check_parity(dv, model, onnx_path, sample_dicts)
    print(f"Parity check passed: {ok}")

    # Benchmark pickle
    def pkl_predict(d):
        X = dv.transform([d])
        return model.predict(X)[0]

    pkl_stats = benchmark_latency(pkl_predict, sample_dicts)
    print(f"Pickle latency: avg={pkl_stats['avg_ms']:.3f}ms p95={pkl_stats['p95_ms']:.3f}ms")

    # Benchmark ONNX
    sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    def onnx_predict(d):
        X = dv.transform([d]).toarray().astype(np.float32)
        return sess.run(None, {input_name: X})[0]

    onnx_stats = benchmark_latency(onnx_predict, sample_dicts)
    print(f"ONNX latency:   avg={onnx_stats['avg_ms']:.3f}ms p95={onnx_stats['p95_ms']:.3f}ms")


if __name__ == "__main__":
    main()