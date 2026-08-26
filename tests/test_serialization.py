"""Tests for ONNX export and pickle/ONNX parity."""
import os
import pytest
from mlops_practitioner.export import load_pickle_model, export_to_onnx, check_parity


@pytest.fixture(scope="module")
def onnx_path(tmp_path_factory):
    dv, model = load_pickle_model()
    path = str(tmp_path_factory.mktemp("onnx") / "model.onnx")
    export_to_onnx(dv, model, output_path=path)
    return path


def test_onnx_file_is_created(onnx_path):
    assert os.path.exists(onnx_path)
    assert os.path.getsize(onnx_path) > 0


def test_onnx_parity_with_pickle(onnx_path):
    dv, model = load_pickle_model()
    sample_dicts = [{"PU_DO": "43_151", "trip_distance": 2.5}]
    assert check_parity(dv, model, onnx_path, sample_dicts) is True