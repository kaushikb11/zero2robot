import onnx
import pytest
import torch

from curriculum.common.assert_parity import assert_parity
from curriculum.common.export_onnx import (
    CONTRACT_VERSION,
    INPUT_NAME,
    META_ACT_DIM_KEY,
    META_OBS_DIM_KEY,
    META_VERSION_KEY,
    OUTPUT_NAME,
    export_policy,
)

OBS_DIM = 4
ACT_DIM = 2


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(OBS_DIM, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, ACT_DIM),
        )

    def forward(self, observation):
        return self.net(observation)


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return TinyPolicy().eval()


def test_export_stamps_contract_metadata(model, tmp_path):
    out_path = export_policy(model, OBS_DIM, ACT_DIM, tmp_path / "policy.onnx")
    proto = onnx.load(str(out_path))
    meta = {p.key: p.value for p in proto.metadata_props}

    assert meta[META_VERSION_KEY] == CONTRACT_VERSION == "v1"
    assert meta[META_OBS_DIM_KEY] == str(OBS_DIM)
    assert meta[META_ACT_DIM_KEY] == str(ACT_DIM)

    graph = proto.graph
    assert graph.input[0].name == INPUT_NAME == "observation"
    assert graph.output[0].name == OUTPUT_NAME == "action"
    in_shape = [
        d.dim_value for d in graph.input[0].type.tensor_type.shape.dim
    ]
    out_shape = [
        d.dim_value for d in graph.output[0].type.tensor_type.shape.dim
    ]
    assert in_shape == [1, OBS_DIM]
    assert out_shape == [1, ACT_DIM]


def test_export_rejects_wrong_act_dim(model, tmp_path):
    with pytest.raises(ValueError, match="contract v1"):
        export_policy(model, OBS_DIM, ACT_DIM + 1, tmp_path / "policy.onnx")


def test_parity_roundtrip(model, tmp_path):
    out_path = export_policy(model, OBS_DIM, ACT_DIM, tmp_path / "policy.onnx")
    max_delta = assert_parity(model, out_path, OBS_DIM, n=32, seed=0, tol=1e-4)
    assert 0.0 <= max_delta < 1e-4


def test_parity_catches_perturbed_weights(model, tmp_path):
    out_path = export_policy(model, OBS_DIM, ACT_DIM, tmp_path / "policy.onnx")
    with torch.no_grad():
        model.net[0].weight += 1.0  # torch and ONNX now disagree
    with pytest.raises(AssertionError, match="parity FAILED"):
        assert_parity(model, out_path, OBS_DIM, n=32, seed=0, tol=1e-4)
