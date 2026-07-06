"""Shared fixtures: sample contract-v1 ONNX policies built via the REAL export.

We deliberately build sample policies through curriculum/common/export_onnx.py
(the same path a chapter uses), so the tests exercise the actual contract-v1
metadata the grader validates — not a hand-rolled stand-in.
"""

from __future__ import annotations

from pathlib import Path

import onnx
import pytest
import torch

from curriculum.common.envs.pusht import PushTEnv
from curriculum.common.export_onnx import export_policy


class TinyPolicy(torch.nn.Module):
    """Deterministic obs->action MLP. Fixed-seed weights => a fixed policy.

    Not trained — the tests care about *determinism and contract conformance*,
    not skill. A tanh keeps actions in [-1, 1] (the env clips anyway)."""

    def __init__(self, obs_dim: int, act_dim: int, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.fc1 = torch.nn.Linear(obs_dim, 32)
        self.fc2 = torch.nn.Linear(32, act_dim)
        with torch.no_grad():
            for layer in (self.fc1, self.fc2):
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=gen) * 0.3)
                layer.bias.zero_()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc2(torch.relu(self.fc1(obs))))


def _export(path: Path, obs_dim: int, act_dim: int, seed: int = 0) -> Path:
    model = TinyPolicy(obs_dim, act_dim, seed=seed)
    return export_policy(model, obs_dim, act_dim, path)


@pytest.fixture(scope="session")
def sample_onnx(tmp_path_factory) -> Path:
    """A conformant contract-v1 PushT policy (obs=10, act=2)."""
    path = tmp_path_factory.mktemp("policy") / "sample.onnx"
    return _export(path, PushTEnv.OBS_DIM, PushTEnv.ACT_DIM)


@pytest.fixture
def wrong_dims_onnx(tmp_path) -> Path:
    """Conformant metadata but the WRONG dims for PushT (obs=8, act=3)."""
    return _export(tmp_path / "wrong_dims.onnx", 8, 3)


@pytest.fixture
def no_metadata_onnx(tmp_path, sample_onnx) -> Path:
    """A structurally valid ONNX with the contract metadata stripped."""
    proto = onnx.load(str(sample_onnx))
    del proto.metadata_props[:]
    out = tmp_path / "no_meta.onnx"
    onnx.save(proto, str(out))
    return out
