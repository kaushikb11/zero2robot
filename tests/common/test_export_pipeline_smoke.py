"""CI parity smoke for the per-chapter ONNX export entrypoints (contract v1).

Every demo policy that ships as `<x>.onnx` is produced by a chapter's
export_*_onnx.py: rebuild the trained architecture, load its .pt, export via
curriculum/common/export_onnx.py, and gate on assert_parity (torch vs
onnxruntime, max |action delta| < 1e-4). This test exercises that whole path
WITHOUT a checkpoint or a trained binary — each script exposes `smoke_policy()`,
a random-init instance of the real architecture — so CI proves the pipeline on
every push while no .onnx is ever committed (invariant 5). Real .pt checkpoints
follow the identical export_policy -> assert_parity path; only the weights differ.

The export scripts live in chapter dirs whose names contain dots (ch2.2_sac,
ch4.3_serl), so they cannot be imported as packages — we load each by file path.
"""

import importlib.util
from pathlib import Path

import onnx
import pytest

from curriculum.common.assert_parity import assert_parity
from curriculum.common.export_onnx import INPUT_NAME, OUTPUT_NAME, export_policy

REPO_ROOT = Path(__file__).resolve().parents[2]

# (id, script path, expected obs_dim, expected act_dim) — the demo tensor contracts.
EXPORTERS = [
    ("ppo", "curriculum/phase2_reinforcement/ch2.1_ppo/export_ppo_onnx.py", 5, 1),
    ("sac", "curriculum/phase2_reinforcement/ch2.2_sac/export_sac_onnx.py", 8, 2),
    ("walk", "curriculum/phase2_reinforcement/ch2.5_walk/export_walk_onnx.py", 23, 8),
    ("serl", "curriculum/phase4_capstone/ch4.3_serl/export_serl_onnx.py", 8, 2),
    ("offline", "curriculum/phase4_capstone/ch4_offline_primer/export_offline_onnx.py", 8, 2),
]


def _load(script_rel: str):
    path = REPO_ROOT / script_rel
    spec = importlib.util.spec_from_file_location(f"export_smoke_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # runs class defs only; main() is __name__-guarded
    return module


@pytest.mark.parametrize("name, script, obs_dim, act_dim", EXPORTERS, ids=[e[0] for e in EXPORTERS])
def test_export_smoke_parity(name, script, obs_dim, act_dim, tmp_path):
    module = _load(script)
    policy, got_obs, got_act = module.smoke_policy(seed=0)
    assert (got_obs, got_act) == (obs_dim, act_dim), "smoke_policy dims drifted from the demo contract"

    out_path = export_policy(policy, got_obs, got_act, tmp_path / f"{name}.onnx")
    max_delta = assert_parity(policy, out_path, got_obs, n=32, seed=0, tol=1e-4)
    assert 0.0 <= max_delta < 1e-4

    # The ONNX signature the playground consumes: observation[1,obs] -> action[1,act].
    graph = onnx.load(str(out_path)).graph
    assert graph.input[0].name == INPUT_NAME == "observation"
    assert graph.output[0].name == OUTPUT_NAME == "action"
    in_shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    out_shape = [d.dim_value for d in graph.output[0].type.tensor_type.shape.dim]
    assert in_shape == [1, obs_dim]
    assert out_shape == [1, act_dim]


def test_smoke_export_is_deterministic(tmp_path):
    """Same seed -> identical exported bytes: export is deterministic given the
    weights (invariant 2), so re-running the pipeline never churns the artifact."""
    module = _load(EXPORTERS[1][1])  # sac
    first = export_policy(*module.smoke_policy(seed=0), tmp_path / "a.onnx").read_bytes()
    second = export_policy(*module.smoke_policy(seed=0), tmp_path / "b.onnx").read_bytes()
    assert first == second
