"""SUGGESTED local pytest checks for the ch1.8 exercise candidates.

Run from anywhere:
    pytest curriculum/phase1_imitation/ch1.8_vla/exercises/suggested/checks.py

Conventions (match ch1.1 / ch1.3 / ch1.5 / ch1.7):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the recorded choice.
- The two self-contained gates (ex3 masked-loss bug-hunt, ex4 attention completion)
  are FAST + deterministic — they run in `make check`.
- ex3 SKIPS while the sum-weighted (buggy) loss is still present; asserts the fix gives
  the per-example average on a fixture where the two differ.
- ex4 SKIPS while `attention` raises NotImplementedError.
- The two reproduce checks (ex1 blind==sighted, ex2 pusht>>aloha) TRAIN the policy at the
  full config and are @pytest.mark.slow — excluded from `make check` (which runs only the
  fast gates). A reduced budget yields 0.0 for BOTH tasks (like ch1.5's reduced policy),
  so these must run at defaults; CPU training is deterministic, so seed 0 reproduces.
- Reference bands live in meta.yaml with provenance (exercise-spec: no bare magic
  numbers) — read them, don't inline.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
VLA = REPO / "curriculum/phase1_imitation/ch1.8_vla/vla.py"
sys.path.insert(0, str(HERE))

import ex1_predict_blind as ex1  # noqa: E402
import ex2_predict_pusht_vs_aloha as ex2  # noqa: E402
import ex3_bughunt_masked_loss as ex3  # noqa: E402
import ex4_completion_attention as ex4  # noqa: E402

CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
RC = CHECKS["exercise_config"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex2": CHECKS["ex2"]["answer"]}


def run_vla(data: Path, out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(VLA), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--data", str(data), "--out", str(out),
           "--episodes_per_task", str(RC["episodes_per_task"]), "--epochs", str(RC["epochs"]),
           "--eval_episodes", str(RC["eval_episodes"]), "--model_dim", str(RC["model_dim"]),
           "--hidden", str(RC["hidden"]), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


# ----------------------------------------------------------------- predictions

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1 first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "run ex1 and compare sighted vs --break blind PushT success"


def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2 first")
    assert ex2.PREDICTION == ANSWER_KEY["ex2"], "run ex2: the tiny VLA learns PushT but fails ALOHA"


# ------------------------------------------ ex3 masked-loss bug-hunt (fast)
# A fixture where per-EXAMPLE and per-DIM weighting disagree: example A has 2 valid dims
# with squared-error 4 each; example B has 6 valid dims with squared-error 1 each. The
# correct per-example average is mean(4, 1) = 2.5; the buggy sum-weighted loss is
# (2*4 + 6*1)/(2+6) = 1.75, under-counting the low-DOF task. (A's padded dims carry
# garbage the mask must ignore.)
def _fixture():
    pred = torch.zeros(2, 6)
    target = torch.tensor([[2.0, 2.0, 9.0, 9.0, 9.0, 9.0],   # SE 4 on the 2 valid dims
                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])   # SE 1 on all 6 valid dims
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    return pred, target, mask


def test_ex3_per_example_masked_loss():
    got = float(ex3.masked_flow_loss(*_fixture()))
    if abs(got - 1.75) < 1e-4:
        pytest.skip("still the sum-weighted (buggy) loss (1.75) — fix it to the per-example average")
    assert abs(got - 2.5) < 1e-5, f"per-example masked loss should be 2.5 on this fixture, got {got}"


# ------------------------------------------- ex4 attention completion (fast)
def _ref_attention(q, k, v, key_pad):
    hd = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    scores = scores.masked_fill(key_pad[:, None, None, :], float("-inf"))
    return scores.softmax(dim=-1) @ v


def _attn_inputs():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 2, 4, 8) for _ in range(3))
    key_pad = torch.tensor([[False, False, True, True], [False, True, False, False]])
    return q, k, v, key_pad


def test_ex4_attention_matches_reference():
    q, k, v, key_pad = _attn_inputs()
    try:
        got = ex4.attention(q, k, v, key_pad)
    except NotImplementedError:
        pytest.skip("attention not implemented yet — that's the exercise")
    want = _ref_attention(q, k, v, key_pad)
    assert got.shape == want.shape, f"expected shape {tuple(want.shape)}, got {tuple(got.shape)}"
    assert torch.allclose(got, want, atol=CHECKS["ex4"]["abs_tol"]), "attention output differs from the reference"


def test_ex4_ignores_padded_keys():
    q, k, v, key_pad = _attn_inputs()
    try:
        out1 = ex4.attention(q, k, v, key_pad)
    except NotImplementedError:
        pytest.skip("attention not implemented yet — that's the exercise")
    pad = key_pad[:, None, :, None].expand_as(v)   # (B, heads, L, head_dim)
    v2 = torch.where(pad, torch.full_like(v, 1e6), v)  # blow up VALUES at padded key positions
    out2 = ex4.attention(q, k, v2, key_pad)
    assert torch.allclose(out1, out2, atol=1e-3), "padded keys must get zero attention weight (mask them before softmax)"


# ---------------------------------------------------------- reproduce (SLOW: trains)

@pytest.mark.slow
def test_ex1_blind_barely_changes_pusht(tmp_path):
    data = tmp_path / "data"  # both runs share one regenerated dataset (identical frozen features)
    sighted = run_vla(data, tmp_path / "sighted")
    blind = run_vla(data, tmp_path / "blind", ["--break", "blind"])
    assert sighted["pusht_success_rate"] - sighted["baseline_pusht_success_rate"] >= RC["min_trained_over_untrained"], \
        f"trained PushT should beat untrained: {sighted}"
    assert sighted["pusht_success_rate"] - blind["pusht_success_rate"] <= RC["max_blind_drop"], \
        f"zeroing vision should barely change PushT (it is not load-bearing): sighted={sighted['pusht_success_rate']}, blind={blind['pusht_success_rate']}"


@pytest.mark.slow
def test_ex2_pusht_beats_aloha(tmp_path):
    m = run_vla(tmp_path / "data", tmp_path / "out")
    assert m["pusht_success_rate"] - m["baseline_pusht_success_rate"] >= RC["min_trained_over_untrained"], \
        f"trained PushT should beat untrained: {m}"
    assert m["pusht_success_rate"] - m["aloha_success_rate"] >= RC["min_pusht_over_aloha"], \
        f"the tiny VLA should learn PushT well above ALOHA: {m}"
