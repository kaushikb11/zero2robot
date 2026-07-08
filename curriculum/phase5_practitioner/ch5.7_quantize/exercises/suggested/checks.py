"""SUGGESTED local pytest checks for the ch5.7 exercise candidates.

Run from anywhere:
    pytest curriculum/phase5_practitioner/ch5.7_quantize/exercises/suggested/checks.py

Conventions (match ch1.6 / ch5.1):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces predict-before-run;
  locally we only verify the recorded choice.
- ex2 (quantize_weight completion) is FAST + self-contained (pure numpy) — it runs in `make check`.
  It SKIPS while `quantize_weight` raises NotImplementedError, then asserts the completion matches a
  reference, that per-channel round-trip error is strictly < per-tensor, and that naive no-scale
  rounding collapses the tensor to zeros.
- ex1 and ex3 RUN quantize.py via subprocess (trains the MLP, ~15-25 s CPU) so they are
  @pytest.mark.slow — excluded from `make check`. They assert the DIRECTION (per-channel recovers;
  size shrinks ~4x; int8 not faster; bad calibration explodes), never an exact MSE — int8 arithmetic
  is bitwise on CPU but the checks stay conservative and platform-robust.
- Reference bands live in meta.yaml with provenance (exercise-spec: no bare magic numbers) — read
  them, don't inline. Bands are PROVISIONAL pending author reverification.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
QUANTIZE = REPO / "curriculum/phase5_practitioner/ch5.7_quantize/quantize.py"
sys.path.insert(0, str(HERE))

import ex1_predict_triangle as ex1  # noqa: E402
import ex2_completion_quantize as ex2  # noqa: E402
import ex3_predict_calib_break as ex3  # noqa: E402

CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
RC = CHECKS["exercise_config"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex3": CHECKS["ex3"]["answer"]}
QMAX = 127


def run_quantize(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(QUANTIZE), "--seed", "0", "--no-rerun", "--out", str(out),
           "--demos", str(RC["demos"]), "--epochs", str(RC["epochs"]),
           "--calib_episodes", str(RC["calib_episodes"]), "--eval_episodes", str(RC["eval_episodes"]), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


# ----------------------------------------------------------------- predictions

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1 first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "run ex1: per-channel is most accurate, ~4x smaller, and int8 is NOT faster here"


def test_ex3_prediction_recorded():
    if ex3.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex3 first")
    assert ex3.PREDICTION == ANSWER_KEY["ex3"], "run ex3: narrow calibration saturates activations and explodes the full-integer error"


# ------------------------------------------ ex2 quantize_weight completion (fast)

def _ref_quantize_weight(W: np.ndarray, per_channel: bool):
    r = np.abs(W).max(axis=1, keepdims=True) if per_channel else np.abs(W).max()
    scale = np.maximum(np.asarray(r, np.float32) / QMAX, 1e-8)
    q = np.clip(np.round(W / scale), -QMAX, QMAX).astype(np.int8)
    return q, scale


def _fixture() -> np.ndarray:
    # a weight matrix with a deliberately fat-tailed first row, so per-channel clearly wins
    rng = np.random.default_rng(0)
    W = (0.05 * rng.standard_normal((8, 16))).astype(np.float32)
    W[0] *= 12.0  # one channel with a much larger range — the thing per-tensor scaling wastes bits on
    return W


def _roundtrip_err(W, q, scale) -> float:
    return float(np.abs(W - q.astype(np.float32) * scale).mean())


def test_ex2_matches_reference():
    W = _fixture()
    for per_channel in (False, True):
        try:
            q, scale = ex2.quantize_weight(W, per_channel)
        except NotImplementedError:
            pytest.skip("quantize_weight not implemented yet — that's the exercise")
        ref_q, ref_scale = _ref_quantize_weight(W, per_channel)
        assert np.asarray(q).shape == W.shape, f"q must match W's shape, got {np.asarray(q).shape}"
        assert np.allclose(np.asarray(q, np.float32) * np.asarray(scale), ref_q.astype(np.float32) * ref_scale,
                           atol=CHECKS["ex2"]["abs_tol"]), "dequantized weights differ from the reference symmetric int8"


def test_ex2_per_channel_beats_per_tensor():
    W = _fixture()
    try:
        qt, st = ex2.quantize_weight(W, False)
        qc, sc = ex2.quantize_weight(W, True)
    except NotImplementedError:
        pytest.skip("quantize_weight not implemented yet — that's the exercise")
    if not CHECKS["exercise_config"]["weight_rt_strict_improvement"]:
        return
    assert _roundtrip_err(W, qc, sc) < _roundtrip_err(W, qt, st), \
        "per-channel round-trip error must be strictly smaller than per-tensor (a per-row scale is a refinement)"


def test_ex2_naive_rounding_collapses():
    # The misconception: rounding with NO scale. Trained policy weights are small (std ~0.1),
    # so nearly all of them round to zero — proof the SCALE, not the rounding, carries signal.
    W = (0.12 * np.random.default_rng(1).standard_normal((8, 16))).astype(np.float32)
    zero_frac = float((np.round(W) == 0).mean())
    assert zero_frac >= CHECKS["exercise_config"]["min_naive_round_zero_frac"], \
        f"naive round-to-int8 should collapse most weights to zero, got {zero_frac:.2f}"


# ---------------------------------------------------------- reproduce (SLOW: trains the MLP)

@pytest.mark.slow
def test_ex1_triangle_direction(tmp_path):
    m = run_quantize(tmp_path / "clean")
    assert m["mse_recovery_ratio"] >= RC["min_mse_recovery_ratio"], \
        f"per-channel INT8 must recover action error over per-tensor by >= {RC['min_mse_recovery_ratio']}x: {m}"
    assert m["size_ratio_fp32_over_int8"] >= RC["min_size_ratio"], \
        f"int8 must be >= {RC['min_size_ratio']}x smaller than fp32: {m}"
    assert m["weight_roundtrip_err_per_channel"] < m["weight_roundtrip_err_per_tensor"], \
        f"per-channel weight round-trip error must be strictly smaller than per-tensor: {m}"
    # HONEST + PROVISIONAL (platform-dependent): naive int8 is not faster on a laptop CPU.
    assert m["int8_faster_than_fp32"] is False, \
        f"naive int8 should NOT be faster than fp32 on this CPU (dequant overhead, no fused kernel): {m}"


@pytest.mark.slow
def test_ex3_bad_calibration_explodes(tmp_path):
    good = run_quantize(tmp_path / "good")
    bad = run_quantize(tmp_path / "bad", ["--break", "bad_calib"])
    assert bad["full_int8_action_mse"] >= RC["min_break_explosion"] * good["full_int8_action_mse"], \
        f"narrow calibration must explode the full-integer error by >= {RC['min_break_explosion']}x: good={good}, bad={bad}"
    # The break only attacks the activation path — the weight-only triangle is untouched.
    assert bad["per_tensor_action_mse"] == good["per_tensor_action_mse"] and \
        bad["per_channel_action_mse"] == good["per_channel_action_mse"], \
        "bad calibration must NOT change the weight-only triangle (it only affects the activation scales)"
