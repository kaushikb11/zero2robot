"""SUGGESTED local pytest checks for the ch5.8 exercise candidates.

Run from anywhere:
    pytest curriculum/phase5_practitioner/ch5.8_real_loop/exercises/suggested/checks.py

Conventions (match ch5.1 / ch1.1):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces predict-before-run;
  locally we only verify the recorded choice.
- ex3 (expert_action completion) is FAST + self-contained (pure numpy, no MuJoCo) — it runs in
  `make check`. It SKIPS while expert_action raises NotImplementedError, then asserts the completion
  matches the reference joint target AND that shoulder_pan actually depends on the box (a constant
  answer — the classic "why can't the clone reach a box that moves" bug — is detectably wrong).
- ex1 and ex2 run the FULL loop (~1-2 min CPU each: fetch-cached + record + train + deploy) via
  subprocess, so they are @pytest.mark.slow — excluded from `make check`. They assert the DIRECTION
  (clone >> baselines; obs_swap collapses deployment while training stays clean), never an exact % —
  deploy success is a held-out RATE and MuJoCo contact/servo settling are not bitwise across arches (ch1.6).
- Reference bands live in meta.yaml with provenance (exercise-spec: no bare magic numbers) — read them,
  don't inline. Bands are PROVISIONAL pending author reverification.
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
ARTIFACT = REPO / "curriculum/phase5_practitioner/ch5.8_real_loop/real_loop.py"
sys.path.insert(0, str(HERE))

import ex1_predict_loop as ex1  # noqa: E402
import ex2_predict_obs_swap as ex2  # noqa: E402
import ex3_completion_expert as ex3  # noqa: E402

CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
RC = CHECKS["exercise_config"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex2": CHECKS["ex2"]["answer"]}


def run_loop(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(ARTIFACT), "--seed", "0", "--device", "cpu", "--no-rerun", "--out", str(out),
           "--demos", str(RC["demos"]), "--epochs", str(RC["epochs"]),
           "--eval_episodes", str(RC["eval_episodes"]), "--steps", str(RC["steps"]), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


# ----------------------------------------------------------------- predictions

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1 first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "run ex1 and order clone vs no-op vs random"


def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2 first")
    assert ex2.PREDICTION == ANSWER_KEY["ex2"], "run ex2: obs_swap collapses deploy, training loss is unchanged"


# ------------------------------------------ ex3 expert_action completion (fast)

def _ref_expert(box_xyz: np.ndarray) -> np.ndarray:
    pan = -np.arctan2(box_xyz[1], box_xyz[0]) / ex3.PAN_GAIN
    return np.array([pan, ex3.REACH_POSE[0], ex3.REACH_POSE[1], ex3.REACH_POSE[2], 0.0, 0.3], np.float32)


_BOXES = [np.array([0.28, 0.10, 0.03]), np.array([0.30, -0.12, 0.03]), np.array([0.27, 0.0, 0.03])]


def test_ex3_matches_reference():
    try:
        got = [np.asarray(ex3.expert_action(b), np.float64) for b in _BOXES]
    except NotImplementedError:
        pytest.skip("expert_action not implemented yet — that's the exercise")
    for b, g in zip(_BOXES, got):
        want = _ref_expert(b).astype(np.float64)
        assert g.shape == (6,), f"expected a 6-D joint target, got shape {g.shape}"
        assert np.allclose(g, want, atol=CHECKS["ex3"]["abs_tol"]), \
            f"expert_action({b.tolist()}) = {g.tolist()} != reference {want.tolist()}"


def test_ex3_pan_depends_on_box():
    # shoulder_pan MUST steer with the box's y (its azimuth). A constant answer — the bug where every
    # demo drives to the same spot and the clone can't reach a box that moves — is detectably wrong.
    try:
        left = np.asarray(ex3.expert_action(np.array([0.28, 0.12, 0.03])), np.float64)
        right = np.asarray(ex3.expert_action(np.array([0.28, -0.12, 0.03])), np.float64)
    except NotImplementedError:
        pytest.skip("expert_action not implemented yet — that's the exercise")
    assert abs(left[0] - right[0]) > 0.1, \
        "shoulder_pan is (nearly) constant across box_y — the arm can't aim at a box that moves"


# ---------------------------------------------------------- reproduce (SLOW: full loop)

@pytest.mark.slow
def test_ex1_clone_beats_baselines(tmp_path):
    m = run_loop(tmp_path / "clean")
    assert m["clone_success_rate"] >= RC["min_clone_rate"], \
        f"the recorded-then-cloned policy must reproduce the reach: {m}"
    assert m["noop_success_rate"] <= RC["max_baseline_rate"], f"no-op should fail the reach: {m}"
    assert m["random_success_rate"] <= RC["max_baseline_rate"], f"random should fail the reach: {m}"
    assert m["clone_success_rate"] - max(m["noop_success_rate"], m["random_success_rate"]) >= RC["min_clone_over_baseline"], \
        f"the clone must beat both baselines by a wide margin (the loop closes): {m}"
    assert m["clone_beats_baselines"] is True, m


@pytest.mark.slow
def test_ex2_obs_swap_collapses_deploy_not_training(tmp_path):
    clean = run_loop(tmp_path / "clean")
    swap = run_loop(tmp_path / "swap", ["--break", "obs_swap"])
    # Training is untouched: the loss on the identical recorded dataset barely moves.
    assert abs(swap["final_train_loss"] - clean["final_train_loss"]) < 1e-3, \
        f"obs_swap must NOT change training (it is a deploy-time bug): clean={clean}, swap={swap}"
    # Deployment collapses: the mis-wired observation makes the arm reach the wrong way.
    assert swap["clone_success_rate"] <= RC["max_break_rate"], \
        f"obs_swap should collapse deployment: {swap}"
    assert clean["clone_success_rate"] - swap["clone_success_rate"] >= RC["min_clean_over_break"], \
        f"clean deploy must beat the obs_swap deploy by a wide margin: clean={clean}, swap={swap}"
