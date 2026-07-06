"""SUGGESTED local pytest checks for the ch3.6 exercise candidates.

Run from anywhere:  pytest curriculum/phase3_advanced/ch3.6_compare/exercises/suggested/checks.py

Conventions (mirroring ch3.3-3.5):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- ex2's completion checks SKIP while a blank is still unfilled (that IS the
  exercise) and assert reference agreement once the learner fills them in.
- The smoke-signature check runs compare.py --smoke — HERMETIC (a fresh seeded
  untrained policy, no checkpoint, no dataset), so it always runs in CI and is
  bitwise deterministic (numpy engine + torch-CPU eval).
- The trained-policy checks (ex1, ex3) SKIP when no trained ch1.1 checkpoint is on
  disk (CI is hermetic): the FULL-CIRCLE numbers need a policy the learner trains.
- Reference bands + provenance live in meta.yaml exercise_checks — read them.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parents[1]  # .../ch3.6_compare/
ARTIFACT = CHAPTER / "compare.py"
REPO_ROOT = CHAPTER.parents[2]
sys.path.insert(0, str(HERE))

import ex1_predict_transfer as ex1  # noqa: E402
import ex2_completion_contract as ex2  # noqa: E402
import ex3_gap_knob as ex3  # noqa: E402

CHECKS = yaml.safe_load((CHAPTER / "meta.yaml").read_text())["exercise_checks"]
SMOKE = CHECKS["smoke"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex3": CHECKS["ex3"]["answer"]}  # kept out of the exercise files


# --- ex2 reference (the answer key — kept here, not in the exercise) ---
def ref_block_yaw(p_bar, p_stem):
    d = p_stem - p_bar
    return (np.arctan2(d[1], d[0]) + np.pi / 2.0 + np.pi) % (2.0 * np.pi) - np.pi


def ref_engine_obs(pusher_xy, bar_xy, stem_xy):
    px, py = pusher_xy
    tx, ty = bar_xy
    tyaw = ref_block_yaw(bar_xy, stem_xy)
    return np.array([px, py, tx, ty, np.sin(tyaw), np.cos(tyaw), 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def run_smoke() -> dict:
    with tempfile.TemporaryDirectory(prefix="z2r-ch36-") as tmp:
        cmd = [sys.executable, str(ARTIFACT), "--smoke", "--seed", "0", "--no-rerun", "--out", tmp]
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO_ROOT)
        return json.loads((Path(tmp) / "metrics.json").read_text())


# --- smoke: the hermetic signature reproduces (always runs in CI) ---
def test_smoke_signature_reproduces():
    m = run_smoke()
    tol = SMOKE["reproduce_abs"]
    assert m["mj_success_rate"] == pytest.approx(SMOKE["mj_success_rate"], abs=tol)
    assert m["engine_success_rate"] == pytest.approx(SMOKE["engine_success_rate"], abs=tol)
    assert m["mean_pos_divergence_m"] == pytest.approx(SMOKE["mean_pos_divergence_m"], abs=tol)
    assert m["mean_ang_divergence_rad"] == pytest.approx(SMOKE["mean_ang_divergence_rad"], abs=tol)


# --- ex1: the transfer is partial and diverges (needs a trained policy) ---
def test_ex1_transfer_is_partial():
    policy = ex1.find_policy()
    if policy is None:
        pytest.skip("no trained ch1.1 policy on disk — train one to check the full-circle transfer")
    m = ex1.run_compare(policy, "--episodes", "50")
    # The policy transfers IMPERFECTLY: engine success is below MuJoCo's, and the
    # block poses diverge (angle divergence is real, not ~0). Ordering, not exact number.
    assert m["engine_success_rate"] < m["mj_success_rate"], "a simplified engine should not MATCH MuJoCo's success — that would be suspicious"
    assert m["mj_success_rate"] > 0.0, "the trained policy should succeed in MuJoCo (its own sim)"
    assert m["mean_ang_divergence_rad"] > 0.1, "the trajectories should visibly diverge in angle"


def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_transfer.py first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "the policy transfers PARTLY; the angle diverges most (the rotational gap)."


# --- ex2: the completed contract must match the reference exactly ---
def test_ex2_block_yaw_matches():
    rng = np.random.default_rng(0)
    for _ in range(64):
        bar = rng.uniform(-0.3, 0.3, size=2)
        yaw = rng.uniform(-np.pi, np.pi)
        stem = bar + np.array([ex2.STEM_OFFSET * np.sin(yaw), -ex2.STEM_OFFSET * np.cos(yaw)])
        try:
            got = ex2.block_yaw(bar, stem)
        except NotImplementedError:
            pytest.skip("ex2 block_yaw not filled yet — completing it is the exercise")
        assert got == pytest.approx(ref_block_yaw(bar, stem), abs=CHECKS["ex2"]["reproduce_abs"])
        assert got == pytest.approx(yaw, abs=1e-9)  # and it recovers the yaw it was built from


def test_ex2_obs_assembly_matches():
    rng = np.random.default_rng(1)
    for _ in range(64):
        pusher = rng.uniform(-0.3, 0.3, size=2)
        bar = rng.uniform(-0.3, 0.3, size=2)
        stem = bar + rng.uniform(-0.06, 0.06, size=2)
        try:
            got = ex2.engine_obs(pusher, bar, stem)
        except NotImplementedError:
            pytest.skip("ex2 engine_obs not filled yet — completing it is the exercise")
        assert np.allclose(got, ref_engine_obs(pusher, bar, stem), atol=CHECKS["ex2"]["reproduce_abs"])
        assert got.shape == (10,) and got.dtype == np.float32  # the pusht obs contract


# --- ex3: raising block drag closes the divergence (needs a trained policy) ---
def test_ex3_gap_closes_with_drag():
    policy = ex3.find_policy()
    if policy is None:
        pytest.skip("no trained ch1.1 policy on disk — train one to check the gap knob")
    low = ex3.divergence_at_damp(policy, ex3.LOW_DAMP)
    high = ex3.divergence_at_damp(policy, ex3.HIGH_DAMP)
    floor = CHECKS["ex3"]["divergence_ratio_min"]
    assert low > high, "more block drag should NARROW the sim-to-sim divergence, not widen it"
    assert low / high > floor, f"gap should close by at least {floor}x, got {low / high:.2f}x"


def test_ex3_prediction_recorded():
    if ex3.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex3_gap_knob.py first")
    assert ex3.PREDICTION == ANSWER_KEY["ex3"], "raising the drag toward MuJoCo's quasi-static tee narrows the gap."
