"""SUGGESTED local pytest checks for the ch3.9 exercise candidates.

Run from anywhere:  pytest curriculum/phase3_advanced/ch3.9_mpc/exercises/suggested/checks.py

Conventions (mirroring ch3.6):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- ex3's completion checks SKIP while a blank still raises NotImplementedError
  (that IS the exercise) and assert reference agreement once the learner fills them in.
- The smoke-signature check runs mpc.py --smoke — HERMETIC (no policy, no dataset,
  no training), so it always runs in CI and is bitwise deterministic (numpy sampling
  + CPU mj_step). ex1/ex2 run the FULL plan (~3 s each): self-contained, no external
  artifact needed, so they run in CI too (unlike ch3.6's trained-policy checks).
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
CHAPTER = HERE.parents[1]  # .../ch3.9_mpc/
ARTIFACT = CHAPTER / "mpc.py"
REPO_ROOT = CHAPTER.parents[2]
sys.path.insert(0, str(HERE))

import ex1_predict_planning as ex1  # noqa: E402
import ex2_predict_break as ex2  # noqa: E402
import ex3_completion_update as ex3  # noqa: E402

CHECKS = yaml.safe_load((CHAPTER / "meta.yaml").read_text())["exercise_checks"]
SMOKE = CHECKS["smoke"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex2": CHECKS["ex2"]["answer"]}  # kept out of the exercise files


# --- reference updates (the answer key for ex3 — kept here, not in the exercise) ---
def ref_cem_update(samples, costs, elite_frac):
    n_elite = max(1, int(elite_frac * samples.shape[0]))
    elite = samples[np.argsort(costs)[:n_elite]]
    return elite.mean(axis=0)


def ref_mppi_update(samples, costs, temperature):
    w = np.exp(-(costs - costs.min()) / temperature)
    w = w / w.sum()
    return (w[:, None] * samples).sum(axis=0)


def run_smoke() -> dict:
    with tempfile.TemporaryDirectory(prefix="z2r-ch39-") as tmp:
        cmd = [sys.executable, str(ARTIFACT), "--smoke", "--seed", "0", "--no-rerun", "--out", tmp]
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO_ROOT)
        return json.loads((Path(tmp) / "metrics.json").read_text())


# --- smoke: the hermetic signature reproduces (always runs in CI) ---
def test_smoke_signature_reproduces():
    m = run_smoke()
    tol = SMOKE["reproduce_abs"]
    assert m["mpc_mean_cost"] == pytest.approx(SMOKE["mpc_mean_cost"], abs=tol)
    assert m["random_mean_cost"] == pytest.approx(SMOKE["random_mean_cost"], abs=tol)
    assert m["mpc_upright_frac"] == pytest.approx(SMOKE["mpc_upright_frac"], abs=tol)


# --- ex1: MPC solves swing-up with zero learning, both methods (self-contained full run) ---
def test_ex1_mpc_solves_swingup():
    floor = CHECKS["ex1"]["upright_min"]
    for method in ("cem", "mppi"):
        m = ex1.run_mpc(method)
        assert m["mpc_upright_frac"] >= floor, f"{method}: MPC should swing up and hold (upright >= {floor})"
        assert m["mpc_mean_cost"] < m["random_mean_cost"], f"{method}: MPC cost should beat the no-plan baseline"
        assert m["random_upright_frac"] == 0.0, "the random baseline should never balance the pole"


def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_planning.py first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "MPC solves swing-up with zero learning (upright -> 1.0), both methods."


# --- ex2: a too-short horizon fails, while the full plan solves (the --break) ---
def test_ex2_break_horizon_fails():
    cap = CHECKS["ex2"]["break_upright_max"]
    floor = CHECKS["ex1"]["upright_min"]
    full = ex2.run_mpc()
    broke = ex2.run_mpc("--break", "horizon")
    assert full["mpc_upright_frac"] >= floor, "the full-horizon plan should solve swing-up"
    assert broke["mpc_upright_frac"] <= cap, "a 3-step horizon should FAIL to bring the pole up"
    assert broke["mpc_upright_frac"] < full["mpc_upright_frac"], "crippling the horizon must degrade the result"


def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2_predict_break.py first")
    assert ex2.PREDICTION == ANSWER_KEY["ex2"], "a too-short horizon FAILS: the pole never comes up (upright -> 0.0)."


# --- ex3: the completed CEM + MPPI updates must match the reference ---
def _fixed_inputs():
    rng = np.random.default_rng(0)
    return rng.uniform(-1, 1, size=(64, 25)), rng.uniform(0, 10, size=64)


def test_ex3_cem_update_matches():
    samples, costs = _fixed_inputs()
    try:
        got = ex3.cem_update(samples, costs, 0.1)
    except NotImplementedError:
        pytest.skip("ex3 cem_update not filled yet — completing it is the exercise")
    assert np.allclose(got, ref_cem_update(samples, costs, 0.1), atol=CHECKS["ex3"]["reproduce_abs"])
    assert got.shape == (25,)


def test_ex3_mppi_update_matches():
    samples, costs = _fixed_inputs()
    try:
        got = ex3.mppi_update(samples, costs, 0.3)
    except NotImplementedError:
        pytest.skip("ex3 mppi_update not filled yet — completing it is the exercise")
    assert np.allclose(got, ref_mppi_update(samples, costs, 0.3), atol=CHECKS["ex3"]["reproduce_abs"])
    assert got.shape == (25,)


def test_ex3_cem_and_mppi_agree_at_the_limit():
    """CEM with n_elite==1 keeps the single best sample; MPPI at temperature -> 0
    collapses to the same one. Filled correctly, the two updates then MATCH."""
    samples, costs = _fixed_inputs()
    try:
        cem = ex3.cem_update(samples, costs, 1.0 / 64)   # elite_frac -> exactly 1 elite
        mppi = ex3.mppi_update(samples, costs, 1e-4)      # near-zero temperature
    except NotImplementedError:
        pytest.skip("ex3 not filled yet — completing it is the exercise")
    best = samples[int(np.argmin(costs))]
    assert np.allclose(cem, best, atol=1e-9), "CEM with one elite is the single lowest-cost sample"
    assert np.allclose(mppi, best, atol=1e-6), "MPPI at temperature -> 0 collapses to the lowest-cost sample"
