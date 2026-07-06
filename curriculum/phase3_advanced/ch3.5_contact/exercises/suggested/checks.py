"""SUGGESTED local pytest checks for the ch3.5 exercise candidates.

Run from anywhere:  pytest curriculum/phase3_advanced/ch3.5_contact/exercises/suggested/checks.py

Conventions (mirroring ch3.3/ch3.4):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- ex2's completion checks SKIP while a blank is still unfilled (that IS the
  exercise) and assert reference agreement once the learner fills them in.
- Reference bands + provenance live in meta.yaml exercise_checks (exercise-spec:
  no bare magic numbers) — read them, don't inline. Everything here is pure
  numpy and bitwise deterministic, so the bands are TIGHT.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parents[1]  # .../ch3.5_contact/
sys.path.insert(0, str(HERE))

import ex1_predict_contact as ex1  # noqa: E402
import ex2_completion_contact as ex2  # noqa: E402
import ex3_dt_stability as ex3  # noqa: E402

CHECKS = yaml.safe_load((CHAPTER / "meta.yaml").read_text())["exercise_checks"]
SMOKE = CHECKS["smoke_drop"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex3": CHECKS["ex3"]["answer"]}  # kept out of the exercise files


# --- ex2 reference clamps (the answer key — kept here, not in the exercise) ---
def ref_penalty_normal_force(depth, closing_speed, k, c):
    return max(0.0, k * depth - c * closing_speed)


def ref_project_impulse(lam_accumulated, delta):
    return max(0.0, lam_accumulated + delta)


# --- ex1: contact quality reproduces + the ordering is the point ---
def test_ex1_penetration_signature_reproduces():
    quality = ex1.run_quality("--smoke")
    tol = SMOKE["reproduce_abs"]
    assert quality["penalty"]["max_penetration_frac"] == pytest.approx(SMOKE["penalty_max_penetration_frac"], abs=tol)
    assert quality["penalty"]["rest_penetration_frac"] == pytest.approx(SMOKE["penalty_rest_penetration_frac"], abs=tol)
    assert quality["lcp"]["max_penetration_frac"] == pytest.approx(SMOKE["lcp_max_penetration_frac"], abs=tol)
    assert quality["lcp"]["rest_penetration_frac"] == pytest.approx(SMOKE["lcp_rest_penetration_frac"], abs=tol)
    # The headline ordering: penalty drives much deeper on impact than lcp.
    ratio = quality["penalty"]["max_penetration_frac"] / quality["lcp"]["max_penetration_frac"]
    assert ratio > SMOKE["penetration_ratio_min"], f"penetration ratio {ratio}"
    # And penalty rests INSIDE the table while lcp does not.
    assert quality["penalty"]["rest_penetration_frac"] > quality["lcp"]["rest_penetration_frac"]


def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_contact.py first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "a stiff spring must compress to mg/k to hold a weight up — it rests inside the table."


# --- ex2: completed clamps must match the reference exactly ---
def test_ex2_penalty_clamp_matches():
    rng = np.random.default_rng(0)
    for _ in range(64):
        depth, speed = rng.uniform(0, 0.05), rng.uniform(-5, 5)
        k, c = rng.uniform(1e3, 1e5), rng.uniform(0, 100)
        try:
            got = ex2.penalty_normal_force(depth, speed, k, c)
        except NotImplementedError:
            pytest.skip("ex2 penalty clamp not filled yet — completing it is the exercise")
        assert got == pytest.approx(ref_penalty_normal_force(depth, speed, k, c), abs=CHECKS["ex2"]["reproduce_abs"])
        assert got >= 0.0  # push-only: never negative


def test_ex2_impulse_projection_matches():
    rng = np.random.default_rng(1)
    for _ in range(64):
        lam, delta = rng.uniform(0, 5), rng.uniform(-5, 5)
        try:
            got = ex2.project_impulse(lam, delta)
        except NotImplementedError:
            pytest.skip("ex2 impulse projection not filled yet — completing it is the exercise")
        assert got == pytest.approx(ref_project_impulse(lam, delta), abs=CHECKS["ex2"]["reproduce_abs"])
        assert got >= 0.0  # push-only impulse


# --- ex3: the dt-stability cliff — penalty explodes past dt_crit, lcp holds ---
def test_ex3_stability_cliff():
    safe = ex3.energy_excess_at_dt(0.002)   # under dt_crit
    over = ex3.energy_excess_at_dt(0.03)    # past dt_crit ~ 0.02
    floor = CHECKS["ex3"]["penalty_blowup_factor_min"]
    # Under dt_crit both are well-behaved (no phantom energy above the drop energy).
    assert safe["penalty"] < 1.0 and safe["lcp"] < 1.0
    # Past dt_crit the penalty spring pumps energy without bound; lcp stays bounded.
    assert over["penalty"] > floor, f"penalty energy_excess past dt_crit only {over['penalty']}"
    assert over["lcp"] < 1.0, f"lcp should stay bounded past dt_crit, got {over['lcp']}"


def test_ex3_prediction_recorded():
    if ex3.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex3_dt_stability.py first")
    assert ex3.PREDICTION == ANSWER_KEY["ex3"], "the cliff is the stiff spring's (penalty); the projection-based lcp has no spring to blow up."
