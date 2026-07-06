"""SUGGESTED local pytest checks for the ch3.3 exercise candidates.

Run from anywhere:  pytest curriculum/phase3_advanced/ch3.3_engine/exercises/suggested/checks.py

Conventions (mirroring ch0.1):
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
CHAPTER = HERE.parents[1]  # .../ch3.3_engine/
sys.path.insert(0, str(HERE))

import ex1_predict_conservation as ex1  # noqa: E402
import ex2_completion_integrators as ex2  # noqa: E402
import ex3_dt_order_investigation as ex3  # noqa: E402

CHECKS = yaml.safe_load((CHAPTER / "meta.yaml").read_text())["exercise_checks"]
SMOKE = CHECKS["smoke_orbit"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex3": CHECKS["ex3"]["answer"]}  # kept out of the exercise files


# --- ex2 reference steppers (the answer key — kept here, not in the exercise) ---
def ref_semi(q, v, accel, dt):
    a = accel(q, v)
    v_next = v + dt * a
    return q + dt * v_next, v_next


def ref_rk4(q, v, accel, dt):
    def deriv(q, v):
        return v, accel(q, v)

    k1q, k1v = deriv(q, v)
    k2q, k2v = deriv(q + 0.5 * dt * k1q, v + 0.5 * dt * k1v)
    k3q, k3v = deriv(q + 0.5 * dt * k2q, v + 0.5 * dt * k2v)
    k4q, k4v = deriv(q + dt * k3q, v + dt * k3v)
    q_next = q + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
    v_next = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return q_next, v_next


def _accel(q, v):
    return -q  # a unit spring: bounded, nonlinear-free, exercises every k-stage


def _random_states(n=32):
    rng = np.random.default_rng(0)
    return [(rng.normal(size=3), rng.normal(size=3)) for _ in range(n)]


# --- ex1: energy drift reproduces + the ordering is the point ---
def test_ex1_drift_signature_reproduces():
    drift = ex1.run_drift("--smoke")
    tol = SMOKE["reproduce_abs"]
    assert drift["euler"]["rel_final"] == pytest.approx(SMOKE["euler_rel_final"], abs=tol)
    assert drift["semi_implicit"]["rel_final"] == pytest.approx(SMOKE["semi_rel_final"], abs=tol)
    assert abs(drift["rk4"]["rel_max"]) < SMOKE["rk4_negligible"]
    # The headline ordering: Euler runs away >> semi-implicit is bounded > RK4 is ~exact.
    assert abs(drift["euler"]["rel_final"]) > drift["semi_implicit"]["rel_max"] > abs(drift["rk4"]["rel_max"])


def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_conservation.py first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "which update reuses the OLD velocity for position? that one leaks."


# --- ex2: completed steppers must match the reference exactly ---
def test_ex2_semi_implicit_completion_matches():
    tol = CHECKS["ex2"]["reproduce_abs"]
    for q, v in _random_states():
        try:
            got_q, got_v = ex2.semi_implicit_step(q, v, _accel, 0.03)
        except NotImplementedError:
            pytest.skip("ex2 semi-implicit blank not filled yet — completing it is the exercise")
        want_q, want_v = ref_semi(q, v, _accel, 0.03)
        assert np.allclose(got_q, want_q, atol=tol) and np.allclose(got_v, want_v, atol=tol)


def test_ex2_rk4_completion_matches():
    tol = CHECKS["ex2"]["reproduce_abs"]
    for q, v in _random_states():
        try:
            got_q, got_v = ex2.rk4_step(q, v, _accel, 0.03)
        except NotImplementedError:
            pytest.skip("ex2 RK4 blanks not filled yet — completing them is the exercise")
        want_q, want_v = ref_rk4(q, v, _accel, 0.03)
        assert np.allclose(got_q, want_q, atol=tol) and np.allclose(got_v, want_v, atol=tol)


# --- ex3: the integrator ORDERS are visible in the drift-vs-dt ratios ---
def test_ex3_dt_order_reproduces():
    coarse = ex3.drift_at(0.01, 400)
    fine = ex3.drift_at(0.005, 800)
    ex3_cfg = CHECKS["ex3"]
    euler_ratio = coarse["euler"] / fine["euler"]
    # Euler is first order: halving dt ~halves the drift (ratio near 2).
    assert ex3_cfg["euler_ratio_lo"] < euler_ratio < ex3_cfg["euler_ratio_hi"], f"euler ratio {euler_ratio}"
    # RK4 is negligible at both timesteps and gets smaller as dt shrinks (4th order).
    assert coarse["rk4"] < ex3_cfg["rk4_negligible"] and fine["rk4"] < ex3_cfg["rk4_negligible"]
    assert fine["rk4"] <= coarse["rk4"], "RK4 drift should not grow when dt shrinks"


def test_ex3_prediction_recorded():
    if ex3.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex3_dt_order_investigation.py first")
    assert ex3.PREDICTION == ANSWER_KEY["ex3"], "match each shrink factor to the integrator's order (2^p)."
