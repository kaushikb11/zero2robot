"""SUGGESTED local pytest checks for the ch3.4 exercise candidates.

Run from anywhere:  pytest curriculum/phase3_advanced/ch3.4_constraints/exercises/suggested/checks.py

Conventions (mirroring ch3.3):
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
CHAPTER = HERE.parents[1]  # .../ch3.4_constraints/
sys.path.insert(0, str(HERE))

import ex1_predict_drift as ex1  # noqa: E402
import ex2_completion_constraint as ex2  # noqa: E402
import ex3_chaos_determinism as ex3  # noqa: E402

CHECKS = yaml.safe_load((CHAPTER / "meta.yaml").read_text())["exercise_checks"]
SMOKE = CHECKS["smoke_double"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex3": CHECKS["ex3"]["answer"]}  # kept out of the exercise files

PIVOT = np.array([0.0, 0.0, 0.0])


# --- ex2 reference solver (the answer key — kept here, not in the exercise) ---
def ref_constraint_force(q, v, pairs, lengths, minv, f_ext, baumgarte):
    n_particles, n_con = q.shape[0], len(pairs)
    jac = np.zeros((n_con, 3 * n_particles))
    g = np.zeros(n_con)
    gdot = np.zeros(n_con)
    jdotv = np.zeros(n_con)
    for k, (i, j) in enumerate(pairs):
        if j < 0:
            d, dv = q[i] - PIVOT, v[i]
            jac[k, 3 * i:3 * i + 3] = d
        else:
            d, dv = q[i] - q[j], v[i] - v[j]
            jac[k, 3 * i:3 * i + 3] = d
            jac[k, 3 * j:3 * j + 3] = -d
        g[k] = 0.5 * (d @ d - lengths[k] ** 2)
        gdot[k] = d @ dv
        jdotv[k] = dv @ dv
    a_mat = jac @ (minv[:, None] * jac.T)
    b = -(jac @ (minv * f_ext) + jdotv)
    if baumgarte > 0.0:
        b -= 2.0 * baumgarte * gdot + baumgarte ** 2 * g
    lam = np.linalg.solve(a_mat, b)
    return (jac.T @ lam).reshape(q.shape)


def _random_chains(n=16):
    rng = np.random.default_rng(0)
    chains = []
    for _ in range(n):
        n_links = int(rng.integers(1, 4))  # 1..3 links
        q = rng.normal(size=(n_links, 3))
        v = rng.normal(size=(n_links, 3))
        pairs = [(i, i - 1) for i in range(n_links)]
        lengths = rng.uniform(0.5, 1.5, size=n_links)
        minv = np.repeat(rng.uniform(0.5, 2.0, size=n_links), 3)
        f_ext = rng.normal(size=3 * n_links)
        chains.append((q, v, pairs, lengths, minv, f_ext))
    return chains


# --- ex1: constraint drift reproduces + the ordering is the point ---
def test_ex1_drift_signature_reproduces():
    results = ex1.run_results("--smoke")
    tol = SMOKE["reproduce_abs"]
    assert results["none"]["max_violation"] == pytest.approx(SMOKE["none_max_violation"], abs=tol)
    assert results["baumgarte"]["max_violation"] == pytest.approx(SMOKE["baumgarte_max_violation"], abs=tol)
    assert results["baumgarte"]["final_violation"] == pytest.approx(SMOKE["baumgarte_final_violation"], abs=tol)
    # The headline ordering: naive drifts >> Baumgarte holds.
    ratio = results["none"]["max_violation"] / results["baumgarte"]["max_violation"]
    assert ratio > SMOKE["drift_ratio_min"], f"drift ratio {ratio}"


def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_drift.py first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "the naive solve pins gddot=0, not g=0 — so g itself is free to drift."


# --- ex2: completed constraint solve must match the reference exactly ---
def test_ex2_constraint_completion_matches():
    tol = CHECKS["ex2"]["reproduce_abs"]
    for q, v, pairs, lengths, minv, f_ext in _random_chains():
        for baumgarte in (0.0, 20.0):  # exercise both the naive and Baumgarte paths
            try:
                got = ex2.constraint_force(q, v, pairs, lengths, minv, f_ext, baumgarte)
            except NotImplementedError:
                pytest.skip("ex2 blanks not filled yet — completing them is the exercise")
            want = ref_constraint_force(q, v, pairs, lengths, minv, f_ext, baumgarte)
            assert np.allclose(got, want, atol=tol)


# --- ex3: chaos (seeds diverge) coexists with bitwise determinism (same seed) ---
def test_ex3_chaos_and_determinism():
    tip0, tip1, tip0_again = ex3.tip_final(0), ex3.tip_final(1), ex3.tip_final(0)
    divergence = float(np.linalg.norm(tip0 - tip1))
    reproduce = float(np.linalg.norm(tip0 - tip0_again))
    # Nearby seeds diverge: the chaos signature.
    assert divergence > CHECKS["ex3"]["tip_divergence_min"], f"tip divergence {divergence}"
    # Same seed twice is bitwise identical: deterministic despite the chaos.
    assert reproduce == 0.0, f"same-seed runs differ by {reproduce} — determinism invariant violated"


def test_ex3_prediction_recorded():
    if ex3.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex3_chaos_determinism.py first")
    assert ex3.PREDICTION == ANSWER_KEY["ex3"], "deterministic (same seed -> same bytes) is not predictable (nearby seeds diverge)."
