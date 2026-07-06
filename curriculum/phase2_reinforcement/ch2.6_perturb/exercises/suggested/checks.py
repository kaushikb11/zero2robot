"""SUGGESTED local pytest checks for the ch2.6 (perturb) exercise candidates.

Run from anywhere:
    pytest curriculum/phase2_reinforcement/ch2.6_perturb/exercises/suggested/checks.py

Conventions (match ch2.1 / ch2.2):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- Anything that trains PPO is @pytest.mark.slow, so `make check` (which runs
  `-m "not gpu and not slow"`) stays fast.

RL-doctrine note (ch2.1 spike, H1/H2): perturbed eval is noisy, so BOTH graded
checks assert a seed-robust STRUCTURAL fact, never a single dramatic run:
  - ex1: the SAME perturbation (latency) is the worst on every seed, the clean
    baseline holds, and the extreme-latency point is broken. WHICH perturbation
    wins is the seed-robust claim; the exact success numbers off the cliff are
    left for the learner to read.
  - ex2: the latency curve has the shape solid -> knee -> dead — success is high
    at small delay and ~0 at large delay on every seed. The exact breaking step
    is deliberately left observational (it wobbles seed to seed), exactly the
    honesty the ch2.1 spike demands of a noisy RL metric.
The whole pipeline (PPO train + perturbed eval + the noise RNG) is seeded, so a
fixed-seed run is bit-reproducible on CPU and these checks do not flake
run-to-run; the variance the exercises teach is across PERTURBATION MAGNITUDE.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ARTIFACT = REPO / "curriculum/phase2_reinforcement/ch2.6_perturb/perturb.py"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import ex1_predict_worst as ex1  # noqa: E402
import ex2_investigate_latency as ex2  # noqa: E402

# Reference bands live in the chapter meta.yaml with provenance (exercise-spec:
# no bare magic numbers) — read them, don't inline.
CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]


# --- ex1: predict-then-run — which perturbation hurts most (trains PPO) --------

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_worst.py first")
    assert isinstance(ex1.PREDICTION, str) and len(ex1.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex1_latency_is_worst_across_seeds(tmp_path):
    """Seed-robust structural claim: on every seed the clean-trained balancer
    solves clean, latency is the worst perturbation, and 8-step latency breaks
    it. (Noise and gravity-mismatch it tolerates — that contrast is the lesson.)"""
    r = ex1.measure(tmp_path)
    assert all(w == CHECKS["ex1"]["worst_perturbation"] for w in r["worst"]), (
        f"latency should be the worst perturbation on every seed, got {r['worst']}")
    assert min(r["baseline_success"]) >= CHECKS["ex1"]["baseline_success_min"], (
        f"the clean policy should solve on every seed: {r['baseline_success']}")
    assert max(r["latency_extreme_success"]) <= CHECKS["ex1"]["latency_extreme_success_max"], (
        f"extreme latency should break it on every seed: {r['latency_extreme_success']}")


# --- ex2: hyperparameter investigation — the latency cliff (trains PPO) --------

def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2_investigate_latency.py first")
    assert isinstance(ex2.PREDICTION, str) and len(ex2.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex2_latency_curve_has_a_cliff(tmp_path):
    """Seed-robust shape: success is high at small delay and ~0 at large delay on
    every seed. The exact breaking step is left observational (it moves seed to
    seed) — only the solid->dead SHAPE is asserted."""
    r = ex2.measure(tmp_path)
    lats = r["latencies"]
    curves = np.array(r["success_by_seed"])  # (seed, latency)
    lo_idx = lats.index(min(lats))           # 0-step latency (clean)
    hi_idx = lats.index(max(lats))           # 8-step latency (broken)
    assert curves[:, lo_idx].min() >= CHECKS["ex2"]["lowlat_success_min"], (
        f"small latency should keep the policy solving: {curves[:, lo_idx].tolist()}")
    assert curves[:, hi_idx].max() <= CHECKS["ex2"]["highlat_success_max"], (
        f"large latency should break it on every seed: {curves[:, hi_idx].tolist()}")
    # And it must be a real DROP, not flat noise: every seed loses at least half.
    print(f"\nlatency curves (seed x {lats}):\n{curves}")
    assert (curves[:, lo_idx] - curves[:, hi_idx]).min() >= 0.5, (
        "each seed's success should collapse from clean to max-latency")
