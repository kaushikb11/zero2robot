"""SUGGESTED local pytest checks for the ch2.5 (walk) exercise candidates.

Run from anywhere:
    pytest curriculum/phase2_reinforcement/ch2.5_walk/exercises/suggested/checks.py

Conventions (match ch2.1/ch2.2):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- Anything that trains SAC is @pytest.mark.slow, so `make check` (which runs
  `-m "not gpu and not slow"`) stays fast.

RL-doctrine note (ch2.1 spike, H1/H2): both graded checks assert the STRONG
gait-emergence signal over N seeds — the eval mean forward distance clearly
positive, well past the standing baseline (~ -0.01 m) — NOT a subtle single-run
effect, and NOT a claim that the emergent gait matches the +2.14 m scripted trot
(it does not, at the free-tier budget — that honesty is the lesson). walk.py's
whole pipeline is seeded (torch + numpy + env resets + the domain-randomization
draw), so a FIXED-seed run is bit-reproducible on CPU and these checks do not
flake run-to-run; the variance the exercises teach is ACROSS seeds. ex2's
velocity-blinding effect is deliberately left observational (the learner
interprets it), because a directional observation-design effect is not guaranteed
seed-robust at the free-tier budget.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ARTIFACT = REPO / "curriculum/phase2_reinforcement/ch2.5_walk/walk.py"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import ex1_predict_gait as ex1  # noqa: E402
import ex2_investigate_obs as ex2  # noqa: E402

# Reference bands live in the chapter meta.yaml with provenance (exercise-spec:
# no bare magic numbers) — read them, don't inline.
CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
STAND_DIST = -0.01  # m, standing baseline forward travel (from the env baselines)


# --- ex1: multi-seed predict-then-run (trains SAC; seeded => reproducible) -----

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_gait.py first")
    assert isinstance(ex1.PREDICTION, str) and len(ex1.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex1_gait_emerges_across_seeds(tmp_path):
    """Strong, seed-robust signal: on every seed SAC drives the eval forward
    distance clearly positive — a gait emerges — well past the standing baseline."""
    dists = ex1.measure(tmp_path)["walk"]
    mean_dist = float(np.mean(dists))
    assert mean_dist >= CHECKS["ex1"]["emerges_mean_dist_min"], (
        f"a gait should emerge: mean eval forward distance {mean_dist:+.4f} m "
        f"over seeds {ex1.SEEDS} (stand ~{STAND_DIST} m, trot +2.14 m)")
    assert min(dists) >= CHECKS["ex1"]["emerges_per_seed_dist_min"], (
        f"every seed should walk clearly forward: {[round(d, 4) for d in dists]}")


# --- ex2: observation-design investigation (full obs vs velocity-blinded) ------

def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2_investigate_obs.py first")
    assert isinstance(ex2.PREDICTION, str) and len(ex2.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex2_full_obs_walks_and_reports_blinding_effect(tmp_path):
    """Seed-robust assertion: the FULL observation reliably produces a gait over
    seeds. The velocity-blinding effect is reported for the learner to interpret,
    not asserted per-seed (a directional observation-design effect is not
    guaranteed at this budget — the ch2.1-spike honesty lesson)."""
    results = ex2.measure(tmp_path)
    full_mean = float(np.mean(results["full_obs"]))
    blind_mean = float(np.mean(results["blind_velocity"]))
    assert full_mean >= CHECKS["ex2"]["full_obs_emerges_mean_dist_min"], (
        f"full-obs SAC should walk: mean eval forward distance {full_mean:+.4f} m "
        f"over seeds {ex2.SEEDS} (stand ~{STAND_DIST} m)")
    # Observation, not a hard gate: hiding the torso velocity should not clearly
    # HELP the gait (the policy loses information, it does not gain it).
    print(f"\nobs-design effect: full-obs mean {full_mean:+.4f} m vs "
          f"velocity-blinded mean {blind_mean:+.4f} m")
    assert blind_mean <= full_mean + CHECKS["ex2"]["blind_help_tolerance"], (
        f"blinding the torso velocity should not clearly OUTPERFORM full obs "
        f"(full {full_mean:+.4f} m vs blind {blind_mean:+.4f} m) — if it does, "
        "re-examine the observation-design framing")
