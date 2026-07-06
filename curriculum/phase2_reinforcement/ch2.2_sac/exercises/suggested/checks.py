"""SUGGESTED local pytest checks for the ch2.2 (SAC) exercise candidates.

Run from anywhere:
    pytest curriculum/phase2_reinforcement/ch2.2_sac/exercises/suggested/checks.py

Conventions (match ch2.1):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- Anything that trains SAC is @pytest.mark.slow, so `make check` (which runs
  `-m "not gpu and not slow"`) stays fast.

RL-doctrine note (ch2.1 spike, H1/H2): both graded checks assert the STRONG
learns-signal over N seeds — the eval mean final distance clearly beating the
random baseline (~0.176 m) — NOT a subtle single-run effect. SAC's whole pipeline
is seeded (torch + numpy + env resets), so a FIXED-seed run is bit-reproducible
on CPU and these checks do not flake run-to-run; the variance the exercises teach
is ACROSS seeds. ex2's replay-shrink effect is deliberately left observational
(the learner interprets it), because a directional hyperparameter effect is not
guaranteed seed-robust at the free-tier budget.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ARTIFACT = REPO / "curriculum/phase2_reinforcement/ch2.2_sac/sac.py"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import ex1_predict_multiseed as ex1  # noqa: E402
import ex2_investigate_replay as ex2  # noqa: E402

# Reference bands live in the chapter meta.yaml with provenance (exercise-spec:
# no bare magic numbers) — read them, don't inline.
CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
RANDOM_BASELINE_DIST = 0.176  # m, from the pusher-reach env baselines


# --- ex1: multi-seed predict-then-run (trains SAC; seeded => reproducible) -----

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_multiseed.py first")
    assert isinstance(ex1.PREDICTION, str) and len(ex1.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex1_sac_learns_across_seeds(tmp_path):
    """Strong, seed-robust signal: on every seed SAC drives the eval final
    distance clearly below the random baseline, and the mean is well under it."""
    dists = ex1.measure(tmp_path)["sac"]
    mean_dist = float(np.mean(dists))
    assert mean_dist <= CHECKS["ex1"]["learns_mean_dist_max"], (
        f"SAC should learn to reach: mean eval final distance {mean_dist:.4f} m "
        f"over seeds {ex1.SEEDS} (random ~{RANDOM_BASELINE_DIST} m)")
    assert max(dists) <= CHECKS["ex1"]["learns_per_seed_dist_max"], (
        f"every seed should clearly beat random: {[round(d, 4) for d in dists]}")


# --- ex2: hyperparameter investigation (default vs shrunk replay) --------------

def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2_investigate_replay.py first")
    assert isinstance(ex2.PREDICTION, str) and len(ex2.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex2_default_learns_and_reports_replay_effect(tmp_path):
    """Seed-robust assertion: the DEFAULT buffer reliably learns over seeds. The
    replay-shrink effect is reported for the learner to interpret, not asserted
    per-seed (a directional hyperparameter effect is not guaranteed at this
    budget — that honesty is the ch2.1-spike lesson)."""
    results = ex2.measure(tmp_path)
    default_mean = float(np.mean(results["default_buffer"]))
    small_mean = float(np.mean(results["small_buffer"]))
    assert default_mean <= CHECKS["ex2"]["default_learns_mean_dist_max"], (
        f"default-buffer SAC should learn: mean eval final distance "
        f"{default_mean:.4f} m over seeds {ex2.SEEDS} (random ~{RANDOM_BASELINE_DIST} m)")
    # Observation, not a hard gate: shrinking the replay should not HELP on
    # average (the off-policy bargain weakens or is unchanged, never improves).
    print(f"\nreplay effect: default mean {default_mean:.4f} m vs "
          f"small-buffer mean {small_mean:.4f} m")
    assert small_mean >= default_mean - 0.02, (
        "a 20x-smaller replay should not clearly OUTPERFORM the full buffer "
        f"(default {default_mean:.4f} m vs small {small_mean:.4f} m) — if it does, "
        "re-examine the off-policy-bargain framing")
