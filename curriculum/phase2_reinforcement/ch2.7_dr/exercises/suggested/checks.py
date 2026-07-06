"""SUGGESTED local pytest checks for the ch2.7 (domain randomization) exercises.

Run from anywhere:
    pytest curriculum/phase2_reinforcement/ch2.7_dr/exercises/suggested/checks.py

Conventions (match ch2.1 / ch2.2 / ch2.6):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- Anything that trains PPO is @pytest.mark.slow, so `make check` (which runs
  `-m "not gpu and not slow"`) stays fast.

RL-doctrine note (ch2.1 spike, H1/H2): RL eval is noisy, so BOTH graded checks
assert a seed-robust STRUCTURAL fact, never a single dramatic run. Crucially, the
facts asserted here are PHYSICS, not tuning:
  - ex1: on every seed both policies survive at nominal and both collapse in the
    deep (actuator-saturating) gap — DR cannot rescue a gap the +-12 Nm servos
    cannot hold. The exact near-nominal edge (whether DR shaves a little variance)
    is left observational; it moves with the training budget.
  - ex2: widening --dr_width does not lift deep-gap survival off the floor on any
    width/seed — the ceiling is the robot, not the randomization. The nominal
    premium a very wide band pays is left for the learner to read.
The whole pipeline (PPO train + DR draws + eval) is seeded, so a fixed-seed run is
bit-reproducible on CPU and these checks do not flake run-to-run; the variance the
exercises teach is across DYNAMICS and BAND WIDTH.
"""

import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ARTIFACT = REPO / "curriculum/phase2_reinforcement/ch2.7_dr/dr.py"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import ex1_predict_generalization as ex1  # noqa: E402
import ex2_investigate_dr_width as ex2  # noqa: E402

# Reference bands live in the chapter meta.yaml with provenance (exercise-spec:
# no bare magic numbers) — read them, don't inline.
CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]


# --- ex1: predict-then-run — does DR hold across the gap (trains PPO x2) --------

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_generalization.py first")
    assert isinstance(ex1.PREDICTION, str) and len(ex1.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex1_stable_at_nominal_and_both_fall_in_the_gap():
    """Seed-robust structural claim: on every seed BOTH the narrow and randomized
    policies survive at nominal, and BOTH collapse in the deep gap. DR does not
    rescue the actuator-limited gap here — that boundary is the lesson."""
    r = ex1.measure()
    nom_min = CHECKS["ex1"]["nominal_survival_min"]
    deep_max = CHECKS["ex1"]["deepgap_survival_max"]
    assert min(r["narrow_nominal"]) >= nom_min, f"narrow should stand at nominal every seed: {r['narrow_nominal']}"
    assert min(r["randomized_nominal"]) >= nom_min, f"randomized should stand at nominal every seed: {r['randomized_nominal']}"
    assert max(r["narrow_deepgap"]) <= deep_max, f"narrow should fall in the deep gap every seed: {r['narrow_deepgap']}"
    assert max(r["randomized_deepgap"]) <= deep_max, (
        f"randomized should ALSO fall in the deep gap — DR can't cross the actuator "
        f"ceiling: {r['randomized_deepgap']}")


# --- ex2: hyperparameter investigation — band width vs the ceiling (trains PPO) -

def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2_investigate_dr_width.py first")
    assert isinstance(ex2.PREDICTION, str) and len(ex2.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex2_wider_band_does_not_cross_the_ceiling():
    """Seed-robust shape: across EVERY randomization width, deep-gap survival stays
    on the floor — a wider band cannot buy survival the servos cannot deliver. The
    nominal premium a very wide band pays is left observational (it wobbles)."""
    r = ex2.measure()
    deep_max = CHECKS["ex2"]["deepgap_survival_max"]
    worst = {w: max(r[w]["deepgap_survival"]) for w in ex2.WIDTHS}
    print(f"\ndeep-gap survival (max over seeds) per width: {worst}")
    for width, hi in worst.items():
        assert hi <= deep_max, (
            f"dr_width {width}: deep-gap survival {hi} exceeds the ceiling band — "
            f"if a wider band really crossed the actuator wall, revisit the finding")
