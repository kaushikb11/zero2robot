"""SUGGESTED local pytest checks for the ch4.3 HIL-SERL exercise candidates.

Run from anywhere:
    pytest curriculum/phase4_capstone/ch4.3_serl/exercises/suggested/checks.py

Conventions (match ch2.2 / the primer):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- Anything that trains a policy is @pytest.mark.slow, so `make check` (which runs
  `-m "not gpu and not slow"`) stays fast — these do not run in the fast lane.

RL-doctrine note (ch2.1 spike, H1/H2): the graded checks assert the STRONG,
seed-robust signal — HIL-SERL reaching the threshold in FEWER online samples than
from-scratch — NOT a subtle single-run effect. The whole pipeline is seeded
(torch + numpy + env resets), so a fixed-seed run is bit-reproducible on CPU and
these checks do not flake run-to-run; the variance the exercises teach is ACROSS
seeds. ex2's correction-amount TREND is left observational (the learner reads it);
only "the default corrections build a threshold-clearing prior" is gated.
"""

import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import ex1_predict_sample_efficiency as ex1  # noqa: E402
import ex2_investigate_corrections as ex2  # noqa: E402

# Reference bands live in the chapter meta.yaml with provenance (exercise-spec:
# no bare magic numbers) — read them, don't inline.
CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]


# --- ex1: sample-efficiency predict-then-run (runs both arms; seeded) ----------

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_sample_efficiency.py first")
    assert isinstance(ex1.PREDICTION, str) and len(ex1.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex1_hil_more_sample_efficient_every_seed(tmp_path):
    """The headline, seed-robust: on EVERY seed, HIL-SERL clears the threshold in
    at most `hil_sts_max` online samples, and in STRICTLY fewer than from-scratch
    (which either needs more samples or never reaches it in its budget)."""
    r = ex1.measure(tmp_path)
    hil_max = CHECKS["ex1"]["hil_sts_max"]
    for i, seed in enumerate(ex1.SEEDS):
        hil, scr = r["hil_sts"][i], r["scratch_sts"][i]
        assert hil is not None and hil <= hil_max, (
            f"seed {seed}: HIL-SERL should clear the threshold within {hil_max} online "
            f"samples (the prior's head start); got {hil}")
        assert scr is None or scr > hil, (
            f"seed {seed}: from-scratch should need MORE online samples than HIL-SERL "
            f"({hil}); got scratch={scr}")


# --- ex2: correction-amount investigation (starved vs default) -----------------

def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2_investigate_corrections.py first")
    assert isinstance(ex2.PREDICTION, str) and len(ex2.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex2_default_corrections_build_threshold_clearing_prior(tmp_path):
    """Seed-robust gated claim: with the DEFAULT corrections, the offline prior
    clears the threshold (mean eval distance below `default_prior_dist_max`) on
    every seed — the corrections-as-prior are what buy the sample efficiency. The
    starvation TREND is printed for the learner to interpret, not gated."""
    res = ex2.measure(tmp_path)
    dmax = CHECKS["ex2"]["default_prior_dist_max"]
    for i, seed in enumerate(ex2.SEEDS):
        d = res["default"]["prior_dist"][i]
        assert d <= dmax, (
            f"seed {seed}: the default-corrections prior should clear the threshold "
            f"(eval dist <= {dmax} m); got {d:.4f} m")
    for arm in ("starved", "default"):
        print(f"\n{arm}: prior_dist {res[arm]['prior_dist']}  HIL sts {res[arm]['hil_sts']}")
