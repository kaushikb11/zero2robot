"""SUGGESTED local pytest checks for the ch4 offline-primer exercise candidates.

Run from anywhere:
    pytest curriculum/phase4_capstone/ch4_offline_primer/exercises/suggested/checks.py

Conventions (match ch2.2):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- Anything that trains a policy is @pytest.mark.slow, so `make check` (which runs
  `-m "not gpu and not slow"`) stays fast — these do not run in the fast lane.

RL-doctrine note (ch2.1 spike, H1/H2): the graded checks assert the STRONG,
seed-robust signal — offline RL beating BC by a margin over N seeds — NOT a
subtle single-run effect. The whole pipeline is seeded (torch + numpy + env
resets), so a fixed-seed run is bit-reproducible on CPU and these checks do not
flake run-to-run; the variance the exercises teach is ACROSS seeds. ex2's
data-quality TREND is left observational (the learner interprets it); only the
per-arm "offline beats BC" claim is gated, because that is the seed-robust part.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import ex1_predict_multiseed as ex1  # noqa: E402
import ex2_investigate_dataset_quality as ex2  # noqa: E402
import ex3_predict_naive_blowup as ex3  # noqa: E402

# Reference bands live in the chapter meta.yaml with provenance (exercise-spec:
# no bare magic numbers) — read them, don't inline.
CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]


# --- ex1: multi-seed predict-then-run (trains BC + offline; seeded) ------------

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_multiseed.py first")
    assert isinstance(ex1.PREDICTION, str) and len(ex1.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex1_offline_beats_bc_across_seeds(tmp_path):
    """The headline, seed-robust: offline RL beats BC on the mixed dataset — a
    clear margin in mean success AND a lower final distance on EVERY seed."""
    r = ex1.measure(tmp_path)
    bc_mean = float(np.mean(r["bc_success"]))
    off_mean = float(np.mean(r["offline_success"]))
    margin = CHECKS["ex1"]["offline_beats_bc_success_margin"]
    assert off_mean - bc_mean >= margin, (
        f"offline RL should beat BC in mean success by >= {margin}: "
        f"offline {off_mean:.3f} vs BC {bc_mean:.3f} over seeds {ex1.SEEDS}")
    dgap = CHECKS["ex1"]["offline_dist_below_bc_per_seed"]
    for i, seed in enumerate(ex1.SEEDS):
        assert r["offline_dist"][i] <= r["bc_dist"][i] - dgap, (
            f"seed {seed}: offline final dist {r['offline_dist'][i]:.4f} m should be "
            f">= {dgap} below BC's {r['bc_dist'][i]:.4f} m")


# --- ex2: dataset-quality investigation (default vs cleaner data) --------------

def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2_investigate_dataset_quality.py first")
    assert isinstance(ex2.PREDICTION, str) and len(ex2.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex2_offline_beats_bc_on_every_quality(tmp_path):
    """Seed-robust assertion: offline RL beats BC on BOTH the mostly-junk and the
    mostly-clean dataset. The TREND (does cleaning close the gap?) is reported for
    the learner to interpret, not gated — BC staying capped across qualities is
    the observation, and a directional trend is not guaranteed seed-robust at this
    budget (the ch2.1-spike honesty)."""
    results = ex2.measure(tmp_path)
    margin = CHECKS["ex2"]["offline_beats_bc_each_arm_margin"]
    for arm, r in results.items():
        bc_mean = float(np.mean(r["bc"]))
        off_mean = float(np.mean(r["offline"]))
        print(f"\n{arm}: BC mean {bc_mean:.3f} vs offline mean {off_mean:.3f} (gap {off_mean - bc_mean:+.3f})")
        assert off_mean - bc_mean >= margin, (
            f"offline should beat BC on the {arm} dataset by >= {margin}: "
            f"offline {off_mean:.3f} vs BC {bc_mean:.3f} over seeds {ex2.SEEDS}")


# --- ex3: naive maximize-Q blows up the critic on narrow data ------------------

def test_ex3_prediction_recorded():
    if ex3.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex3_predict_naive_blowup.py first")
    assert isinstance(ex3.PREDICTION, str) and len(ex3.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"
    assert ex3.SELF_EXPLANATION is not None and len(str(ex3.SELF_EXPLANATION)) > 5, \
        "write SELF_EXPLANATION too — commit WHY maximize-Q breaks before running"


@pytest.mark.slow
def test_ex3_naive_blows_up_on_narrow_not_broad(tmp_path):
    """Seed-robust assertion of the chapter's sharpest failure: naive maximize-Q
    (no data anchor) inflates the critic's mean |Q| on the NARROW expert-only log
    while it stays bounded on the BROAD expert+random mix — coverage is what keeps
    the critic honest. This is the reason offline RL needs the advantage constraint
    (not just twin-Q). The gap is enormous (~9x), so it is robust over seeds."""
    r = ex3.measure(tmp_path)
    q_narrow = float(np.mean(r["narrow"]["abs_q"]))
    q_broad = float(np.mean(r["broad"]["abs_q"]))
    print(f"\nnaive |Q|: narrow {q_narrow:.2f} vs broad {q_broad:.2f} (ratio {q_narrow / q_broad:.1f}x)")
    narrow_min = CHECKS["ex3"]["naive_narrow_abs_q_min"]
    broad_max = CHECKS["ex3"]["naive_broad_abs_q_max"]
    assert q_narrow >= narrow_min, (
        f"naive maximize-Q should inflate |Q| >= {narrow_min} on the narrow "
        f"expert-only log: measured {q_narrow:.2f} over seeds {ex3.SEEDS}")
    assert q_broad <= broad_max, (
        f"on the broad expert+random mix coverage should keep |Q| <= {broad_max}: "
        f"measured {q_broad:.2f} over seeds {ex3.SEEDS}")
