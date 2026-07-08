"""SUGGESTED local pytest checks for the ch4.2 (DAgger) exercise candidates.

Run from anywhere:
    pytest curriculum/phase4_capstone/ch4.2_corrections/exercises/suggested/checks.py

Conventions (match ch1.1 / ch2.1):
- Prediction gates (PREDICTION / FINDINGS unset) SKIP rather than fail: the site
  enforces predict-before-run; locally we only verify the measurements reproduce.
- Anything that trains a DAgger loop is @pytest.mark.slow, so `make check` (which
  runs `-m "not gpu and not slow"`) stays fast.

Why these checks do not flake: dagger.py is bit-reproducible on CPU at a fixed
seed (env resets, torch, numpy, and the beta-mix RNG are all seeded), so every
run at seed 0 produces the SAME metrics.json. The recovery's *significance* is a
seeded-band claim measured across seeds 0-2 (meta.yaml reference_run); the checks
here assert the seed-0 point estimates, which reproduce exactly.
"""

import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import ex1_predict_recovery as ex1  # noqa: E402
import ex2_investigate_rounds as ex2  # noqa: E402

# Reference bands live in the chapter meta.yaml with provenance (exercise-spec:
# no bare magic numbers) — read them, don't inline.
CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]


# --- ex1: predict-then-run — DAgger recovers BC --------------------------------

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_recovery.py first")
    assert isinstance(ex1.PREDICTION, str) and len(ex1.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex1_dagger_recovers_bc(tmp_path):
    # Measured 2026-07-07 (cpu, seed 0, default config): BC 0.065, best DAgger3
    # 0.215, diff CI [+0.083, +0.217] excludes 0. Seeds 0-2 all recover.
    m = ex1.measure(tmp_path)
    assert m["bc_rate"] <= CHECKS["ex1"]["bc_rate_max"], \
        f"BC should covariate-shift to a low success rate, got {m['bc_rate']}"
    assert m["best_rate"] >= CHECKS["ex1"]["best_rate_min"], \
        f"the best DAgger round should recover well above BC, got {m['best_rate']}"
    assert m["recovery_significant"], \
        f"the recovery diff CI should exclude 0 at N={m['n_pooled']}: {m['recovery_diff_ci_lo']}..{m['recovery_diff_ci_hi']}"
    assert m["best_rate"] > m["bc_rate"], "best round must beat BC"


# --- ex2: investigation — the peak round is not the last -----------------------

def test_ex2_findings_recorded():
    if ex2.FINDINGS is None:
        pytest.skip("FINDINGS not set — record what you saw in ex2_investigate_rounds.py first")
    assert isinstance(ex2.FINDINGS, str) and len(ex2.FINDINGS) > 5, \
        "record which round peaked and whether the curve was monotonic"


@pytest.mark.slow
def test_ex2_peak_is_not_the_last_round(tmp_path):
    # Measured 2026-07-07 (cpu, seed 0): round_rates [0.065,0.085,0.155,0.215,0.085].
    # The peak (round 3, 0.215) clearly beats the LAST round (round 4, 0.085) —
    # over-iterating regressed. That is why you select the best round, not the last.
    m = ex2.measure(tmp_path)
    rates = m["round_rates"]
    peak = max(rates[1:])  # best DAgger round
    assert peak - rates[-1] >= CHECKS["ex2"]["peak_beats_last_min"], \
        f"at seed 0 the peak should beat the last round (regression): {rates}"
    assert peak == m["best_rate"], "best_rate should equal the peak DAgger round"
