"""SUGGESTED local pytest checks for the ch3.2 (World Models II) exercise candidates.

Run from anywhere:
    pytest curriculum/phase3_advanced/ch3.2_dreamer/exercises/suggested/checks.py

Conventions (match ch3.1):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- Anything that trains the pipeline is @pytest.mark.slow, so `make check` (which
  runs `-m "not gpu and not slow"`) stays fast.

Why these checks don't flake: the whole pipeline — PushT resets, scripted expert,
torch inits, imagination sampling — is seeded, so a FIXED-seed run is bit-
reproducible on CPU. The reliable assertions are the ROBUST orderings (imagination
is rosier than reality; the real policy fails; the delusion is horizon-invariant),
not any single fragile number.
"""

import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ARTIFACT = REPO / "curriculum/phase3_advanced/ch3.2_dreamer/dreamer.py"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import ex1_predict_gap as ex1  # noqa: E402
import ex2_horizon_investigation as ex2  # noqa: E402

# Reference bands live in the chapter meta.yaml with provenance (exercise-spec:
# no bare magic numbers) — read them, don't inline.
CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]


# --- ex1: predict-then-run (the imagination gap) — trains pipeline, seeded --------

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_gap.py first")
    assert isinstance(ex1.PREDICTION, str) and len(ex1.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex1_imagination_is_rosier_and_real_fails(tmp_path):
    # Measured 2026-07-07 (cpu, default config, seeds 0-1), post actor/critic
    # stop-gradient fix: gap +0.205..+0.240, real success 0.00, real tee-dist ~0.16,
    # imagined tee-dist ~0.005..0.011.
    # Robust signals: imagination is rosier than reality (positive gap), the real
    # policy does NOT solve the task, and the dream parks the block while reality
    # does not (imagined tee-dist < real tee-dist). Answer B.
    results = ex1.measure(tmp_path)
    for seed, m in results.items():
        assert m["imagination_gap"] >= CHECKS["ex1"]["gap_min"], (
            f"seed {seed}: imagination should be rosier than reality "
            f"(gap {m['imagination_gap']:+.3f})")
        assert m["real_success_rate"] <= CHECKS["ex1"]["real_success_max"], (
            f"seed {seed}: the imagination-trained policy should NOT solve the real "
            f"task (real success {m['real_success_rate']:.2f})")
        assert m["real_final_tee_dist"] >= CHECKS["ex1"]["real_tee_dist_min"], (
            f"seed {seed}: the real block should barely move from its ~0.17 m spawn "
            f"(real tee-dist {m['real_final_tee_dist']:.3f} m)")
        assert m["imagined_final_tee_dist"] < m["real_final_tee_dist"], (
            f"seed {seed}: the DREAM should park the block closer than reality does "
            f"(imagined {m['imagined_final_tee_dist']:.3f} vs real {m['real_final_tee_dist']:.3f})")


# --- ex2: investigation (imagination horizon vs the gap) --------------------------

def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2_horizon_investigation.py first")
    assert isinstance(ex2.PREDICTION, str) and len(ex2.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b + a reason)"


@pytest.mark.slow
def test_ex2_horizon_does_not_buy_real_skill(tmp_path):
    # Measured 2026-07-07 (cpu, seeds 0-1) AFTER the actor/critic stop-gradient fix:
    # a properly-trained actor parks the DREAM block at EVERY horizon (imagined tee-dist
    # ~0.01-0.02 m at both h=5 and h=30) while REAL stays stuck (~0.16 m, 0% success) at
    # BOTH. The delusion is HORIZON-INVARIANT: rolling a wrong world model further does
    # not buy real skill, because the model is wrong regardless of how far you roll it.
    results = ex2.measure(tmp_path)
    for horizon in (5, 30):
        imag = [m["imagined_final_tee_dist"] for m in results[horizon]]
        real = [m["real_final_tee_dist"] for m in results[horizon]]
        imag_mean, real_mean = sum(imag) / len(imag), sum(real) / len(real)
        assert imag_mean <= CHECKS["ex2"]["imag_dist_max"], (
            f"horizon {horizon}: the dream should park the block "
            f"(imagined tee-dist {imag_mean:.3f} m)")
        assert real_mean >= CHECKS["ex2"]["real_tee_dist_min"], (
            f"horizon {horizon}: reality should stay stuck "
            f"(real tee-dist {real_mean:.3f} m)")
