"""SUGGESTED local pytest checks for the ch3.1 (World Models I) exercise candidates.

Run from anywhere:
    pytest curriculum/phase3_advanced/ch3.1_world_models/exercises/suggested/checks.py

Conventions (match ch2.1):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- Anything that trains a world model is @pytest.mark.slow, so `make check` (which
  runs `-m "not gpu and not slow"`) stays fast.

Why these checks don't flake: the whole pipeline — PushT resets, scripted expert,
torch inits, batch order — is seeded, so a FIXED-seed run is bit-reproducible on
CPU. The reliable assertions are the ROBUST orderings (WM beats copy-last on
average; a starved latent does not), not any single fragile number.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ARTIFACT = REPO / "curriculum/phase3_advanced/ch3.1_world_models/wm.py"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import ex1_predict_crossover as ex1  # noqa: E402
import ex2_latent_dim_investigation as ex2  # noqa: E402

# Reference bands live in the chapter meta.yaml with provenance (exercise-spec:
# no bare magic numbers) — read them, don't inline.
CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]


# --- ex1: predict-then-run (crossover) — trains WM, seeded => reproducible --------

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_crossover.py first")
    assert isinstance(ex1.PREDICTION, str) and len(ex1.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex1_world_model_beats_copy_last_and_crosses_over(tmp_path):
    # Measured 2026-07-06 (cpu, default config, seeds 0-2): ratio 2.18-2.45,
    # crossover k 2-3. The robust signals: WM beats copy-last on average, and the
    # crossover happens within a few steps (answer B — loses at k=1, overtakes later).
    results = ex1.measure(tmp_path)
    for seed, m in results.items():
        assert m["pred_ratio_copy_over_wm"] >= CHECKS["ex1"]["pred_ratio_min"], (
            f"seed {seed}: world model should beat copy-last on average "
            f"(ratio {m['pred_ratio_copy_over_wm']:.2f})")
        assert 1 <= m["crossover_k"] <= CHECKS["ex1"]["crossover_k_max"], (
            f"seed {seed}: world model should overtake copy-last within a few steps "
            f"(crossover_k {m['crossover_k']}) — it should LOSE at k=1 but win soon after")


# --- ex2: investigation (latent capacity vs prediction) ---------------------------

def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2_latent_dim_investigation.py first")
    assert isinstance(ex2.PREDICTION, str) and len(ex2.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b + a reason)"


@pytest.mark.slow
def test_ex2_latent_capacity_buys_prediction(tmp_path):
    # Measured 2026-07-06 (cpu, seeds 0-1): latent_dim=2 ratio ~0.3 (loses to
    # copy-last), latent_dim=16 ratio ~2.2-2.4. Starving the latent collapses
    # prediction; the ordering (full > tiny) is the robust, seed-stable signal.
    results = ex2.measure(tmp_path)
    tiny = float(np.mean(results[2]))
    full = float(np.mean(results[16]))
    assert tiny <= CHECKS["ex2"]["tiny_latent_ratio_max"], \
        f"a starved latent (dim 2) should NOT reliably beat copy-last (mean ratio {tiny:.2f})"
    assert full >= CHECKS["ex2"]["full_latent_ratio_min"], \
        f"the default latent (dim 16) should clearly beat copy-last (mean ratio {full:.2f})"
    assert full > tiny, f"more latent capacity should help prediction: dim16 {full:.2f} vs dim2 {tiny:.2f}"
