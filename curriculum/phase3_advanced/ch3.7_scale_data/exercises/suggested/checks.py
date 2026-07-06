"""SUGGESTED local pytest checks for the ch3.7 (Datasets at Scale) exercises.

Run from anywhere:
    pytest curriculum/phase3_advanced/ch3.7_scale_data/exercises/suggested/checks.py

Conventions (match ch2.1):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- Anything that trains BC (ex1, ex3) is @pytest.mark.slow, so `make check` (which
  runs `-m "not gpu and not slow"`) stays fast. Those runs ARE deterministic on
  CPU (seeded torch + numpy + env resets), so the assertions are the seed-robust
  ORDERING (augmented > source), never a bare magic number.
- ex2 is DETERMINISTIC — pure zero-pad/mask on toy arrays, zero flake, runs in
  `make check`.

Bands live in the chapter meta.yaml with provenance (exercise-spec: no inline
magic numbers) — read them, don't hardcode.
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

import ex1_predict_scale as ex1  # noqa: E402
import ex2_completion_wrangle as ex2  # noqa: E402
import ex3_hparam_aug_amount as ex3  # noqa: E402

CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]


# --- ex2: code-completion, DETERMINISTIC (no training) -------------------------

def reference_mix(pusht_act, aloha_act, pad_dim=6):
    """The correct zero-pad + action_mask, mirroring scale_data.py's wrangle region."""
    n = len(pusht_act) + len(aloha_act)
    mixed = np.zeros((n, pad_dim), np.float32)
    mask = np.zeros((n, pad_dim), np.float32)
    np_, na = len(pusht_act), pusht_act.shape[1]
    mixed[:np_, :na] = pusht_act
    mask[:np_, :na] = 1.0
    mixed[np_:, :aloha_act.shape[1]] = aloha_act
    mask[np_:, :aloha_act.shape[1]] = 1.0
    return mixed, mask


TOY_PUSHT = np.array([[0.5, -0.2], [0.1, 0.9], [-0.4, 0.0]], np.float32)
TOY_ALOHA = np.array([[1, 0, -1, 0, 0, 1], [0, 1, 1, -1, 0, -1]], np.float32)


def test_ex2_mix_matches_reference():
    try:
        mixed, mask = ex2.mix_embodiments(TOY_PUSHT, TOY_ALOHA)
    except NotImplementedError:
        pytest.skip("mix_embodiments not implemented yet — that's the exercise")
    ref_mixed, ref_mask = reference_mix(TOY_PUSHT, TOY_ALOHA)
    np.testing.assert_allclose(mixed, ref_mixed, atol=0,
                               err_msg="mixed action tensor mismatch — check the pusht/aloha row order and padding")
    np.testing.assert_allclose(mask, ref_mask, atol=0,
                               err_msg="action_mask mismatch — 1.0 on real dims, 0.0 on padding")


def test_ex2_pusht_mask_density():
    """The pedagogical number: a PushT row constrains 2 of 6 dims -> mask density 1/3."""
    try:
        _, mask = ex2.mix_embodiments(TOY_PUSHT, TOY_ALOHA)
    except NotImplementedError:
        pytest.skip("mix_embodiments not implemented yet — that's the exercise")
    density = float(mask[:len(TOY_PUSHT)].mean())
    assert density == pytest.approx(CHECKS["ex2"]["pusht_mask_density"], abs=1e-4), (
        f"pusht rows should use 2 of 6 dims (density 1/3), got {density:.4f} — "
        "did you pad to pad_dim and mask only the real dims?")


# --- ex1: predict-then-run, trains BC twice (seeded => reproducible) -----------

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_scale.py first")
    assert isinstance(ex1.PREDICTION, str) and len(ex1.PREDICTION) > 5, \
        "a prediction should say something falsifiable (A/B/C + a reason)"


@pytest.mark.slow
def test_ex1_augmentation_helps(tmp_path):
    # Measured 2026-07-06 (cpu, seed 0, default config): source 0.02 -> augmented
    # 0.14. The reliable signal is the ORDERING (augmented > source); the seed-robust
    # floor on the gain is meta.yaml ex1.scale_effect_min (min observed +0.08 over
    # seeds 0-2, band set at +0.05).
    source, augmented = ex1.measure(0, tmp_path)
    assert augmented > source, (
        f"augmentation should not hurt: source {source:.2f} vs augmented {augmented:.2f}")
    assert augmented - source >= CHECKS["ex1"]["scale_effect_min"], (
        f"augmentation gain {augmented - source:+.2f} below the seed-robust floor "
        f"{CHECKS['ex1']['scale_effect_min']} — more valid demos should raise success")


# --- ex3: hyperparameter-investigation, trains BC per aug amount ---------------

def test_ex3_prediction_recorded():
    if ex3.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex3_hparam_aug_amount.py first")
    assert isinstance(ex3.PREDICTION, str) and len(ex3.PREDICTION) > 5, \
        "a prediction should say something falsifiable (A/B/C + a reason)"


@pytest.mark.slow
def test_ex3_more_augmentation_helps(tmp_path):
    # aug_per_demo 0 is the source-only baseline (no demo survives to add); 8 is the
    # default. Measured seed 0: 0.02 (none) -> 0.14 (aug 8). The seed-robust claim is
    # monotone improvement, not a magnitude.
    none = ex3.success_at(0, tmp_path)
    full = ex3.success_at(8, tmp_path)
    assert full - none >= CHECKS["ex3"]["monotone_min"], (
        f"aug_per_demo 8 should beat 0: {none:.2f} -> {full:.2f} "
        f"(need +{CHECKS['ex3']['monotone_min']})")
