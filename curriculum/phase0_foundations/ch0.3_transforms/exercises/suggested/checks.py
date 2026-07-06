"""SUGGESTED local pytest checks for the ch0.3 exercise candidates.

Run from anywhere:  pytest curriculum/phase0_foundations/ch0.3_transforms/exercises/suggested/checks.py

Conventions (mirroring ch0.1):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- ex2's check SKIPS while the known injected bug is still present (finding and
  fixing it is the exercise) and asserts MuJoCo agreement once it's fixed.
- ex3's check SKIPS while rotate_vector is unfinished, asserts once completed.
- Reference values + seeded bands live in meta.yaml with their provenance
  (exercise-spec: no bare magic numbers) — read them, don't inline.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parents[1]  # .../ch0.3_transforms/
sys.path.insert(0, str(HERE))

import ex1_predict_compose_order as ex1  # noqa: E402
import ex2_bughunt_quat_convention as ex2  # noqa: E402
import ex3_complete_rotate_vector as ex3  # noqa: E402

ANSWER_KEY = {"ex1": "B"}  # kept out of the exercise files to avoid spoilers
CHECKS = yaml.safe_load((CHAPTER / "meta.yaml").read_text())["exercise_checks"]


def test_ex1_compose_order_gap_reproduces():
    rotation_gap, translation_gap = ex1.compose_both_orders()
    band = CHECKS["ex1"]
    # Premise: the two z-yaw rotations commute (identical rotation), but the
    # rigid transforms do NOT — the translation lands ~0.09 m apart.
    assert rotation_gap < CHECKS["mju_agreement_max"], f"rotations should match; gap {rotation_gap}"
    assert abs(translation_gap - band["compose_order_gap"]) < band["compose_order_band"], (
        f"translation gap {translation_gap} did not reproduce {band['compose_order_gap']}")


def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_compose_order.py first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "measure it: which part moved when you swapped the order?"


def test_ex2_fix_restores_mujoco_agreement():
    error = ex2.break_error()
    band = CHECKS["ex2"]
    if abs(error - band["buggy_break_err"]) < band["buggy_band"]:
        pytest.skip("ex2 still has the injected quaternion-order bug — finding and fixing it is the exercise")
    assert error < band["fixed_break_max"], f"still {error} off MuJoCo — is the whole quaternion in [w,x,y,z] order?"


def test_ex3_completed_rotate_matches_mujoco():
    generator = np.random.default_rng(0)
    reference = np.zeros(3)
    max_error = 0.0
    for _ in range(256):
        q = generator.standard_normal(4)
        q = q / np.linalg.norm(q)
        v = generator.standard_normal(3)
        try:
            ours = ex3.rotate_vector(q, v)
        except NotImplementedError:
            pytest.skip("ex3 rotate_vector is not completed yet — that's the exercise")
        mujoco.mju_rotVecQuat(reference, v, q)
        max_error = max(max_error, float(np.max(np.abs(ours - reference))))
    assert max_error < CHECKS["ex3"]["rotate_max_err"], f"rotate_vector is {max_error} off MuJoCo"
