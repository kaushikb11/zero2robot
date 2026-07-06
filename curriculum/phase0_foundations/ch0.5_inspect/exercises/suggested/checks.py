"""SUGGESTED local pytest checks for the ch0.5 exercise candidates.

Run from anywhere:  pytest curriculum/phase0_foundations/ch0.5_inspect/exercises/suggested/checks.py

Conventions (match ch0.1/0.4):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- The bug-hunt (ex2) and code-completion (ex3) checks SKIP while the exercise is
  still in its starting state (the fix is the learner's job) and assert the
  contract once the file changes.
- Reference values + bands live in meta.yaml with provenance (exercise-spec: no
  bare magic numbers) — read them, don't inline.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # lerobot reads/writes fully offline

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parents[1]  # .../ch0.5_inspect/
sys.path.insert(0, str(HERE))

import ex1_predict_yaw_swap as ex1  # noqa: E402
import ex2_bughunt_target_index as ex2  # noqa: E402
import ex3_complete_episode_reached as ex3  # noqa: E402

ANSWER_KEY = {"ex1": "B"}  # kept out of the exercise files to avoid spoilers
CHECKS = yaml.safe_load((CHAPTER / "meta.yaml").read_text())["exercise_checks"]
EX1, EX2, EX3 = CHECKS["ex1"], CHECKS["ex2"], CHECKS["ex3"]


# ------------------------------------------------------------------- ex1
def test_ex1_yaw_swap_collapses_success_only(tmp_path):
    good = ex1.inspect_metrics("none", tmp_path / "none")
    bad = ex1.inspect_metrics("yaw-swap", tmp_path / "yaw")
    # A reading bug changes the READING, never the data: schema + structure hold.
    for key in EX1["invariant_keys"]:
        assert good[key] == bad[key], f"{key} changed with the yaw decode: {good[key]} != {bad[key]}"
    assert good["n_episodes"] == EX1["n_episodes"]
    # Correct decode: the stand-in demos end on target. Swapped: success collapses.
    assert good["success_rate"] == EX1["success_rate_none"], "expected all stand-in demos to read reached with a correct yaw decode"
    assert bad["success_rate"] == EX1["success_rate_yaw_swap"], "atan2(cos, sin) should reflect every angle -> nothing reads as reached"


def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_yaw_swap.py first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "measured it yet? compare success_rate vs the schema/structure keys across the two runs"


# ------------------------------------------------------------------- ex2
def test_ex2_frame_errors_measures_against_the_target():
    on_state = np.asarray(EX2["on_target_state"], dtype=np.float32)
    off_state = np.asarray(EX2["off_target_state"], dtype=np.float32)
    on_pos, _ = ex2.frame_errors(on_state)
    if abs(on_pos - EX2["on_target_pos_err"]) > EX2["pos_err_band"]:
        pytest.skip("ex2 still measures distance to the wrong point — finding and fixing the index is the exercise")
    # Once fixed, the reading matches the contract: on-target reaches, adrift doesn't.
    off_pos, _ = ex2.frame_errors(off_state)
    assert abs(off_pos - EX2["off_target_pos_err"]) < EX2["pos_err_band"], "block 0.20 m from target must read pos_err ~0.20"
    assert [ex2.reached(on_state), ex2.reached(off_state)] == EX2["expected_reached"]


# ------------------------------------------------------------------- ex3
def test_ex3_episode_reached_reads_the_terminal_frame():
    try:
        got = [bool(ex3.episode_reached(ex3.ENDS_ON_TARGET)),
               bool(ex3.episode_reached(ex3.ENDS_ADRIFT))]
    except NotImplementedError:
        pytest.skip("ex3 episode_reached not completed yet — that's the exercise")
    assert got == EX3["reached_trajectory"], "a demo that ENDS on target reads reached; one that ends adrift does not"
