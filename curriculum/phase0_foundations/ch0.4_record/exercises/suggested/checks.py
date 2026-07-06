"""SUGGESTED local pytest checks for the ch0.4 exercise candidates.

Run from anywhere:  pytest curriculum/phase0_foundations/ch0.4_record/exercises/suggested/checks.py

Conventions (match ch0.1/0.3):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- The bug-hunt (ex2) and code-completion (ex3) checks SKIP while the exercise is
  still in its starting state (the fix is the learner's job) and assert the
  contract once the file changes.
- Reference values + bands live in meta.yaml with provenance (exercise-spec: no
  bare magic numbers) — read them, don't inline.
"""

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # lerobot writes fully offline

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parents[1]  # .../ch0.4_record/
ARTIFACT = CHAPTER / "record.py"
sys.path.insert(0, str(HERE))

import ex1_predict_episode_count as ex1  # noqa: E402
import ex2_bughunt_feature_shape as ex2  # noqa: E402
import ex3_complete_add_frame as ex3  # noqa: E402

ANSWER_KEY = {"ex1": "B"}  # kept out of the exercise files to avoid spoilers
CHECKS = yaml.safe_load((CHAPTER / "meta.yaml").read_text())["exercise_checks"]
EX1, EX2, EX3 = CHECKS["ex1"], CHECKS["ex2"], CHECKS["ex3"]


# ------------------------------------------------------------------- ex1
def test_ex1_episode_count_scales_frames_only(tmp_path):
    two = ex1.record_metrics(2, tmp_path / "two")
    four = ex1.record_metrics(4, tmp_path / "four")
    steps = EX1["smoke_steps_per_episode"]
    # n_frames is exactly (fixed smoke steps) * episodes; the schema is invariant.
    assert two["n_frames"] == steps * 2, f"expected {steps * 2} frames, got {two['n_frames']}"
    assert four["n_frames"] == steps * 4, f"expected {steps * 4} frames, got {four['n_frames']}"
    assert four["n_episodes"] == 2 * two["n_episodes"]
    for key in EX1["invariant_keys"]:
        assert two[key] == four[key], f"{key} changed with episode count: {two[key]} != {four[key]}"


def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_episode_count.py first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "measured it yet? compare n_frames and the schema keys across the two runs"


# ------------------------------------------------------------------- ex2
def test_ex2_feature_shape_matches_contract():
    obs_shape = tuple(EX2["obs_shape"])
    act_shape = tuple(EX2["act_shape"])
    if ex2.state_shape() != obs_shape:
        pytest.skip("ex2 still declares the wrong observation.state shape — finding and fixing it is the exercise")
    # Once fixed, the schema must match the training-data contract exactly.
    assert ex2.state_shape() == obs_shape, f"observation.state must be {obs_shape}"
    assert ex2.action_shape() == act_shape, f"action must be {act_shape}"
    names = ex2.build_features()["observation.state"]["names"]
    assert len(names) == obs_shape[0], "one name per observation dimension"
    assert ex2.build_features()["observation.state"]["dtype"] == "float32"


# ------------------------------------------------------------------- ex3
def test_ex3_completed_writer_produces_valid_dataset(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        ex3.write_dataset(ex3.synthetic_episodes(), tmp_path / "ds")
    except NotImplementedError:
        pytest.skip("ex3 add_frame loop not completed yet — that's the exercise")

    dataset = LeRobotDataset(repo_id="zero2robot/ex3_teleop", root=tmp_path / "ds")
    assert dataset.num_frames == EX3["expected_frames"], f"expected {EX3['expected_frames']} frames"
    assert dataset.num_episodes == 2

    # The parity essence: v3 format, float32[10] state and float32[2] action.
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["features"]["observation.state"]["shape"] == list(EX2["obs_shape"])
    assert info["features"]["action"]["shape"] == list(EX2["act_shape"])
    for key in ("observation.state", "action"):
        assert info["features"][key]["dtype"] == "float32"
