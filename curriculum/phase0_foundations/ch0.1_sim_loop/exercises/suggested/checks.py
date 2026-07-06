"""SUGGESTED local pytest checks for the ch0.1 exercise candidates.

Run from anywhere:  pytest curriculum/phase0_foundations/ch0.1_sim_loop/exercises/suggested/checks.py

Conventions:
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- ex2's check SKIPS while the known injected bug is still present (that fix is
  the learner's job) and asserts reference agreement once the file changes.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parents[1]  # .../ch0.1_sim_loop/
sys.path.insert(0, str(HERE))

import ex1_predict_timestep as ex1  # noqa: E402
import ex3_friction_investigation as ex3  # noqa: E402

ANSWER_KEY = {"ex1": "C"}  # kept out of the exercise files to avoid spoilers
# Reference signature + seeded bands live in meta.yaml with their provenance
# (exercise-spec: no bare magic numbers) — read them, don't inline.
EX2 = yaml.safe_load((CHAPTER / "meta.yaml").read_text())["exercise_checks"]["ex2"]


def run_metrics(script: Path, out: Path, *extra: str) -> dict:
    cmd = [sys.executable, str(script), "--smoke", "--seed", "0", "--out", str(out), *extra]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=CHAPTER)
    return json.loads((out / "metrics.json").read_text())


def test_ex1_timestep_instability_reproduces():
    calm = ex1.run_flat_push(0.002)
    wild = ex1.run_flat_push(0.05)
    # Premise of the exercise: the default timestep keeps the box on the floor,
    # the 25x timestep launches it during a flat horizontal push.
    assert calm["max_box_z"] < ex1.AIRBORNE_Z, f"baseline unexpectedly airborne: {calm}"
    assert wild["max_box_z"] > ex1.AIRBORNE_Z, f"large timestep did not go airborne: {wild}"


def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_timestep.py first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "measured it yet? the box's z-track has the answer"


def test_ex2_fix_restores_reference_metrics(tmp_path):
    reference = run_metrics(CHAPTER / "sim_loop.py", tmp_path / "ref", "--no-rerun")
    candidate = run_metrics(HERE / "ex2_bughunt_sticky_shove.py", tmp_path / "ex2")
    if abs(candidate["lateral_drift"] - EX2["buggy_lateral_drift"]) < EX2["buggy_band"]:
        pytest.skip("ex2 still has the injected bug — finding and fixing it is the exercise")
    assert candidate["lateral_drift"] == pytest.approx(
        reference["lateral_drift"], abs=EX2["ref_lateral_drift_abs"])
    for got, want in zip(candidate["box_final_pos"], reference["box_final_pos"]):
        assert got == pytest.approx(want, abs=EX2["ref_box_final_pos_abs"])


def test_ex3_friction_ordering_reproduces():
    drift = {mu: ex3.measure_lateral_drift(mu) for mu in (0.4, 1.0, 2.0)}
    # Premise: more friction, less drift — and the spread is large (~100x, not ~2x).
    assert drift[0.4] > drift[1.0] > drift[2.0], f"ordering did not reproduce: {drift}"
    assert drift[2.0] < 0.01 < drift[0.4], f"spread did not reproduce: {drift}"


def test_ex3_prediction_recorded():
    if ex3.PREDICTION is None:
        pytest.skip("PREDICTION not set — write your predicted ordering in ex3_friction_investigation.py first")
    assert isinstance(ex3.PREDICTION, str) and len(ex3.PREDICTION) > 10, "a prediction should say something falsifiable"
