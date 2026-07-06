"""SUGGESTED local pytest checks for the ch0.2 exercise candidates.

Run from anywhere:  pytest curriculum/phase0_foundations/ch0.2_scene/exercises/suggested/checks.py

Conventions (mirrors ch0.1):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- ex2's check SKIPS while the injected bug is still present (that fix is the
  learner's job) and asserts reference agreement once the file changes.
- Reference values + seeded bands live in meta.yaml with their provenance
  (exercise-spec: no bare magic numbers) — read them, don't inline.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parents[1]  # .../ch0.2_scene/
sys.path.insert(0, str(HERE))

import ex1_predict_dof as ex1  # noqa: E402
import ex3_gain_investigation as ex3  # noqa: E402

CHECKS = yaml.safe_load((CHAPTER / "meta.yaml").read_text())["exercise_checks"]


def run_metrics(script: Path, out: Path, *extra: str) -> dict:
    cmd = [sys.executable, str(script), "--smoke", "--seed", "0", "--out", str(out), *extra]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=CHAPTER)
    return json.loads((out / "metrics.json").read_text())


def test_ex1_dof_counts_and_height_reproduce():
    planar = ex1.run(ex1.PLANAR_JOINT)
    free = ex1.run(ex1.FREE_JOINT)
    c = CHECKS["ex1"]
    # Premise: the joint TYPE sets the DOF count, and the DOF count decides
    # whether the block can leave the table plane.
    assert planar["nq"] == c["planar_nq"], f"planar nq changed: {planar}"
    assert free["nq"] == c["free_nq"], f"freejoint nq changed: {free}"
    assert planar["max_stem_z"] == pytest.approx(c["planar_max_stem_z"], abs=1e-3), "planar block should stay pinned flat"
    assert free["max_stem_z"] > c["free_max_stem_z_min"], f"freejoint block did not rise off the plane: {free}"


def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_dof.py first")
    assert ex1.PREDICTION == CHECKS["ex1"]["answer"], "measured it yet? the block's z-track has the answer"


def test_ex2_fix_restores_reference_metrics(tmp_path):
    c = CHECKS["ex2"]
    reference = run_metrics(CHAPTER / "scene.py", tmp_path / "ref", "--no-rerun")
    candidate = run_metrics(HERE / "ex2_bughunt_pusher_axis.py", tmp_path / "ex2")
    if abs(candidate["tee_final_pose"][1] - c["buggy_tee_final_y"]) < c["buggy_band"]:
        pytest.skip("ex2 still has the injected bug — finding and fixing it is the exercise")
    for got, want in zip(candidate["tee_final_pose"], reference["tee_final_pose"]):
        assert got == pytest.approx(want, abs=c["ref_tee_final_pose_abs"])


def test_ex3_gain_ordering_reproduces():
    travel = {kv: ex3.measure_travel(kv) for kv in (5, 20, 80)}
    ref = CHECKS["ex3"]["travel_by_kv"]
    abs_tol = CHECKS["ex3"]["travel_abs"]
    # Premise: more gain, farther push — monotonically, and the spread is real.
    assert travel[5] < travel[20] < travel[80], f"ordering did not reproduce: {travel}"
    for kv in (5, 20, 80):
        assert travel[kv] == pytest.approx(ref[kv], abs=abs_tol), f"kv={kv} drifted: {travel[kv]} vs {ref[kv]}"


def test_ex3_prediction_recorded():
    if ex3.PREDICTION is None:
        pytest.skip("PREDICTION not set — write your predicted ordering in ex3_gain_investigation.py first")
    assert isinstance(ex3.PREDICTION, str) and len(ex3.PREDICTION) > 10, "a prediction should say something falsifiable"
