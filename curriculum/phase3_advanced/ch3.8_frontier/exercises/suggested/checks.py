"""SUGGESTED local pytest checks for the ch3.8 exercise candidates.

Run from anywhere:
    pytest curriculum/phase3_advanced/ch3.8_frontier/exercises/suggested/checks.py

Conventions (match ch1.8 / ch1.9):
- Prediction / reading gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the recorded choice.
- The two self-contained gates (ex2 probe-leakage bug-hunt, ex3 linear-probe completion)
  are FAST + deterministic — numpy only, no torch, no checkpoint.
- ex2 SKIPS while the leaking (buggy) probe is still present; asserts the fix drives a
  random, signal-free layer's R^2 back near 0.
- ex3 SKIPS while `linear_probe_r2` raises NotImplementedError.
- The reproduce check (ex1: trained coord R^2 >> random) runs probe.py at the default
  config on cpu and is @pytest.mark.slow — excluded from a fast run. CPU training is
  deterministic, so seed 0 reproduces.
- Reference bands live in meta.yaml with provenance (exercise-spec: no bare magic
  numbers) — read them, don't inline.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PROBE = REPO / "curriculum/phase3_advanced/ch3.8_frontier/probe.py"
sys.path.insert(0, str(HERE))

import ex1_predict_probe as ex1  # noqa: E402
import ex2_bughunt_probe_leakage as ex2  # noqa: E402
import ex3_completion_linear_probe as ex3  # noqa: E402
import ex4_read_dual_system as ex4  # noqa: E402

CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
RC = CHECKS["exercise_config"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex4": CHECKS["ex4"]["answer"]}


def run_probe(out: Path) -> dict:
    cmd = [sys.executable, str(PROBE), "--seed", "0", "--device", "cpu", "--no-rerun", "--out", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


# ----------------------------------------------------------------- predictions
def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1 first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "run ex1: only the routed-coord R^2 separates trained from random"


def test_ex4_reading_recorded():
    if ex4.PREDICTION is None:
        pytest.skip("PREDICTION not set — do the ex4 guided reading and record your choice")
    assert ex4.PREDICTION == ANSWER_KEY["ex4"], "re-read: System 2 (VLM) conditions System 1 (fast action head)"


# ------------------------------------------ ex2 probe-leakage bug-hunt (fast)
# Random, signal-free features vs an independent target: a leakage-free held-out probe
# must score near 0. The buggy version (scoring on the train rows it fit) scores high.
def _random_fixture():
    rng = np.random.default_rng(0)
    return rng.standard_normal((200, 32)), rng.standard_normal(200)


def test_ex2_no_leakage_on_random_features():
    feats, target = _random_fixture()
    r2 = ex2.probe_r2(feats, target)
    if r2 >= 0.2:
        pytest.skip("still scoring on the TRAIN rows (leakage inflates a signal-free layer) — fix it to score held-out")
    assert r2 < 0.2, f"a random, signal-free layer must probe near 0 on held-out rows, got R^2={r2:.3f}"


# ------------------------------------------- ex3 linear-probe completion (fast)
def _ref_probe_r2(feats, target, ridge):
    n, cut = len(feats), len(feats) // 2
    x = np.concatenate([feats, np.ones((n, 1), np.float64)], axis=1)
    xtr, ytr = x[:cut], target[:cut].astype(np.float64)
    w = np.linalg.solve(xtr.T @ xtr + ridge * np.eye(x.shape[1]), xtr.T @ ytr)
    pred = x[cut:] @ w
    yte = target[cut:].astype(np.float64)
    resid = ((yte - pred) ** 2).sum()
    total = ((yte - yte.mean()) ** 2).sum()
    return float(1.0 - resid / total) if total > 0 else 0.0


def _signal_fixture():
    rng = np.random.default_rng(1)
    feats = rng.standard_normal((200, 8))
    target = feats @ rng.standard_normal(8) + 0.1 * rng.standard_normal(200)
    return feats, target


def test_ex3_matches_reference():
    feats, target = _signal_fixture()
    try:
        got = ex3.linear_probe_r2(feats, target, 1.0)
    except NotImplementedError:
        pytest.skip("linear_probe_r2 not implemented yet — that's the exercise")
    want = _ref_probe_r2(feats, target, 1.0)
    assert abs(got - want) < CHECKS["ex3"]["abs_tol"], f"held-out R^2 differs from reference: got {got}, want {want}"


def test_ex3_recovers_linear_signal():
    feats, target = _signal_fixture()
    try:
        got = ex3.linear_probe_r2(feats, target, 1.0)
    except NotImplementedError:
        pytest.skip("linear_probe_r2 not implemented yet — that's the exercise")
    assert got > 0.8, f"a linearly-encoded target should probe high, got R^2={got:.3f}"


# ---------------------------------------------------------- reproduce (SLOW: runs probe.py)
@pytest.mark.slow
def test_ex1_routed_coord_separates_trained_from_random(tmp_path):
    m = run_probe(tmp_path / "run")
    assert m["trained_coord_r2"] - m["control_coord_r2"] >= RC["min_coord_r2_gap"], \
        f"trained routed-coord R^2 should beat the random control by >= {RC['min_coord_r2_gap']}: {m}"
    assert abs(m["trained_task_acc"] - m["control_task_acc"]) <= RC["max_task_acc_gap"], \
        f"task-id is decodable from BOTH (the 'recovered an input' control): {m}"
