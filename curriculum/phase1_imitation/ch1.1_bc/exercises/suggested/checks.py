"""SUGGESTED local pytest checks for the ch1.1 exercise candidates.

Run from anywhere:
    pytest curriculum/phase1_imitation/ch1.1_bc/exercises/suggested/checks.py

Conventions (match ch0.1):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- ex1's check SKIPS while the known injected bug is still present (finding it
  is the learner's job) and asserts agreement with bc.py once the file changes.
- Anything that trains at more than smoke scale is @pytest.mark.slow.
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
ARTIFACT = REPO / "curriculum/phase1_imitation/ch1.1_bc/bc.py"
GEN_DEMOS = REPO / "curriculum/common/envs/pusht/gen_demos.py"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import ex2_completion_rollout as ex2  # noqa: E402
import ex3_data_vs_epochs as ex3  # noqa: E402
import ex4_perturbation_curve as ex4  # noqa: E402

ANSWER_KEY = {"ex4": "B"}  # kept out of the exercise files to avoid spoilers

# ex1 discrimination config: small enough to run in ~1 min, large enough that
# the policy reliably reaches the block (at smoke scale no rollout touches it
# and the bug is invisible in every metric — measured).
EX1_CONFIG = ["--epochs", "80", "--hidden_dim", "128", "--eval_episodes", "5",
              "--seed", "0", "--no-rerun", "--device", "cpu"]  # cpu: matches the
# meta.yaml reference band's provenance and keeps candidate/reference on the same
# (deterministic) device, so the comparison is exact rather than device-dependent.
# Reference signature + seeded bands live in the chapter meta.yaml with their
# provenance (exercise-spec: no bare magic numbers) — read them, don't inline.
EX1 = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]["ex1"]


def run_metrics(script: Path, data: Path, out: Path) -> dict:
    cmd = [sys.executable, str(script), "--data", str(data), "--out", str(out), *EX1_CONFIG]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


@pytest.fixture(scope="module")
def demos30(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("demos") / "demos30"
    subprocess.run([sys.executable, str(GEN_DEMOS), "--episodes", "30", "--seed", "0",
                    "--out", str(out), "--no-video"],
                   check=True, capture_output=True, cwd=REPO)
    return out


@pytest.mark.slow
def test_ex1_fix_restores_reference_metrics(tmp_path, demos30):
    candidate = run_metrics(HERE / "ex1_bughunt_normalize_twice.py", demos30, tmp_path / "ex1")
    if abs(candidate["mean_episode_return"] - EX1["buggy_mean_episode_return"]) < EX1["buggy_band"]:
        pytest.skip("ex1 still has the injected bug — finding and fixing it is the exercise")
    reference = run_metrics(ARTIFACT, demos30, tmp_path / "ref")
    # a correct fix makes the run byte-for-byte the chapter's; allow float slack
    assert candidate["mean_episode_return"] == pytest.approx(
        reference["mean_episode_return"], abs=EX1["ref_mean_episode_return_abs"])
    assert candidate["success_rate"] == pytest.approx(
        reference["success_rate"], abs=EX1["ref_success_rate_abs"])
    assert candidate["final_train_loss"] == pytest.approx(
        reference["final_train_loss"], abs=EX1["ref_train_loss_abs"]), \
        "training was never broken — did you change it?"


def test_ex2_rollout_succeeds_with_expert():
    from curriculum.common.envs.pusht import PushTEnv, ScriptedExpert

    env = PushTEnv()
    try:
        for episode in range(3):
            expert = ScriptedExpert(seed=episode)
            success, episode_return = ex2.rollout_episode(
                lambda obs: expert.action(env), env, seed=10_000 + episode)
            assert success, "the scripted expert never fails these seeds — the loop is the suspect"
            assert -60.0 < episode_return < 2.0, (
                f"return {episode_return} out of range: accumulate the per-step reward, "
                "once per step, starting from the reset")
    except NotImplementedError:
        pytest.skip("rollout_episode not implemented yet — that's the exercise")


def test_ex2_rollout_fails_with_do_nothing_policy():
    from curriculum.common.envs.pusht import PushTEnv

    env = PushTEnv()
    try:
        success, episode_return = ex2.rollout_episode(
            lambda obs: np.zeros(2, dtype=np.float32), env, seed=10_000)
    except NotImplementedError:
        pytest.skip("rollout_episode not implemented yet — that's the exercise")
    assert not success, "a frozen pusher cannot succeed; is the loop reading info['success']?"
    assert episode_return < -50.0, "a full 300-step timeout has a large negative return"


def test_ex3_prediction_recorded():
    if ex3.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex3_data_vs_epochs.py first")
    assert isinstance(ex3.PREDICTION, str) and len(ex3.PREDICTION) > 10, \
        "a prediction should say something falsifiable"


@pytest.mark.slow
def test_ex3_data_beats_epochs(tmp_path):
    # Measured 2026-07-03 (cpu): baseline 0.00 success / 0.33 val loss;
    # 10x data 0.15 / 0.055; 10x epochs 0.10 / 0.82 (memorized 20 episodes).
    results = ex3.measure(tmp_path)
    baseline, arm_data, arm_epochs = results.values()
    assert arm_data["success_rate"] >= max(baseline["success_rate"],
                                           arm_epochs["success_rate"]), \
        f"10x data should win on success: {results}"
    assert arm_data["final_val_loss"] < 0.5 * arm_epochs["final_val_loss"], \
        f"10x epochs on 20 demos should overfit hard (val loss blows up): {results}"


def test_ex4_prediction_recorded():
    if ex4.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex4_perturbation_curve.py first")
    assert ex4.PREDICTION == ANSWER_KEY["ex4"], "measured it yet? run ex4 and look at the two curves"


@pytest.mark.slow
def test_ex4_perturbation_curves_reproduce(tmp_path):
    # Measured 2026-07-03 (cpu): 50 demos [0.07, 0.27, 0.13, 0.27];
    # 200 demos [0.33, 0.33, 0.33, 0.47]. Demo count dominates at every
    # delta; curve SHAPE is noisy at 15 episodes/point, so don't assert it.
    curves = ex4.measure(tmp_path)
    small, large = curves[50], curves[200]
    assert np.mean(large) > np.mean(small) + 0.05, \
        f"200 demos should clearly beat 50 across the sweep: {curves}"
    assert large[0] > small[0], f"200 demos should win even unshoved: {curves}"
