"""SUGGESTED local pytest checks for the ch1.9 exercise candidates.

Run from anywhere:
    pytest curriculum/phase1_imitation/ch1.9_bridge/exercises/suggested/checks.py

Conventions (match ch1.3 / ch1.5 / ch1.6):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the recorded choice.
- The two self-contained exercises (ex3 env-state bug-hunt, ex4 chunk-timestamps
  completion) are FAST and deterministic — they run in `make check`.
- ex3 SKIPS while `bridge_state_feature` still emits a STATE-typed feature (the
  bug), then asserts the fixed version types the bridged feature as ENV.
- ex4 SKIPS while `action_chunk_timestamps` raises NotImplementedError, then
  checks it against the fps/chunk_size the aloha_cube dataset actually uses.
- Anything that trains (reduced config, runs bridge.py which trains BOTH ACTs) is
  @pytest.mark.slow — excluded from `make check`.
- Reference bands live in meta.yaml with provenance (exercise-spec: no bare magic
  numbers) — read them, don't inline.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
BRIDGE = REPO / "curriculum/phase1_imitation/ch1.9_bridge/bridge.py"
sys.path.insert(0, str(HERE))

import ex1_predict_comparison as ex1  # noqa: E402
import ex2_predict_train_dist as ex2  # noqa: E402
import ex3_bughunt_env_state as ex3  # noqa: E402
import ex4_completion_chunk_timestamps as ex4  # noqa: E402

CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
RC = CHECKS["exercise_config"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex2": CHECKS["ex2"]["answer"]}


def run_bridge(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(BRIDGE), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), "--chunk_size", str(RC["chunk_size"]),
           "--model_dim", str(RC["model_dim"]), "--num_demos", str(RC["num_demos"]),
           "--epochs", str(RC["epochs"]), "--eval_episodes", str(RC["eval_episodes"]),
           *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


# ----------------------------------------------------------------- predictions

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1 first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], \
        "run ex1: the task-tuned from-scratch 1.3 ACT (entity tokens) beats the general official config"


def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2 first")
    assert ex2.PREDICTION == ANSWER_KEY["ex2"], \
        "run ex2: evaluating on TRAIN seeds inflates the success rate above held-out"


# ------------------------------------------------ ex3 env-state bug-hunt (fast)

def test_ex3_bridged_feature_is_env_typed():
    feats = {"observation.state": ("STATE", (10,)), "action": ("ACTION", (6,))}
    out = ex3.bridge_state_feature(feats)
    assert ex3.ENV_STATE_KEY in out, "the bridged feature must be re-keyed to observation.environment_state"
    assert "observation.state" not in out, "the original observation.state key should be consumed"
    if out[ex3.ENV_STATE_KEY][0] == "STATE":
        pytest.skip("ex3 still types the bridged feature STATE — that's the bug; re-type it ENV")
    assert out[ex3.ENV_STATE_KEY] == ("ENV", (10,)), \
        f"bridged feature must be ('ENV', (10,)); got {out[ex3.ENV_STATE_KEY]}"
    assert out["action"] == ("ACTION", (6,)), "the action feature must be left untouched"


# --------------------------------------------- ex4 chunk-timestamps completion (fast)

def test_ex4_chunk_timestamps():
    try:
        got = ex4.action_chunk_timestamps(10, 4)
    except NotImplementedError:
        pytest.skip("ex4 action_chunk_timestamps not implemented yet — that's the exercise")
    assert got == pytest.approx([0.0, 0.1, 0.2, 0.3]), f"fps=10 K=4 -> [0,0.1,0.2,0.3], got {got}"
    assert ex4.action_chunk_timestamps(50, 16) == pytest.approx([i / 50 for i in range(16)])
    assert ex4.action_chunk_timestamps(10, 1) == pytest.approx([0.0]), "K=1 is a single current-frame offset"


# ---------------------------------------------------------- reproduce (slow)

@pytest.mark.slow
def test_ex1_reproduce_from_scratch_not_worse(tmp_path):
    # The measured direction (provenance in meta): the task-tuned from-scratch 1.3 ACT
    # (which splits the obs into four entity tokens) is NOT beaten by the general
    # official config on this task. We assert the direction (from-scratch >= official),
    # not the magnitude, since the reduced budget makes both rates small and noisy.
    m = run_bridge(tmp_path / "cmp")
    assert m["scratch_success_rate"] is not None, "the comparison must run the from-scratch baseline"
    assert m["scratch_success_rate"] >= m["official_success_rate"], \
        f"from-scratch {m['scratch_success_rate']} should not trail official {m['official_success_rate']}: {m}"


@pytest.mark.slow
def test_ex2_reproduce_train_dist_inflates(tmp_path):
    # The Break-It signature: evaluating on the TRAIN seeds does not UNDER-report
    # vs held-out. The methodological error is the lesson; the gap size is config-
    # dependent (provenance in meta), so we assert the direction, not a magnitude.
    held = run_bridge(tmp_path / "held")
    train = run_bridge(tmp_path / "train", extra=["--break", "train_dist", "--no-compare"])
    assert train["official_success_rate"] >= held["official_success_rate"], \
        f"train-distribution eval should not be below held-out: {train} vs {held}"
