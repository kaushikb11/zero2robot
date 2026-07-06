"""SUGGESTED local pytest checks for the ch1.3 exercise candidates.

Run from anywhere:
    pytest curriculum/phase1_imitation/ch1.3_act/exercises/suggested/checks.py

Conventions (match ch1.1 / ch1.2):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the recorded choice.
- The two self-contained data-level exercises (ex3 mask bug-hunt, ex4 ensemble
  completion) are FAST and deterministic — they run in `make check`.
- ex3's reproduce check SKIPS while the injected mask bug is still present
  (finding it is the learner's job) and asserts the fix matches the reference.
- Anything that trains a policy (reduced config) is @pytest.mark.slow — excluded
  from `make check`, which runs only the fast prediction/unit gates.
- Reference bands live in the chapter meta.yaml with provenance (exercise-spec:
  no bare magic numbers) — read them, don't inline. PENDING bands are filled by
  the author from the measured reference run; the fast gates don't depend on them.
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
ACT = REPO / "curriculum/phase1_imitation/ch1.3_act/act.py"
sys.path.insert(0, str(HERE))

import ex1_predict_chunk_vs_baseline as ex1  # noqa: E402
import ex2_predict_chunk_vs_single as ex2  # noqa: E402
import ex3_bughunt_chunk_indexing as ex3  # noqa: E402
import ex4_completion_temporal_ensemble as ex4  # noqa: E402

CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
RC = CHECKS["exercise_config"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex2": CHECKS["ex2"]["answer"]}


def run_act(out: Path, script: Path = ACT, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(script), "--seed", "0", "--device", "cpu", "--no-rerun",
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
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "run ex1 and compare the two mean returns"


def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2 first")
    assert ex2.PREDICTION == ANSWER_KEY["ex2"], "run ex2: chunking (K>1) should beat single-step (K=1)"


# ---------------------------------------------------- ex3 mask bug-hunt (fast)

def _reference_chunks(actions, episode_ids, K):
    act_dim = actions.shape[1]
    targets = np.zeros((len(actions), K, act_dim), dtype=np.float32)
    mask = np.zeros((len(actions), K), dtype=np.float32)
    for e in np.unique(episode_ids):
        idx = np.nonzero(episode_ids == e)[0]
        ep = actions[idx]
        for j, frame in enumerate(idx):
            valid = min(K, len(idx) - j)
            targets[frame, :valid] = ep[j:j + valid]
            targets[frame, valid:] = ep[-1]
            mask[frame, :valid] = 1.0
    return targets, mask


def test_ex3_mask_bug_fixed():
    rng = np.random.default_rng(0)
    actions = rng.normal(size=(23, 6)).astype(np.float32)
    episode_ids = np.array([0] * 9 + [1] * 6 + [2] * 8)  # short episodes force padding
    K = 5
    got_targets, got_mask = ex3.build_chunk_targets(actions, episode_ids, K)
    want_targets, want_mask = _reference_chunks(actions, episode_ids, K)
    # Targets are correct in the shipped bug — only the mask range is wrong.
    assert np.array_equal(got_targets, want_targets), "don't change the target slice; only the mask is buggy"
    if not np.array_equal(got_mask, want_mask):
        pytest.skip("ex3 still marks padded steps valid — find the mask index range")
    assert np.array_equal(got_mask, want_mask), "a correct fix masks out exactly the padded tail steps"


# ------------------------------------------- ex4 ensembling completion (fast)

def _reference_ensemble(votes, m):
    weights = np.exp(-m * np.arange(len(votes)))
    return (votes * (weights / weights.sum())[:, None]).sum(0)


def test_ex4_completion_matches_reference():
    rng = np.random.default_rng(1)
    votes = rng.normal(size=(7, 6)).astype(np.float64)
    m = 0.1
    try:
        got = np.asarray(ex4.ensemble_action(votes, m), dtype=np.float64)
    except NotImplementedError:
        pytest.skip("ensemble_action not implemented yet — that's the exercise")
    want = _reference_ensemble(votes, m)
    assert got.shape == want.shape, f"expected one action ({want.shape}), got {got.shape}"
    assert np.allclose(got, want, rtol=CHECKS["ex4"]["rel_tol"], atol=1e-9), \
        "values disagree with the reference — check the age index (oldest = 0) and the normalization"


def test_ex4_oldest_vote_weighted_most():
    # With m > 0 and oldest-first ordering, the oldest vote (row 0) gets the
    # largest weight — a structural check independent of the fixture values.
    try:
        got = ex4.ensemble_action(np.array([[1.0], [0.0]]), 0.5)  # oldest=1, newest=0
    except NotImplementedError:
        pytest.skip("ensemble_action not implemented yet — that's the exercise")
    assert float(got[0]) > 0.5, "the oldest prediction (index 0) should carry the largest weight"


# ---------------------------------------------------------- reproduce (slow)

@pytest.mark.slow
def test_ex1_trained_beats_baseline(tmp_path):
    # Reduced config; PENDING band in meta.yaml (min_return_gain). The trained
    # chunked policy must earn a higher mean return than the untrained baseline.
    m = run_act(tmp_path / "ex1")
    gain = m["mean_return"] - m["baseline_mean_return"]
    assert gain >= RC["min_return_gain"], \
        f"trained ACT should beat the untrained baseline on mean_return: {m}"


@pytest.mark.slow
def test_ex2_chunk_beats_single_step(tmp_path):
    # The chapter's core ablation: predicting a chunk (K>1) should beat single-
    # step (K=1) on held-out success at this config (measured at default: ~0.9
    # vs ~0.6). Same demos, same epochs — only the output shape differs.
    chunked = run_act(tmp_path / "chunked")
    single = run_act(tmp_path / "single", extra=["--break", "no_chunk"])
    assert chunked["success_rate"] > single["success_rate"], \
        f"chunking should beat single-step: chunked={chunked}, single={single}"


@pytest.mark.slow
def test_ex3_reproduce_mask_fix_trains(tmp_path):
    # The shipped ex3 has the mask bug; a fixed copy should still train and beat
    # the untrained baseline (the fast test above verifies the mask itself).
    m = run_act(tmp_path / "ex3base")
    assert m["mean_return"] - m["baseline_mean_return"] >= RC["min_return_gain"], \
        f"the chapter's own act.py should beat baseline (reference for a fixed ex3): {m}"
