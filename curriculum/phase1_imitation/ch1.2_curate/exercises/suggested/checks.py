"""SUGGESTED local pytest checks for the ch1.2 exercise candidates.

Run from anywhere:
    pytest curriculum/phase1_imitation/ch1.2_curate/exercises/suggested/checks.py

Conventions (match ch1.1):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the recorded choice.
- ex3's reproduce check SKIPS while the injected bug is still present (finding
  it is the learner's job) and asserts the fix restores curated > raw.
- Anything that trains BC (reduced config) is @pytest.mark.slow — excluded from
  `make check`, which runs only the fast prediction/unit gates.
- Reference bands live in the chapter meta.yaml with provenance (exercise-spec:
  no bare magic numbers) — read them, don't inline.
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
CURATE = REPO / "curriculum/phase1_imitation/ch1.2_curate/curate.py"
sys.path.insert(0, str(HERE))

import ex1_predict_curate_vs_raw as ex1  # noqa: E402
import ex2_predict_break as ex2  # noqa: E402
import ex4_completion_disagreement as ex4  # noqa: E402

CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
RC = CHECKS["exercise_config"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex2": CHECKS["ex2"]["answer"]}


def run_curate(out: Path, script: Path = CURATE, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(script), "--seed", "0", "--no-rerun", "--out", str(out),
           "--careful", str(RC["careful"]), "--sloppy", str(RC["sloppy"]),
           "--epochs", str(RC["epochs"]), "--eval_episodes", str(RC["eval_episodes"]),
           "--hidden_dim", str(RC["hidden_dim"]), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


# ----------------------------------------------------------------- predictions

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1 first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "run ex1 and compare the two success rates"


def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2 first")
    assert ex2.PREDICTION == ANSWER_KEY["ex2"], "run ex2: the break lands BETWEEN raw and honest curation"


# ------------------------------------------------------------------- ex4 (fast)

def _reference_disagreement(obs, actions, episode_ids, k):
    span = obs.max(0) - obs.min(0)
    obs_n = (obs - obs.min(0)) / np.where(span < 1e-4, 1.0, span)
    out = np.zeros(len(obs), dtype=np.float64)
    for i in range(len(obs)):
        other = episode_ids != episode_ids[i]
        d = np.linalg.norm(obs_n[other] - obs_n[i], axis=1)
        nn = np.nonzero(other)[0][np.argsort(d)[:k]]
        out[i] = actions[nn].std(axis=0).mean()
    return out


def test_ex4_completion_matches_reference():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(60, 4)).astype(np.float32)
    obs[:, 3] = 1.0  # a constant dimension: the normalizer must not divide by zero
    actions = rng.normal(size=(60, 2)).astype(np.float32)
    episode_ids = np.repeat(np.arange(6), 10)
    try:
        got = np.asarray(ex4.frame_disagreement(obs, actions, episode_ids, k=5), dtype=np.float64)
    except NotImplementedError:
        pytest.skip("frame_disagreement not implemented yet — that's the exercise")
    want = _reference_disagreement(obs, actions, episode_ids, k=5)
    assert got.shape == want.shape, f"expected one value per frame ({want.shape}), got {got.shape}"
    assert np.allclose(got, want, rtol=CHECKS["ex4"]["rel_tol"], atol=1e-6), \
        "values disagree with the reference — check the out-of-episode masking and the std"


# ---------------------------------------------------------- reproduce (slow)

@pytest.mark.slow
def test_ex1_curated_beats_raw(tmp_path):
    # Measured 2026-07-05 (cpu, default config): raw 0.08, curated 0.22.
    m = run_curate(tmp_path / "ex1")
    assert m["curated_success_rate"] - m["raw_success_rate"] >= CHECKS["ex1"]["min_delta"], \
        f"curated should beat raw on held-out success: {m}"
    assert m["n_kept"] < m["n_episodes"], "curation is supposed to keep FEWER episodes"


@pytest.mark.slow
def test_ex2_break_lands_between(tmp_path):
    # Measured 2026-07-05 (cpu, default config): raw 0.08, curated 0.22,
    # break(low_disagreement) 0.12 — below curation, and it has the LOWER
    # mean disagreement (0.378 vs 0.380), which is the whole trap.
    honest = run_curate(tmp_path / "honest")
    broken = run_curate(tmp_path / "broken", extra=["--break", "low_disagreement"])
    assert broken["curated_success_rate"] < honest["curated_success_rate"], \
        f"the break should underperform honest curation: honest={honest}, broken={broken}"
    assert broken["mean_disagreement_kept"] <= honest["mean_disagreement_kept"] + 1e-6, \
        "the break selects for LOW disagreement — its kept set should not disagree more than honest curation's"


@pytest.mark.slow
def test_ex3_fix_restores_curation_win(tmp_path):
    ex3 = HERE / "ex3_bughunt_filter.py"
    m = run_curate(tmp_path / "ex3", script=ex3)
    delta = m["curated_success_rate"] - m["raw_success_rate"]
    if delta <= CHECKS["ex3"]["buggy_delta_max"]:
        pytest.skip("ex3 still has the injected bug — the 'curated' policy is no better than raw; find the mask")
    assert delta >= CHECKS["ex3"]["fixed_min_delta"], \
        f"a correct fix makes curated beat raw like the chapter's: {m}"
