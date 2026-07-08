"""SUGGESTED local pytest checks for the ch5.1 exercise candidates.

Run from anywhere:
    pytest curriculum/phase5_practitioner/ch5.1_vit/exercises/suggested/checks.py

Conventions (match ch1.7 / ch1.8):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the recorded choice.
- ex3 (patchify completion) is FAST + self-contained (pure numpy) — it runs in `make check`.
  It SKIPS while `patchify` raises NotImplementedError, then asserts the completion matches a
  reference AND that the classic interleave bug (no permute) is detectably WRONG.
- ex1 and ex2 TRAIN the ViT (~40 s CPU each) via subprocess, so they are @pytest.mark.slow —
  excluded from `make check`. They assert the DIRECTION (trained > random > majority;
  interleave ~= clean; shuffle collapses the gap), never an exact % — the probe is a
  rendered-image, small-held-out, seed-noisy metric (ch1.6).
- Reference bands live in meta.yaml with provenance (exercise-spec: no bare magic numbers) —
  read them, don't inline. Bands are PROVISIONAL pending author reverification.
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
VIT = REPO / "curriculum/phase5_practitioner/ch5.1_vit/vit.py"
sys.path.insert(0, str(HERE))

import ex1_predict_probe as ex1  # noqa: E402
import ex2_predict_patch_bug as ex2  # noqa: E402
import ex3_completion_patchify as ex3  # noqa: E402

CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
RC = CHECKS["exercise_config"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex2": CHECKS["ex2"]["answer"]}


def run_vit(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(VIT), "--seed", "0", "--device", "cpu", "--no-rerun", "--out", str(out),
           "--episodes", str(RC["episodes"]), "--epochs", str(RC["epochs"]), "--warmup", str(RC["warmup"]),
           "--dim", str(RC["dim"]), "--depth", str(RC["depth"]), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


# ----------------------------------------------------------------- predictions

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1 first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "run ex1 and order trained vs random vs majority"


def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2 first")
    assert ex2.PREDICTION == ANSWER_KEY["ex2"], "run ex2: interleave is silent, shuffle_pos collapses the gap"


# ------------------------------------------ ex3 patchify completion (fast)

def _ref_patchify(images: np.ndarray, patch: int) -> np.ndarray:
    """Correct patchify: split, permute grid axes to the front, flatten each patch."""
    B, H, W, C = images.shape
    gh, gw = H // patch, W // patch
    x = images.reshape(B, gh, patch, gw, patch, C).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(B, gh * gw, patch * patch * C)


def _interleave_bug(images: np.ndarray, patch: int) -> np.ndarray:
    """The classic mistake: skip the permute. Same SHAPE, scrambled patches."""
    B, H, W, C = images.shape
    gh, gw = H // patch, W // patch
    return images.reshape(B, gh, patch, gw, patch, C).reshape(B, gh * gw, patch * patch * C)


def _fixture():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(3, 16, 16, 3), dtype=np.uint8).astype(np.float32), 8


def test_ex3_matches_reference():
    images, patch = _fixture()
    try:
        got = np.asarray(ex3.patchify(images, patch), dtype=np.float64)
    except NotImplementedError:
        pytest.skip("patchify not implemented yet — that's the exercise")
    want = _ref_patchify(images, patch).astype(np.float64)
    assert got.shape == want.shape, f"expected shape {want.shape}, got {got.shape}"
    assert np.allclose(got, want, atol=CHECKS["ex3"]["abs_tol"]), \
        "patches differ from reference — did you permute the grid axes in front of the pixel axes?"


def test_ex3_is_not_the_interleave_bug():
    # The interleave bug has the RIGHT shape but the WRONG contents. A correct answer must
    # differ from it — this is the trap ex2 makes you diagnose.
    images, patch = _fixture()
    try:
        got = np.asarray(ex3.patchify(images, patch), dtype=np.float64)
    except NotImplementedError:
        pytest.skip("patchify not implemented yet — that's the exercise")
    bug = _interleave_bug(images, patch).astype(np.float64)
    assert not np.allclose(got, bug), \
        "your patchify equals the interleave bug (no permute) — patches are scrambled though the shape looks right"


# ---------------------------------------------------------- reproduce (SLOW: trains the ViT)

@pytest.mark.slow
def test_ex1_trained_beats_random_beats_majority(tmp_path):
    m = run_vit(tmp_path / "clean")
    assert m["probe_acc_trained"] - m["probe_acc_random"] >= RC["min_trained_over_random"], \
        f"a trained ViT's probe must beat a same-shape random-init ViT: {m}"
    assert m["probe_acc_trained"] - m["majority_baseline"] >= RC["min_trained_over_majority"], \
        f"a trained ViT's probe must beat the majority guess by a wide margin: {m}"
    assert m["probe_acc_random"] > m["majority_baseline"], \
        f"even a random projection reads quadrant above majority (bag-of-patches): {m}"


@pytest.mark.slow
def test_ex2_interleave_silent_shuffle_collapses(tmp_path):
    clean = run_vit(tmp_path / "clean")
    interleave = run_vit(tmp_path / "interleave", ["--break", "patch_interleave"])
    shuffle = run_vit(tmp_path / "shuffle", ["--break", "shuffle_pos"])
    # The reshape bug is SILENT to the coarse probe (a permuted bag is the same bag).
    assert clean["probe_acc_trained"] - interleave["probe_acc_trained"] <= RC["max_interleave_drop"], \
        f"patch_interleave should be ~silent to the probe, not tank it: clean={clean}, interleave={interleave}"
    # Scrambling the position tags erases the trained model's EDGE over random.
    assert shuffle["probe_gap"] <= RC["max_shuffle_gap"], \
        f"shuffle_pos should collapse the trained-over-random gap: {shuffle}"
