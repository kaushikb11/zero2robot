"""SUGGESTED local pytest checks for the ch5.5 exercise candidates.

Run from anywhere:
    pytest curriculum/phase5_practitioner/ch5.5_fast/exercises/suggested/checks.py

Conventions (match ch1.7 / ch5.1):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the recorded choice.
- ex2 (idct completion) is FAST + self-contained (pure numpy). It SKIPS while `idct` raises
  NotImplementedError, then asserts the completion round-trips the orthonormal transform to
  machine precision AND that a naively "un-scaled" inverse would NOT.
- ex1 and ex3 run the CODEC via subprocess. Unlike ch5.1's training gates these are FAST
  (pure-numpy, no policy, no rendering — well under a second), so they run in `make check`,
  NOT @pytest.mark.slow. They assert the DIRECTION (fast < naive tokens at comparable rmse;
  --break craters rmse + jerk), never an exact count — see meta.yaml for the measured values.
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
FAST = REPO / "curriculum/phase5_practitioner/ch5.5_fast/fast.py"
sys.path.insert(0, str(HERE))

import ex1_predict_compression as ex1  # noqa: E402
import ex2_completion_idct as ex2  # noqa: E402
import ex3_predict_break as ex3  # noqa: E402

CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
RC = CHECKS["exercise_config"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex3": CHECKS["ex3"]["answer"]}


def run_fast(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(FAST), "--seed", "0", "--no-rerun", "--out", str(out),
           "--horizon", str(RC["horizon"]), "--episodes_per_task", str(RC["episodes_per_task"]),
           "--q_scale", str(RC["q_scale"]), "--num_merges", str(RC["num_merges"]), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


# ----------------------------------------------------------------- predictions

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1 first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "run ex1: FAST uses fewer tokens at comparable error (Parseval + BPE on zeros)"


def test_ex3_prediction_recorded():
    if ex3.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex3 first")
    assert ex3.PREDICTION == ANSWER_KEY["ex3"], "run ex3: a time-domain budget craters rmse + smoothness"


# ------------------------------------------ ex2 idct completion (fast, self-contained)

def test_ex2_round_trips_orthonormal_transform():
    D = ex2.dct_matrix(24)
    try:
        rng = np.random.default_rng(0)
        chunk = rng.standard_normal((24, 4))
        recon = np.asarray(ex2.idct(D @ chunk, D), dtype=np.float64)
    except NotImplementedError:
        pytest.skip("idct not implemented yet — that's the exercise")
    assert recon.shape == chunk.shape, f"expected {chunk.shape}, got {recon.shape}"
    assert np.allclose(recon, chunk, atol=CHECKS["ex2"]["abs_tol"]), \
        "idct(dct(x)) must return x — the inverse of an orthonormal transform is its transpose"


def test_ex2_is_not_the_unscaled_bug():
    # A common wrong answer keeps the forward matrix (idct = D @ coeffs) instead of its
    # transpose. For a non-symmetric D that does NOT round-trip; a correct idct must differ.
    D = ex2.dct_matrix(24)
    try:
        rng = np.random.default_rng(1)
        chunk = rng.standard_normal((24, 4))
        recon = np.asarray(ex2.idct(D @ chunk, D), dtype=np.float64)
    except NotImplementedError:
        pytest.skip("idct not implemented yet — that's the exercise")
    wrong = D @ (D @ chunk)  # forgot the transpose
    assert not np.allclose(recon, wrong), \
        "your idct applied D again instead of D.T — that does not invert the transform"


# ---------------------------------------------------------- reproduce (FAST: pure-numpy codec)

def test_ex1_fewer_tokens_at_comparable_error(tmp_path):
    m = run_fast(tmp_path / "clean")
    assert m["compression_ratio"] >= RC["min_compression_ratio"], \
        f"FAST must use fewer tokens than per-step binning: {m}"
    assert m["naive_tokens"] > m["fast_tokens"], f"fast_tokens must be below naive_tokens: {m}"
    assert m["fast_recon_rmse"] <= m["naive_recon_rmse"] * RC["max_rmse_ratio"], \
        f"FAST's reconstruction error must stay comparable to per-step binning (Parseval): {m}"


def test_ex3_time_domain_break_craters_rmse_and_smoothness(tmp_path):
    clean = run_fast(tmp_path / "clean")
    broke = run_fast(tmp_path / "break", ["--break", "time_domain"])
    assert broke["fast_recon_rmse"] >= clean["fast_recon_rmse"] * RC["min_break_rmse_ratio"], \
        f"--break time_domain should crater reconstruction error: clean={clean}, break={broke}"
    assert broke["fast_error_jerk"] >= clean["fast_error_jerk"] * RC["min_break_jerk_ratio"], \
        f"--break time_domain should make the reconstruction jerky: clean={clean}, break={broke}"
