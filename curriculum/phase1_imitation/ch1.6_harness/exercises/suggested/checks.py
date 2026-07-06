"""SUGGESTED local pytest checks for the ch1.6 exercise candidates.

Run from anywhere:
    pytest curriculum/phase1_imitation/ch1.6_harness/exercises/suggested/checks.py

Conventions (match ch1.1 / ch1.3 / ch1.4 / ch1.5):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the recorded choice.
- The two self-contained math exercises (ex3 Wilson-CI completion, ex4 boundary-CI
  bug-hunt) are FAST and deterministic — they run in `make check`.
- ex3 SKIPS while `wilson_ci` raises NotImplementedError and then checks it against
  an independent reference AND the textbook values (0/10 -> [0,0.2775], 5/10 ->
  [0.2366,0.7634]).
- ex4 SKIPS while `report_ci` is still the Wald interval (zero-width at k=0), then
  asserts the fixed Wilson interval gives a POSITIVE upper bound at 0/20.
- Anything that trains (reduced config) is @pytest.mark.slow — excluded from
  `make check`, which runs only the fast math/prediction gates.
- Reference bands live in meta.yaml with provenance (exercise-spec: no bare magic
  numbers) — read them, don't inline.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
HARNESS = REPO / "curriculum/phase1_imitation/ch1.6_harness/harness.py"
sys.path.insert(0, str(HERE))

import ex1_predict_significance as ex1  # noqa: E402
import ex2_predict_heldout as ex2  # noqa: E402
import ex3_completion_wilson as ex3  # noqa: E402
import ex4_bughunt_boundary_ci as ex4  # noqa: E402

CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
RC = CHECKS["exercise_config"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex2": CHECKS["ex2"]["answer"]}
Z95 = 1.959963985


def _ref_wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Independent reference Wilson interval, for checking ex3/ex4."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def run_harness(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(HARNESS), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), "--num_demos", str(RC["num_demos"]),
           "--num_demos_weak", str(RC["num_demos_weak"]), "--hidden_dim", str(RC["hidden_dim"]),
           "--epochs", str(RC["epochs"]), "--eval_episodes", str(RC["eval_episodes"]),
           "--n_seeds", str(RC["n_seeds"]), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


# ----------------------------------------------------------------- predictions

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1 first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], \
        "run ex1: the ranking is not significant at N=20 but is once pooled"


def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2 first")
    assert ex2.PREDICTION == ANSWER_KEY["ex2"], \
        "run ex2: held-out success is significantly below train-distribution success"


# ----------------------------------------------- ex3 Wilson-CI completion (fast)

def test_ex3_wilson_matches_reference_and_textbook():
    tol = float(CHECKS["ex3"]["abs_tol"])
    try:
        got_edge = ex3.wilson_ci(0, 10)
    except NotImplementedError:
        pytest.skip("ex3 wilson_ci not implemented yet — that's the exercise")
    # Independent reference agreement across a spread of (k, n).
    for k, n in [(0, 10), (5, 10), (10, 10), (2, 20), (45, 200), (0, 1)]:
        assert all(abs(a - b) < tol for a, b in zip(ex3.wilson_ci(k, n), _ref_wilson(k, n))), \
            f"wilson_ci({k},{n}) = {ex3.wilson_ci(k, n)} != reference {_ref_wilson(k, n)}"
    # Textbook values (Brown, Cai & DasGupta 2001), to ~1e-3.
    assert abs(got_edge[0] - 0.0) < 1e-3 and abs(got_edge[1] - 0.2775) < 1e-3, \
        f"0/10 should be ~[0, 0.2775], got {got_edge}"
    mid = ex3.wilson_ci(5, 10)
    assert abs(mid[0] - 0.2366) < 1e-3 and abs(mid[1] - 0.7634) < 1e-3, \
        f"5/10 should be ~[0.2366, 0.7634], got {mid}"


def test_ex3_n_zero_is_whole_interval():
    try:
        assert ex3.wilson_ci(0, 0) == (0.0, 1.0), "n=0 carries no information -> the whole [0,1]"
    except NotImplementedError:
        pytest.skip("ex3 wilson_ci not implemented yet — that's the exercise")


# -------------------------------------------- ex4 boundary-CI bug-hunt (fast)

def test_ex4_boundary_has_positive_upper_bound():
    lo, hi = ex4.report_ci(0, 20)
    if hi <= 1e-9:
        pytest.skip("ex4 report_ci is still the Wald interval (zero width at k=0) — find and fix it")
    # A correct (Wilson) fix: 0/20 -> [0, ~0.161], a HONEST band, not a point.
    ref_lo, ref_hi = _ref_wilson(0, 20)
    assert lo == 0.0, "the lower bound at k=0 must clamp to 0"
    assert abs(hi - ref_hi) < 1e-6, f"0/20 upper bound should be the Wilson {ref_hi:.4f}, got {hi:.4f}"


def test_ex4_matches_wilson_in_the_middle():
    # Gate on the boundary bug signature (0/20 collapses to zero width under Wald);
    # the mid-range Wald interval is non-degenerate, so it can't flag the bug itself.
    if ex4.report_ci(0, 20)[1] <= 1e-9:
        pytest.skip("ex4 report_ci is still the Wald interval — find and fix it")
    got = ex4.report_ci(5, 10)
    assert all(abs(a - b) < 1e-6 for a, b in zip(got, _ref_wilson(5, 10))), \
        f"fixed report_ci should be the Wilson interval: {got} vs {_ref_wilson(5, 10)}"


# ---------------------------------------------------------- reproduce (slow)

@pytest.mark.slow
def test_ex1_reproduce_small_overlaps_large_separates(tmp_path):
    # The chapter's rock at the reduced config: the strong-vs-weak ranking is NOT
    # significant at the small N (diff CI straddles 0) but IS once pooled, and the
    # difference CI strictly TIGHTENS as N grows (measured seed 0: N=20 [-0.08,+0.30]
    # -> N=120 [+0.03,+0.16]).
    m = run_harness(tmp_path / "ex1")
    assert m["small_significant"] is False, f"small-N ranking should not be significant: {m}"
    assert m["pooled_significant"] is True, f"pooled ranking should be significant: {m}"
    small_w = m["small_diff_ci_hi"] - m["small_diff_ci_lo"]
    pooled_w = m["pooled_diff_ci_hi"] - m["pooled_diff_ci_lo"]
    assert pooled_w < small_w, f"the diff CI must tighten with more episodes: {pooled_w} !< {small_w}"


@pytest.mark.slow
def test_ex2_reproduce_heldout_gap_is_real(tmp_path):
    # Held-out success is no higher than train-distribution success, and the gap is
    # significant at the pooled N (measured seed 0: 0.01 held-out vs 0.11 train).
    m = run_harness(tmp_path / "ex2")
    assert m["heldout_pooled_rate"] <= m["strong_pooled_rate"], \
        f"held-out should not beat train-distribution: {m}"
    assert m["heldout_gap_significant"] is True, f"the generalization gap should be real: {m}"


@pytest.mark.slow
def test_wilson_and_bootstrap_agree(tmp_path):
    # Two independent derivations of the same band: the analytic Wilson interval and
    # the seeded percentile bootstrap should land within a few points of each other.
    m = run_harness(tmp_path / "boot")
    assert abs(m["strong_pooled_ci_lo"] - m["strong_pooled_bootstrap_lo"]) < 0.06, m
    assert abs(m["strong_pooled_ci_hi"] - m["strong_pooled_bootstrap_hi"]) < 0.06, m
