"""SUGGESTED local pytest checks for the ch5.4 exercise candidates.

Run from anywhere:
    pytest curriculum/phase5_practitioner/ch5.4_vla_shape/exercises/suggested/checks.py

Conventions (match ch1.8 / ch5.1 / ch5.3):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces predict-before-run;
  locally we only verify the recorded choice.
- ex3 (block-mask completion) is FAST + self-contained (pure numpy) — it runs in `make check`. It SKIPS
  while `block_mask` raises NotImplementedError, then asserts the completion matches the reference AND is
  NOT the classic "fully bidirectional" bug (the prefix reading the suffix).
- ex1 TRAINS the two-tower (~90 s CPU) via subprocess, so it is @pytest.mark.slow. It asserts the
  DIRECTION — the held-out flow-MSE gap (cut - full) exceeds a conservative floor — which is byte-
  reproducible on a machine and seed-robust (gaps +0.56..+0.98 across seeds 0/1/2). Never an exact MSE.
- ex2's higher bar (the PushT rollout) FLOORS for both masks at free-tier, so it is deliberately NOT
  gated on a success number (the mechanism lives in the flow-MSE gap, ex1) — see the skipped stub below.
- Reference bands live in meta.yaml with provenance (exercise-spec: no bare magic numbers). Bands are
  PROVISIONAL pending author reverification on the reference tier.
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
ARTIFACT = REPO / "curriculum/phase5_practitioner/ch5.4_vla_shape/vla_shape.py"
sys.path.insert(0, str(HERE))

import ex1_predict_cut_cross as ex1  # noqa: E402
import ex2_predict_rollout_floor as ex2  # noqa: E402
import ex3_completion_block_mask as ex3  # noqa: E402

CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
RC = CHECKS["exercise_config"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex2": CHECKS["ex2"]["answer"]}


def run_artifact(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(ARTIFACT), "--seed", "0", "--device", "cpu", "--no-rerun", "--out", str(out),
           "--episodes", str(RC["episodes"]), "--epochs", str(RC["epochs"]),
           "--eval_episodes", str(RC["eval_episodes"]), "--horizon", str(RC["horizon"]),
           "--model_dim", str(RC["model_dim"]), "--layers", str(RC["layers"]), "--heads", str(RC["heads"]),
           *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


# ----------------------------------------------------------------- predictions

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1 first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "run ex1: cutting suffix->prefix collapses the held-out flow fit"


def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2 first")
    assert ex2.PREDICTION == ANSWER_KEY["ex2"], "run ex2: the PushT rollout floors for BOTH masks at free-tier"


# ------------------------------------------ ex3 block-mask completion (fast)

def _ref_block_mask(P: int, H: int, cut_cross: bool) -> np.ndarray:
    S = P + H
    a = np.zeros((S, S), dtype=bool)
    a[:P, :P] = True                     # prefix <-> prefix
    a[P:, P:] = True                     # suffix <-> suffix
    if not cut_cross:
        a[P:, :P] = True                 # suffix -> prefix (the cross-attention)
    return a                             # prefix -> suffix stays False (blocked)


def test_ex3_matches_reference():
    for P, H in [(14, 8), (3, 2), (5, 4)]:
        for cut in (False, True):
            try:
                got = np.asarray(ex3.block_mask(P, H, cut), dtype=bool)
            except NotImplementedError:
                pytest.skip("block_mask not implemented yet — that's the exercise")
            want = _ref_block_mask(P, H, cut)
            assert got.shape == want.shape, f"expected shape {want.shape}, got {got.shape} (P={P},H={H})"
            assert np.array_equal(got, want), f"mask blocks differ from reference at P={P},H={H},cut={cut}"


def test_ex3_is_not_the_fully_bidirectional_bug():
    # The classic mistake: allow the PREFIX to read the SUFFIX (a fully bidirectional mask). That breaks
    # the action-independent, KV-cacheable prefix — a correct answer must leave prefix->suffix BLOCKED.
    P, H = 14, 8
    try:
        got = np.asarray(ex3.block_mask(P, H, cut_cross=False), dtype=bool)
    except NotImplementedError:
        pytest.skip("block_mask not implemented yet — that's the exercise")
    assert not got[:P, P:].any(), \
        "prefix tokens attend to suffix (action) tokens — the fully-bidirectional bug; the prefix must stay action-independent"
    assert got[P:, :P].all(), "suffix must attend to the whole prefix under the full mask (the cross-attention)"


# ---------------------------------------------------------- reproduce (SLOW: trains the two-tower)

@pytest.mark.slow
def test_ex1_cut_cross_collapses_flow_fit(tmp_path):
    # The REPRODUCIBLE, byte-deterministic headline (seeds 0/1/2, measured): severing the suffix->prefix
    # cross-attention raises the trained expert's HELD-OUT velocity MSE (it loses its only path to the
    # state). We gate the DIRECTION (flow_mse_gap >= a conservative floor), never an absolute MSE.
    m = run_artifact(tmp_path / "out")
    assert m["flow_mse_gap"] >= RC["min_flow_mse_gap"], \
        f"cutting suffix->prefix must collapse the held-out flow fit (gap >= {RC['min_flow_mse_gap']}): {m}"
    assert m["flow_mse_cut"] > m["flow_mse_full"], f"severed fit must be worse than full routing: {m}"


@pytest.mark.slow
def test_ex2_rollout_floors_not_gated(tmp_path):
    # PENDING (Scale Lab): the PushT closed-loop rollout is the HIGHER bar and FLOORS at free-tier for
    # BOTH masks (a from-scratch two-tower on ch1.7's frozen-RANDOM vision backbone can't drive PushT —
    # ch1.8's ceiling). The mechanism is the flow-MSE gap (test_ex1). Re-enable a rollout gate only once
    # an ALIGNED encoder (ch5.2) + a bigger tier make the rollout reachable. See meta.yaml reference_run.
    pytest.skip("PushT rollout floors for both masks at free-tier (Scale-Lab bar) — mechanism is gated via ex1")
