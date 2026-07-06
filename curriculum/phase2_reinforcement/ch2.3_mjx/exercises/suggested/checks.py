"""SUGGESTED local pytest checks for the ch2.3 (PPO on MJX) exercise candidates.

Run from anywhere:
    pytest curriculum/phase2_reinforcement/ch2.3_mjx/exercises/suggested/checks.py

Conventions (match ch2.1, RL-doctrine per 00-kickoff/ch2.1-feasibility-spike-scope):
- Prediction gates (PREDICTION unset) SKIP rather than fail — the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- Anything that trains or sweeps (compiles + runs the artifact) is
  @pytest.mark.slow, so `make check` (which runs `-m "not gpu and not slow"`)
  stays fast. NO mandatory Break-It (RL variance makes single-run bugs unreliable
  — the spike's H1 finding; these lean on predict-then-run + hyperparameter
  investigation instead).

Why these two don't flake:
- ex1 sweeps TIMINGS — inherently machine-dependent — so it asserts only the
  QUALITATIVE shape (256 envs clearly out-throughputs 16). The band is a loose
  floor (1.3x) well under the measured ~2.08x, so slow machines still pass.
- ex2 trains PPO, but the whole pipeline is seeded from one PRNGKey and CPU-jax is
  bitwise-deterministic, so a fixed-seed run reproduces exactly. The signal is a
  large, stable GAP (64-env solves, 256-env doesn't at the same budget), not a
  fragile threshold.
"""

import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import ex1_predict_throughput_cliff as ex1  # noqa: E402
import ex2_investigate_env_count as ex2  # noqa: E402

# Reference bands live in the chapter meta.yaml with provenance (exercise-spec:
# no bare magic numbers) — read them, don't inline.
CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]


# --- ex1: predict-then-run, throughput sweep (timing => qualitative assertion) --

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_throughput_cliff.py first")
    assert isinstance(ex1.PREDICTION, str) and len(ex1.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex1_throughput_climbs_with_parallel_envs():
    # Measured 2026-07-06 (cpu-jax): 16 envs ~50k, 256 envs ~105k env-steps/s (~2.08x).
    # Timings vary by machine, so assert only the SHAPE: 256 clearly beats 16.
    tp = ex1.measure()
    assert set(ex1.SWEEP_ENVS) <= set(tp), f"sweep did not report both env counts: {tp}"
    ratio = tp[256] / tp[16]
    assert ratio >= CHECKS["ex1"]["throughput_256_over_16_min"], (
        f"256-env throughput should clearly exceed 16-env (parallelism win): ratio {ratio:.2f}x "
        f"— on a very constrained machine this can shrink; the cliff/plateau is still the lesson")


# --- ex2: hyperparameter-investigation, trains PPO (seeded => reproducible) ------

def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2_investigate_env_count.py first")
    assert isinstance(ex2.PREDICTION, str) and len(ex2.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex2_more_envs_fewer_updates_learns_less(tmp_path):
    # Measured 2026-07-06 (cpu-jax, seed 0, total_steps 300000): num_envs 64 (36
    # updates) eval 407.2; num_envs 256 (9 updates) eval 90.4. The FASTER config
    # learns LESS — throughput is not learning.
    results = ex2.measure(tmp_path)
    ref, fast = results["num_envs_64"], results["num_envs_256"]
    assert ref >= CHECKS["ex2"]["ref_num_envs64_eval_min"], \
        f"num_envs 64 should solve at the default budget (eval {ref:.0f}, expected >= {CHECKS['ex2']['ref_num_envs64_eval_min']})"
    assert ref - fast >= CHECKS["ex2"]["solve_over_fast_margin"], (
        f"the 64-env run should beat the throughput-heavy 256-env run by a wide margin at a fixed "
        f"env-step budget: 64-env {ref:.0f} vs 256-env {fast:.0f} (fewer gradient updates learns less)")
