"""SUGGESTED local pytest checks for the ch2.8 (runtime) exercise candidates.

Run from anywhere:
    pytest curriculum/phase2_reinforcement/ch2.8_runtime/exercises/suggested/checks.py

Conventions (match ch2.1 / ch2.2):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- Anything that invokes the artifact is @pytest.mark.slow, so `make check` (which
  runs `-m "not gpu and not slow"`) stays fast. These runs are cheap (virtual
  clock, ~0.5 s each) but still shell out, so they carry the marker.

RL-doctrine note (ch2.1 spike): neither exercise is a Break-It bug-hunt. ex1 is a
predict-then-run on the control-rate cliff; ex2 is a hyperparameter-investigation
on queue_depth vs rate. Both are DETERMINISTIC per seed under --clock virtual, so
the assertions are exact, not statistical — the graded signal is the same on
every seed (no flake, no seeded-band tuning needed).
"""

import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import ex1_predict_control_rate as ex1  # noqa: E402
import ex2_investigate_queue_depth as ex2  # noqa: E402

# Reference bands live in the chapter meta.yaml with provenance (exercise-spec:
# no bare magic numbers) — read them, don't inline.
CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]


# --- ex1: predict-then-run, control-rate cliff (deterministic per seed) --------

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your Hz threshold in ex1 first")
    assert isinstance(ex1.PREDICTION, str) and len(ex1.PREDICTION) > 5, \
        "a prediction should name a falsifiable threshold (a Hz number + a reason)"


@pytest.mark.slow
def test_ex1_balances_above_cliff_and_falls_below(tmp_path):
    # Measured 2026-07-06 (cpu, --clock virtual, seeds 0-2): balances (501 steps)
    # at control_hz >= 10 on every seed; falls at 5 Hz on every seed.
    band = CHECKS["ex1"]
    results = ex1.measure(tmp_path)
    hi = results[band["balances_at_hz"]]
    lo = results[band["falls_at_hz"]]
    assert all(hi["balanced"]), (
        f"the pole should still balance at {band['balances_at_hz']} Hz on every "
        f"seed, got {hi['balanced']}")
    assert min(hi["steps"]) >= band["balanced_steps_min"], (
        f"a balanced run should survive ~501 control steps, got {hi['steps']}")
    assert not any(lo["balanced"]), (
        f"the pole should FALL at {band['falls_at_hz']} Hz on every seed "
        f"(the control rate is too slow), got {lo['balanced']}")


@pytest.mark.slow
def test_ex1_latency_grows_as_rate_drops(tmp_path):
    """The teachable warning sign: sense->act latency rises monotonically as the
    control rate drops, and is already large at the last surviving rate."""
    results = ex1.measure(tmp_path)
    fast = sum(results[50.0]["latency_ms"]) / 3.0
    slow = sum(results[10.0]["latency_ms"]) / 3.0
    assert fast < slow, f"latency should grow as control_hz drops: 50Hz={fast}, 10Hz={slow}"


# --- ex2: queue_depth investigation — deeper buffer != rescued controller ------

def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record what you expect the deep queue to change")
    assert isinstance(ex2.PREDICTION, str) and len(ex2.PREDICTION) > 5, \
        "a prediction should say what the deep queue will and won't change"


@pytest.mark.slow
def test_ex2_deep_queue_removes_drops_but_not_the_fall(tmp_path):
    # Measured 2026-07-06: at control_hz 5, queue_depth 1 drops 20-35 /obs msgs and
    # the pole falls; queue_depth 100 drops 0 and the pole STILL falls (same step).
    band = CHECKS["ex2"]
    results = ex2.measure(tmp_path)
    shallow = results[band["shallow_queue_depth"]]
    deep = results[band["deep_queue_depth"]]
    # The deep queue eliminates the dropped-message symptom...
    assert all(d == 0 for d in deep["obs_dropped"]), (
        f"a depth-{band['deep_queue_depth']} queue should drop no /obs messages, "
        f"got {deep['obs_dropped']}")
    assert all(d > 0 for d in shallow["obs_dropped"]), (
        f"the depth-1 queue should drop messages at {band['slow_control_hz']} Hz, "
        f"got {shallow['obs_dropped']}")
    # ...but does NOT rescue the pole: still falls, at the same moment.
    assert not any(deep["balanced"]), (
        f"a deeper queue must not rescue a too-slow controller, got {deep['balanced']}")
    assert deep["steps"] == shallow["steps"], (
        "the fall is rate-driven, not buffer-driven: deep and shallow queues "
        f"should fall at the same step, got deep={deep['steps']} shallow={shallow['steps']}")
