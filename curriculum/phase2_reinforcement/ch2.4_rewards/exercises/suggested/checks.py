"""SUGGESTED local pytest checks for the ch2.4 (Reward Design) exercises.

Run from anywhere:
    pytest curriculum/phase2_reinforcement/ch2.4_rewards/exercises/suggested/checks.py

Conventions (match ch2.1):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- Anything that trains PPO (even a short run) is @pytest.mark.slow, so `make
  check` (which runs `-m "not gpu and not slow"`) stays fast.

The two check families mirror the H2 finding from the ch2.1 spike: the
reward-hack check TRAINS (seeded => reproducible, but slow), and asserts only the
ORDERING (reward rises while forward distance stays near 0), never a magnitude.
The fix-the-hack check is DETERMINISTIC — a pure-function property on hand-built
states, zero flake — which is the more reliable exercise shape on a noisy RL
metric.
"""

import sys
from pathlib import Path

import pytest
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import ex1_predict_hack as ex1  # noqa: E402
import ex2_completion_fix_hack as ex2  # noqa: E402
import ex3_bughunt_gae_truncation as ex3  # noqa: E402

# Reference bands live in the chapter meta.yaml with provenance (exercise-spec:
# no bare magic numbers) — read them, don't inline.
CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]


# --- ex1: predict-then-run, trains the hack design (seeded => reproducible) -----

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1_predict_hack.py first")
    assert isinstance(ex1.PREDICTION, str) and len(ex1.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex1_hack_reward_rises_but_robot_goes_nowhere(tmp_path):
    # Measured (cpu, seed 0): the hack's own return rises ~10x over training while
    # forward distance stays ~0 (walks nowhere). The seed-robust claims are the
    # two orderings below, not the magnitudes.
    hack = ex1.measure(tmp_path)
    forward_abs = abs(hack["forward_m"])
    rise = hack["train_return_last"] / max(1e-6, abs(hack["train_return_first"]))
    assert forward_abs <= CHECKS["ex1"]["hack_forward_abs_max"], (
        f"the height-hack should NOT walk: |forward_m|={forward_abs:.3f} m — if it "
        "walked, the proxy accidentally worked (report it; that would be the finding)")
    assert rise >= CHECKS["ex1"]["hack_return_rise_min"], (
        f"the hack's own reward should climb as PPO games it: rise={rise:.1f}x — "
        "the mismatch is 'reward up, behaviour wrong', so the reward must go up")


# --- ex2: code-completion, DETERMINISTIC (pure-function property, no training) --

def _fixed(info):
    return ex2.fixed_reward(info, [0.0] * 8, 1.0)


def test_ex2_fixed_reward_responds_to_forward():
    """The property that fixes the hack: the reward must INCREASE with forward
    velocity (the height-only hack does not). Two states differ only in speed."""
    try:
        slow = _fixed(ex2.make_info(forward_vel=0.0, height=0.27, up_z=1.0, action=[0.0] * 8))
        fast = _fixed(ex2.make_info(forward_vel=0.9, height=0.27, up_z=1.0, action=[0.0] * 8))
    except NotImplementedError:
        pytest.skip("fixed_reward not implemented yet — that's the exercise")
    gap = fast - slow
    assert gap >= CHECKS["ex2"]["fixed_forward_min"], (
        f"walking forward must score higher than standing still: gap={gap:.3f} — "
        "did you add the forward term (info['reward_terms']['forward'])?")


def test_ex2_fixed_reward_is_not_the_height_hack():
    """Guard against 'fixing' it by returning the untouched height-only hack: a
    pure height reward is INSENSITIVE to forward_vel, which the test above already
    catches, but pin it explicitly so the intent is unambiguous."""
    try:
        a = _fixed(ex2.make_info(forward_vel=0.0, height=0.30, up_z=1.0, action=[0.0] * 8))
        b = _fixed(ex2.make_info(forward_vel=0.8, height=0.30, up_z=1.0, action=[0.0] * 8))
    except NotImplementedError:
        pytest.skip("fixed_reward not implemented yet — that's the exercise")
    assert a != b, (
        "your reward ignores forward_vel entirely — that is still the height-only "
        "hack; add a term that depends on forward progress")


# --- ex3: bug-hunt, DETERMINISTIC (GAE on a hand-built trajectory, no training) -

def test_ex3_prediction_recorded():
    if ex3.PREDICTION is None:
        pytest.skip("PREDICTION not set — call which env's last-step advantage is "
                    "larger in ex3_bughunt_gae_truncation.py first")
    assert ex3.PREDICTION == "env0", (
        "measured it yet? the TRUNCATED env keeps its future value, so its advantage "
        "must exceed the terminated env's — run ex3 and read the two columns")


def test_ex3_gae_bootstraps_truncation():
    """SKIP while the injected bug is present (the truncated episode is scored as a
    terminal, so both envs come out identical), assert the correct GAE once fixed.

    The correct closed-form (gamma 0.99, lambda 0.95) is deterministic — no seeds,
    no training. Bands + provenance live in meta.yaml (no bare magic numbers)."""
    band = CHECKS["ex3"]
    adv, ret = ex3.compute_gae(*ex3.build_trajectory())
    trunc, term = adv[:, 0], adv[:, 1]  # env0 truncated, env1 terminated
    if torch.allclose(trunc, term, atol=band["equal_atol"]):
        pytest.skip("ex3 still has the injected bug — the truncated episode gets no "
                    "bootstrap, so it scores identically to the terminated one. Find "
                    "the mask that conflates `done` (out-of-time) with `terminated` "
                    "(fell).")
    # A correct fix credits the truncated last step with the discarded future value...
    assert trunc[1].item() == pytest.approx(band["trunc_last_adv"], abs=band["abs"]), (
        f"truncated last-step advantage should bootstrap V(next): got {trunc[1]:.4f}, "
        f"expected {band['trunc_last_adv']}")
    assert trunc[0].item() == pytest.approx(band["trunc_first_adv"], abs=band["abs"])
    # ...and must leave the genuine terminal path untouched (future really is zero).
    assert term[1].item() == pytest.approx(band["term_last_adv"], abs=band["abs"]), (
        "you changed the TERMINATED path — a real fall must NOT bootstrap. Only the "
        "truncation mask was broken; fix only that.")
    assert term[0].item() == pytest.approx(band["term_first_adv"], abs=band["abs"])
