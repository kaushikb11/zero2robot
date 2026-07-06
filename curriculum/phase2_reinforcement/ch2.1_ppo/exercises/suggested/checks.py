"""SUGGESTED local pytest checks for the ch2.1 (PPO) exercise candidates.

Run from anywhere:
    pytest curriculum/phase2_reinforcement/ch2.1_ppo/exercises/suggested/checks.py

Conventions (match ch1.1):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the measurements reproduce.
- Anything that trains (even a short PPO run) is @pytest.mark.slow, so `make
  check` (which runs `-m "not gpu and not slow"`) stays fast.

H2 note (feasibility spike): the two check FAMILIES here behave very differently
on a noisy RL metric, and that contrast is the finding:
  - ex1 is DETERMINISTIC — it checks the GAE ALGORITHM on fixed toy arrays, no
    training, so it has zero flake.
  - ex2 trains PPO. Because the whole pipeline is seeded (torch + numpy + env
    resets), a FIXED-seed run is bit-reproducible on CPU, so the check does not
    flake run-to-run either. The reliable assertion is "the reference SOLVES";
    the ablation effect is only visible in the multi-seed AVERAGE, never
    guaranteed per seed (measured: no-norm-adv still hits 500 on some seeds).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ARTIFACT = REPO / "curriculum/phase2_reinforcement/ch2.1_ppo/ppo.py"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import ex1_completion_gae as ex1  # noqa: E402
import ex2_predict_ablation as ex2  # noqa: E402

# Reference bands live in the chapter meta.yaml with provenance (exercise-spec:
# no bare magic numbers) — read them, don't inline.
CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]


def reference_gae(rewards, values, terminated, done, bootstrap, next_value, gamma, gae_lambda):
    """The correct GAE, mirroring ppo.py's compute_advantages for one env. The
    (1 - terminated) mask is the whole point: a truncated step keeps its
    bootstrap; a terminated step does not."""
    adv = np.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(len(rewards))):
        next_v = next_value if t == len(rewards) - 1 else values[t + 1]
        if done[t]:
            next_v = bootstrap[t]
        delta = rewards[t] + gamma * next_v * (1.0 - terminated[t]) - values[t]
        last = delta + gamma * gae_lambda * (1.0 - done[t]) * last
        adv[t] = last
    return adv


# --- ex1: code-completion, DETERMINISTIC (no training, no RL variance) ---------

TOY = dict(
    rewards=np.ones(6), values=np.array([5.0, 4.0, 3.0, 6.0, 5.0, 4.0]),
    terminated=np.array([0, 0, 1, 0, 0, 0.0]), done=np.array([0, 0, 1, 0, 0, 1.0]),
    bootstrap=np.array([0, 0, 0.0, 0, 0, 10.0]), next_value=4.0, gamma=0.99, gae_lambda=0.95,
)


def test_ex1_gae_matches_reference():
    try:
        got = ex1.compute_gae(**TOY)
    except NotImplementedError:
        pytest.skip("compute_gae not implemented yet — that's the exercise")
    expected = reference_gae(**TOY)
    np.testing.assert_allclose(got, expected, atol=1e-6,
                               err_msg="GAE mismatch — check the bootstrap mask and the lambda reset")


def test_ex1_truncation_beats_termination():
    """The pedagogical property: the truncated step's advantage must clearly
    exceed the terminated step's, because only the truncated one bootstraps."""
    try:
        adv = ex1.compute_gae(**TOY)
    except NotImplementedError:
        pytest.skip("compute_gae not implemented yet — that's the exercise")
    gap = float(adv[5] - adv[2])  # step 5 truncated, step 2 terminated
    assert gap >= CHECKS["ex1"]["truncated_minus_terminated_adv_min"], (
        f"truncated-step adv should dominate the terminated-step adv, gap={gap:.2f} — "
        "did you bootstrap on truncation but mask termination?")


# --- ex2: predict-then-run, trains PPO (multi-seed, seeded => reproducible) ----

def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2_predict_ablation.py first")
    assert isinstance(ex2.PREDICTION, str) and len(ex2.PREDICTION) > 5, \
        "a prediction should say something falsifiable (a/b/c + a reason)"


@pytest.mark.slow
def test_ex2_reference_solves_and_beats_ablation_on_average(tmp_path):
    # Measured 2026-07-06 (cpu, seeds 0-2, default config): reference solves
    # every seed [500, 500, 500]; no-norm-adv [332, 500, 500]. The reliable
    # signal is that the reference SOLVES; the ablation is worse only on average.
    results = ex2.measure(tmp_path)
    ref, ablation = results["reference"], results["no-norm-adv"]
    ref_mean = float(np.mean(ref))
    assert ref_mean >= CHECKS["ex2"]["ref_eval_min"], \
        f"reference PPO should reliably solve cartpole (mean {ref_mean:.0f} over seeds {ex2.SEEDS})"
    assert min(ref) >= 400.0, f"every reference seed should nearly solve: {ref}"
    # Deterministic aggregate comparison (not a per-seed claim — seeds tie at 500):
    assert ref_mean >= float(np.mean(ablation)), \
        f"advantage norm should not HURT on average: ref {ref} vs no-norm-adv {ablation}"
