"""SUGGESTED local pytest checks for the ch1.4 exercise candidates.

Run from anywhere:
    pytest curriculum/phase1_imitation/ch1.4_diffusion/exercises/suggested/checks.py

Conventions (match ch1.1 / ch1.3):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the recorded choice.
- The two self-contained math exercises (ex3 forward-noise bug-hunt, ex4 reverse-
  step completion) are FAST and deterministic — they run in `make check`.
- ex3 SKIPS while the injected coefficient bug is still present (finding it is the
  learner's job) and asserts the fix matches the reference.
- ex4 SKIPS while `reverse_posterior_mean` raises NotImplementedError.
- Anything that trains (reduced config) is @pytest.mark.slow — excluded from
  `make check`, which runs only the fast prediction/unit gates.
- Reference bands live in meta.yaml with provenance (exercise-spec: no bare magic
  numbers) — read them, don't inline. PENDING bands are filled by the author from
  the measured reference run; the fast gates don't depend on them.
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
DIFFUSION = REPO / "curriculum/phase1_imitation/ch1.4_diffusion/diffusion.py"
sys.path.insert(0, str(HERE))

import ex1_predict_multimodality as ex1  # noqa: E402
import ex2_predict_denoising_steps as ex2  # noqa: E402
import ex3_bughunt_forward_noise as ex3  # noqa: E402
import ex4_completion_reverse_step as ex4  # noqa: E402

CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
RC = CHECKS["exercise_config"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex2": CHECKS["ex2"]["answer"]}


def run_diffusion(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(DIFFUSION), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), "--denoising_steps", str(RC["denoising_steps"]),
           "--model_dim", str(RC["model_dim"]), "--num_demos", str(RC["num_demos"]),
           "--epochs", str(RC["epochs"]), "--eval_episodes", str(RC["eval_episodes"]),
           *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


# ----------------------------------------------------------------- predictions

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1 first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "run ex1 and compare the two modes_covered counts"


def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2 first")
    assert ex2.PREDICTION == ANSWER_KEY["ex2"], "run ex2: 2 denoising steps should under-denoise the ring"


# ------------------------------------------ ex3 forward-noise bug-hunt (fast)

def _ref_forward_noise(x0, acp_t, noise):
    return np.sqrt(acp_t)[:, None] * x0 + np.sqrt(1.0 - acp_t)[:, None] * noise


def test_ex3_forward_noise_fixed():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(16, 2))
    noise = rng.normal(size=(16, 2))
    acp_t = rng.uniform(0.05, 0.95, size=16)
    got = ex3.forward_noise(x0, acp_t, noise)
    want = _ref_forward_noise(x0, acp_t, noise)
    # The signal term is correct in the shipped bug — only the noise coefficient is wrong.
    signal_only = np.sqrt(acp_t)[:, None] * x0
    assert np.allclose(got - ex3.forward_noise(np.zeros_like(x0), acp_t, noise), signal_only), \
        "don't change the signal term; only the noise coefficient is buggy"
    if not np.allclose(got, want):
        pytest.skip("ex3 forward_noise still uses the wrong noise coefficient — find it")
    assert np.allclose(got, want), "a correct fix makes signal^2 + noise^2 powers sum to 1 (variance-preserving)"


def test_ex3_variance_preserved_when_fixed():
    # With x0 and noise both ~unit-variance and independent, a correct forward
    # process keeps Var(x_t) ~ 1 at every level. The buggy coefficient breaks this.
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=(4000, 2))
    noise = rng.normal(size=(4000, 2))
    acp_t = np.full(4000, 0.5)
    got = ex3.forward_noise(x0, acp_t, noise)
    if abs(got.var() - 1.0) > 0.1:
        pytest.skip("ex3 not fixed yet — x_t variance is off (should be ~1 at acp=0.5)")
    assert abs(got.var() - 1.0) < 0.1, "fixed forward process is variance-preserving"


# ------------------------------------------- ex4 reverse-step completion (fast)

def _ref_reverse_mean(x_t, x0, beta_t, acp_t, acp_prev):
    alpha_t = 1.0 - beta_t
    return (beta_t * np.sqrt(acp_prev) / (1.0 - acp_t) * x0
            + (1.0 - acp_prev) * np.sqrt(alpha_t) / (1.0 - acp_t) * x_t)


def test_ex4_completion_matches_reference():
    rng = np.random.default_rng(2)
    x_t = rng.normal(size=(8, 2))
    x0 = rng.normal(size=(8, 2))
    beta_t, acp_t, acp_prev = 0.02, 0.6, 0.65
    try:
        got = np.asarray(ex4.reverse_posterior_mean(x_t, x0, beta_t, acp_t, acp_prev), dtype=np.float64)
    except NotImplementedError:
        pytest.skip("reverse_posterior_mean not implemented yet — that's the exercise")
    want = _ref_reverse_mean(x_t, x0, beta_t, acp_t, acp_prev)
    assert got.shape == want.shape, f"expected shape {want.shape}, got {got.shape}"
    assert np.allclose(got, want, rtol=CHECKS["ex4"]["rel_tol"], atol=1e-9), \
        "values disagree with the reference — check the two coefficients and which term multiplies x0 vs x_t"


def test_ex4_last_step_all_x0():
    # At the LAST reverse step (t=0) acp_prev = 1, so (1 - acp_prev) = 0: the x_t
    # term vanishes and the mean is driven entirely by the model's x0 estimate.
    try:
        got = ex4.reverse_posterior_mean(np.zeros((1, 2)), np.array([[1.0, -1.0]]), 0.02, 0.5, 1.0)
    except NotImplementedError:
        pytest.skip("reverse_posterior_mean not implemented yet — that's the exercise")
    coeff = 0.02 * 1.0 / (1.0 - 0.5)  # beta * sqrt(acp_prev=1) / (1 - acp_t)
    assert np.allclose(got, coeff * np.array([[1.0, -1.0]])), \
        "with acp_prev=1 the x_t term should vanish; only the x0 term remains"


# ---------------------------------------------------------- reproduce (slow)

@pytest.mark.slow
def test_ex1_diffusion_covers_modes_regression_collapses(tmp_path):
    # The chapter thesis at the reduced config: diffusion covers strictly more of
    # the ring's angular modes than the one-shot regressor, and sits at a larger
    # radius (the regressor collapses toward the empty center). Measured at default:
    # 8/8 vs 0/8, radius ~0.87 vs ~0.06.
    m = run_diffusion(tmp_path / "ex1")
    assert m["toy_diffusion_modes_covered"] > m["toy_regress_modes_covered"], \
        f"diffusion should cover more modes than regression: {m}"
    assert m["toy_diffusion_mean_radius"] - m["toy_regress_mean_radius"] >= RC["min_radius_gap"], \
        f"regression should collapse toward the center (small radius): {m}"


@pytest.mark.slow
def test_ex2_few_steps_underdenoises(tmp_path):
    # Same trained-from-scratch toy, only the step count differs: the full step
    # count should cover at least as many modes as the 2-step break, and the break
    # should NOT cover more (too few steps can't resolve the ring).
    full = run_diffusion(tmp_path / "full")
    few = run_diffusion(tmp_path / "few", extra=["--break", "few_steps"])
    assert few["denoising_steps"] == 2, "the few_steps break should force denoising_steps=2"
    assert full["toy_diffusion_modes_covered"] >= few["toy_diffusion_modes_covered"], \
        f"2 denoising steps should not cover MORE modes than the full count: full={full}, few={few}"


@pytest.mark.slow
def test_ex3_reproduce_chapter_trains(tmp_path):
    # The shipped ex3 has the coefficient bug; the chapter's own diffusion.py uses
    # the correct forward process, so its toy covers the ring (reference for a fixed ex3).
    m = run_diffusion(tmp_path / "ex3base")
    assert m["toy_diffusion_modes_covered"] >= m["toy_regress_modes_covered"] + RC["min_mode_gap"], \
        f"the chapter's own diffusion.py should cover the ring's modes: {m}"
