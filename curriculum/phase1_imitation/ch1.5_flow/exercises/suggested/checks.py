"""SUGGESTED local pytest checks for the ch1.5 exercise candidates.

Run from anywhere:
    pytest curriculum/phase1_imitation/ch1.5_flow/exercises/suggested/checks.py

Conventions (match ch1.1 / ch1.3 / ch1.4):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the recorded choice.
- The two self-contained math exercises (ex3 interpolation bug-hunt, ex4 Euler-
  sampler completion) are FAST and deterministic — they run in `make check`.
- ex3 SKIPS while the swapped-coefficient bug is still present (finding it is the
  learner's job) and asserts the fix puts the endpoints at noise (t=0) / data (t=1).
- ex4 SKIPS while `euler_sample` raises NotImplementedError.
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
FLOW = REPO / "curriculum/phase1_imitation/ch1.5_flow/flow.py"
DIFFUSION = REPO / "curriculum/phase1_imitation/ch1.4_diffusion/diffusion.py"
sys.path.insert(0, str(HERE))

import ex1_predict_multimodality as ex1  # noqa: E402
import ex2_predict_flow_vs_diffusion_steps as ex2  # noqa: E402
import ex3_bughunt_interpolation as ex3  # noqa: E402
import ex4_completion_euler_sampler as ex4  # noqa: E402

CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
RC = CHECKS["exercise_config"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex2": CHECKS["ex2"]["answer"]}


def run_flow(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(FLOW), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), "--flow_steps", str(RC["flow_steps"]),
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
    assert ex2.PREDICTION == ANSWER_KEY["ex2"], "run ex2: flow should cover the ring at few steps where 2-step diffusion cannot"


# ------------------------------------------ ex3 interpolation bug-hunt (fast)

def _ref_interpolate(data, noise, t):
    return (1.0 - t)[:, None] * noise + t[:, None] * data


def test_ex3_interpolation_endpoints():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(16, 2))
    noise = rng.normal(size=(16, 2))
    t = rng.uniform(0.05, 0.95, size=16)
    got = ex3.flow_interpolate(data, noise, t)
    want = _ref_interpolate(data, noise, t)
    if not np.allclose(got, want):
        pytest.skip("ex3 flow_interpolate still has the coefficients swapped — find it")
    # A correct fix pins the endpoints: t=0 -> noise, t=1 -> data.
    zeros, ones = np.zeros(8), np.ones(8)
    d, n = rng.normal(size=(8, 2)), rng.normal(size=(8, 2))
    assert np.allclose(ex3.flow_interpolate(d, n, zeros), n), "at t=0 the path must equal the noise"
    assert np.allclose(ex3.flow_interpolate(d, n, ones), d), "at t=1 the path must equal the data"


def test_ex3_midpoint_is_average_when_fixed():
    # At t=0.5 the straight path sits exactly halfway — but the midpoint is (data+noise)/2
    # for the SWAPPED coefficients too, so it can't detect the bug on its own. Gate on an
    # ASYMMETRIC time (t=0.25 must lean toward the NOISE), then assert the midpoint.
    rng = np.random.default_rng(1)
    data = rng.normal(size=(32, 2))
    noise = rng.normal(size=(32, 2))
    t_q = np.full(32, 0.25)
    if not np.allclose(ex3.flow_interpolate(data, noise, t_q), 0.75 * noise + 0.25 * data):
        pytest.skip("ex3 not fixed yet — at t=0.25 the path should lean toward the NOISE (0.75 noise + 0.25 data)")
    got = ex3.flow_interpolate(data, noise, np.full(32, 0.5))
    assert np.allclose(got, 0.5 * (data + noise)), "fixed interpolation is the straight-line midpoint at t=0.5"


# ------------------------------------------- ex4 Euler-sampler completion (fast)

def test_ex4_constant_velocity_lands_on_target():
    # If the velocity is the constant (data - noise), Euler integration from the
    # noise reaches exactly the data at t=1 for ANY step count (a straight path is
    # integrated exactly by Euler). This is the flow-matching ideal.
    rng = np.random.default_rng(2)
    noise = rng.normal(size=(8, 2))
    data = rng.normal(size=(8, 2))
    const_v = data - noise
    try:
        got = np.asarray(ex4.euler_sample(lambda x, t: const_v, noise, steps=RC["flow_steps"]), dtype=np.float64)
    except NotImplementedError:
        pytest.skip("euler_sample not implemented yet — that's the exercise")
    assert got.shape == data.shape, f"expected shape {data.shape}, got {got.shape}"
    assert np.allclose(got, data, atol=CHECKS["ex4"]["abs_tol"]), \
        "a straight (constant-velocity) path should land exactly on the data at t=1"


def test_ex4_step_count_and_times():
    # The sampler must call velocity exactly `steps` times, at times 0, dt, ..., and
    # advance by dt each time. We record the times it queries and the final point.
    try:
        seen: list[float] = []

        def velocity(x, t):
            seen.append(float(t))
            return np.ones_like(x)

        x0 = np.zeros((1, 3))
        got = ex4.euler_sample(velocity, x0, steps=4)
    except NotImplementedError:
        pytest.skip("euler_sample not implemented yet — that's the exercise")
    assert len(seen) == 4, f"expected 4 velocity evaluations for steps=4, got {len(seen)}"
    assert np.allclose(seen, [0.0, 0.25, 0.5, 0.75]), f"times should be i*dt for i in 0..steps-1, got {seen}"
    # With unit velocity and dt=0.25 over 4 steps, the point advances by 1.0 total.
    assert np.allclose(got, 1.0), "unit velocity integrated over t in [0,1] should advance the point by 1.0"


# ---------------------------------------------------------- reproduce (slow)

@pytest.mark.slow
def test_ex1_flow_covers_modes_regression_collapses(tmp_path):
    # The chapter thesis at the reduced config: flow covers strictly more of the
    # ring's angular modes than the one-shot regressor, and sits at a larger radius
    # (the regressor collapses toward the empty center). Measured at default:
    # 8/8 vs 0/8, radius ~0.94 vs ~0.06.
    m = run_flow(tmp_path / "ex1")
    assert m["toy_flow_modes_covered"] > m["toy_regress_modes_covered"], \
        f"flow should cover more modes than regression: {m}"
    assert m["toy_flow_mean_radius"] - m["toy_regress_mean_radius"] >= RC["min_radius_gap"], \
        f"regression should collapse toward the center (small radius): {m}"


@pytest.mark.slow
def test_ex2_flow_beats_diffusion_in_few_step_regime(tmp_path):
    # The flow-specific claim: at a few sampling steps, flow (integrated at EFF_STEPS
    # from a well-trained net) covers at least as many ring modes as 2-step diffusion,
    # and strictly more than diffusion's few_steps break by a measured margin.
    f = run_flow(tmp_path / "flow")
    cmd = [sys.executable, str(DIFFUSION), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(tmp_path / "diff"), "--denoising_steps", str(RC["flow_steps"]),
           "--model_dim", str(RC["model_dim"]), "--num_demos", str(RC["num_demos"]),
           "--epochs", str(RC["epochs"]), "--eval_episodes", str(RC["eval_episodes"]),
           "--break", "few_steps"]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    d = json.loads((tmp_path / "diff" / "metrics.json").read_text())
    assert f["toy_flow_lowstep_modes_covered"] - d["toy_diffusion_modes_covered"] >= RC["min_fewstep_mode_gap"], \
        f"flow at few Euler steps should cover more modes than 2-step diffusion: flow={f}, diff={d}"


@pytest.mark.slow
def test_ex3_reproduce_chapter_trains(tmp_path):
    # The shipped ex3 has the swapped-coefficient bug; the chapter's own flow.py uses
    # the correct interpolation, so its toy covers the ring (reference for a fixed ex3).
    m = run_flow(tmp_path / "ex3base")
    assert m["toy_flow_modes_covered"] >= m["toy_regress_modes_covered"] + RC["min_mode_gap"], \
        f"the chapter's own flow.py should cover the ring's modes: {m}"
