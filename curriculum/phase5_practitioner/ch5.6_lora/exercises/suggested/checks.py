"""SUGGESTED local pytest checks for the ch5.6 exercise candidates.

Run from anywhere:
    pytest curriculum/phase5_practitioner/ch5.6_lora/exercises/suggested/checks.py

Conventions (match ch1.7 / ch1.8 / ch5.1):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces predict-before-run;
  locally we only verify the recorded choice.
- ex3 (LoRALinear completion) is FAST + self-contained (pure torch, no lora.py run) — it runs in
  `make check`. It SKIPS while the functions raise NotImplementedError, then asserts the completion
  matches a reference, that ZERO-init B is a no-op (adapted == frozen at step 0), and that the
  "forgot to zero it" bug (kaiming B) is detectably NOT a no-op — the learner-generated failure.
- ex1 and ex2 RUN lora.py via subprocess (a few seconds on cpu, but a training subprocess), so they
  are @pytest.mark.slow — excluded from `make check`. They assert the DIRECTION (the rank elbow;
  task_A collapses under BOTH arms), never an exact R^2 — this is a state-based, seed-noisy-but-
  bitwise-on-cpu metric.
- Reference bands live in meta.yaml with provenance (exercise-spec: no bare magic numbers) — read
  them, don't inline. Bands are PROVISIONAL pending author reverification.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
LORA = REPO / "curriculum/phase5_practitioner/ch5.6_lora/lora.py"
sys.path.insert(0, str(HERE))

import ex1_predict_elbow as ex1  # noqa: E402
import ex2_predict_forgetting as ex2  # noqa: E402
import ex3_completion_lora as ex3  # noqa: E402

CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
RC = CHECKS["exercise_config"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex2": CHECKS["ex2"]["answer"]}


def run_lora(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(LORA), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


# ----------------------------------------------------------------- predictions

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1 first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "run ex1: held-out fit rises then plateaus; a small rank recovers most"


def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2 first")
    assert ex2.PREDICTION == ANSWER_KEY["ex2"], "run ex2: freezing W does NOT protect task_A — it collapses too"


# ------------------------------------------ ex3 LoRALinear completion (fast, self-contained)

def _reference_forward(x, W, b, A, B, scaling):
    """W x + b + scaling * B (A x) — the LoRA update the learner reimplements."""
    return x @ W.t() + b + scaling * (x @ A.t()) @ B.t()


def _fixture():
    g = torch.Generator().manual_seed(0)
    in_f, out_f, r, batch = 12, 5, 2, 8
    x = torch.randn(batch, in_f, generator=g)
    W = torch.randn(out_f, in_f, generator=g)
    b = torch.randn(out_f, generator=g)
    A = torch.randn(r, in_f, generator=g)
    B = torch.randn(out_f, r, generator=g)
    return x, W, b, A, B, in_f, out_f, r


def test_ex3_delta_matches_reference():
    x, W, b, A, B, in_f, out_f, r = _fixture()
    scaling = 2.0 / r
    try:
        got = ex3.lora_delta(x, A, B, scaling)
    except NotImplementedError:
        pytest.skip("lora_delta not implemented yet — that's the exercise")
    want = scaling * (x @ A.t()) @ B.t()
    assert got.shape == want.shape, f"expected {want.shape}, got {got.shape}"
    assert torch.allclose(got, want, atol=CHECKS["ex3"]["abs_tol"]), \
        "lora_delta differs from scaling * B (A x) — check the project-down (A) then project-up (B) order"


def test_ex3_zero_init_B_is_a_noop():
    # With the CORRECT init_B, the adapter adds nothing at step 0: adapted output == frozen (W x + b).
    x, W, b, A, _, in_f, out_f, r = _fixture()
    scaling = 2.0 / r
    try:
        B0 = ex3.init_B(out_f, r)
        delta = ex3.lora_delta(x, A, B0, scaling)
    except NotImplementedError:
        pytest.skip("init_B / lora_delta not implemented yet — that's the exercise")
    frozen = x @ W.t() + b
    adapted = frozen + delta
    gap = float((adapted - frozen).abs().max())
    assert gap <= RC["max_zero_init_gap"], \
        f"zero-init B must make the adapter a no-op at step 0 (gap ~ 0), got gap {gap} — is your init_B all zeros?"


def test_ex3_buggy_B_breaks_the_noop():
    # THE LEARNER-GENERATED FAILURE: init B like any other Linear (kaiming) instead of zeroing it,
    # and the adapter is NO LONGER a no-op — the frozen output moves before any training. This is
    # exactly `lora.py --break rand_init_B`.
    x, W, b, A, _, in_f, out_f, r = _fixture()
    scaling = 2.0 / r
    try:
        Bbug = ex3.init_B_buggy(out_f, r)
        delta = ex3.lora_delta(x, A, Bbug, scaling)
    except NotImplementedError:
        pytest.skip("lora_delta not implemented yet — that's the exercise")
    gap = float(delta.abs().max())
    assert gap >= RC["min_break_gap"], \
        f"kaiming B should perturb the frozen output at step 0 (gap >= {RC['min_break_gap']}), got {gap}"


# ---------------------------------------------------------- reproduce (SLOW: runs lora.py)

@pytest.mark.slow
def test_ex1_elbow_small_rank_recovers_most(tmp_path):
    m = run_lora(tmp_path / "clean")
    assert m["frozen_heldout_r2"] <= RC["max_frozen_heldout_r2"], \
        f"frozen zero-shot on the held-out skill must fail (R^2 <= 0): {m['frozen_heldout_r2']}"
    assert m["lora_recovered_frac"] >= RC["min_recovered_frac"], \
        f"rank-4 LoRA must recover most of full-FT's held-out gain: {m['lora_recovered_frac']}"
    assert m["lora_trainable_pct"] <= RC["max_headline_trainable_pct"], \
        f"rank-4 LoRA must train a small fraction of the weights: {m['lora_trainable_pct']}%"
    # the plateau: rank-8 held-out fit sits on the full-FT ceiling
    r8 = m["sweep_heldout_r2"][m["sweep_ranks"].index(8)]
    assert r8 >= RC["min_plateau_heldout_r2"], f"held-out fit must plateau by rank 8: {r8}"


@pytest.mark.slow
def test_ex2_freezing_W_does_not_protect_task_a(tmp_path):
    m = run_lora(tmp_path / "clean")
    # BOTH arms forget task_A — freezing W did not protect it (the honest twist; direction only).
    assert m["frozen_task_a_r2"] - m["lora_task_a_r2"] >= RC["min_task_a_forget"], \
        f"task_A must collapse under LoRA (freezing W did not protect it): {m}"
    assert m["frozen_task_a_r2"] - m["full_task_a_r2"] >= RC["min_task_a_forget"], \
        f"task_A must collapse under full-FT too: {m}"
