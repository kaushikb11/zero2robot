"""SUGGESTED local pytest checks for the ch5.3 exercise candidates.

Run from anywhere:
    pytest curriculum/phase5_practitioner/ch5.3_pixels/exercises/suggested/checks.py

Conventions (match ch1.1 / ch1.8):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the recorded choice.
- The self-contained gate (ex3 InfoNCE completion) is FAST + deterministic — it runs in
  `make check`. It SKIPS while `infonce` raises NotImplementedError.
- The two reproduce checks (ex1 aligned probe < random probe; ex2 unfrozen encoder MEMORIZES ->
  LOWER bc_final_loss) TRAIN at the exercise_config and are @pytest.mark.slow — excluded from
  `make check`.
- Everything is gated on the DIRECTION (aligned < random probe val_mse; trap < frozen BC train
  loss), never an absolute pixel-BC % or a rollout success rate — the rollout floors 0/12 for
  BOTH at free-tier (a Scale Lab) and MuJoCo raster is not bitwise across CPU arches (meta.yaml).
- Reference bands live in meta.yaml with provenance (exercise-spec: no bare magic numbers).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PIXELS = REPO / "curriculum/phase5_practitioner/ch5.3_pixels/pixels.py"
sys.path.insert(0, str(HERE))

import ex1_predict_aligned_vs_random as ex1  # noqa: E402
import ex2_predict_train_encoder as ex2  # noqa: E402
import ex3_completion_infonce as ex3  # noqa: E402

CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
RC = CHECKS["exercise_config"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex2": CHECKS["ex2"]["answer"]}


def run_pixels(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(PIXELS), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), "--episodes", str(RC["episodes"]), "--dim", str(RC["dim"]),
           "--depth", str(RC["depth"]), "--heads", str(RC["heads"]),
           "--align_epochs", str(RC["align_epochs"]), "--bc_epochs", str(RC["bc_epochs"]),
           "--eval_episodes", str(RC["eval_episodes"]), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


# ----------------------------------------------------------------- predictions

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1 first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "run ex1 and compare aligned vs random pixel-BC success"


def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2 first")
    assert ex2.PREDICTION == ANSWER_KEY["ex2"], \
        "run ex2: unfreezing the encoder MEMORIZES the tiny demo set -> LOWER bc_final_loss (overfit signature)"


# ------------------------------------------- ex3 InfoNCE completion (fast)
def _ref_infonce(image_feat, partner_feat, temperature):
    img = F.normalize(image_feat, dim=1)
    par = F.normalize(partner_feat, dim=1)
    logits = img @ par.T / temperature
    labels = torch.arange(len(image_feat))
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _infonce_inputs():
    torch.manual_seed(0)
    return torch.randn(8, 16), torch.randn(8, 16), 0.07


def test_ex3_infonce_matches_reference():
    image_feat, partner_feat, temp = _infonce_inputs()
    try:
        got = ex3.infonce(image_feat, partner_feat, temp)
    except NotImplementedError:
        pytest.skip("infonce not implemented yet — that's the exercise")
    want = _ref_infonce(image_feat, partner_feat, temp)
    assert torch.allclose(got, want, atol=CHECKS["ex3"]["abs_tol"]), \
        f"InfoNCE loss differs from the reference: got {float(got):.6f}, want {float(want):.6f}"


def test_ex3_perfect_alignment_is_low_loss():
    # When image == partner exactly, the diagonal dominates and the loss is near its floor
    # (~log-partition of a perfectly separated batch): far below a random-pairing loss.
    try:
        feat = torch.randn(8, 16)
        aligned = ex3.infonce(feat, feat.clone(), 0.07)
        mismatched = ex3.infonce(feat, torch.randn(8, 16), 0.07)
    except NotImplementedError:
        pytest.skip("infonce not implemented yet — that's the exercise")
    assert aligned < mismatched, "matched pairs must score a lower InfoNCE loss than mismatched pairs"


# ---------------------------------------------------------- reproduce (SLOW: trains)

@pytest.mark.slow
def test_ex1_aligned_probe_beats_random(tmp_path):
    # The REPRODUCIBLE headline (seeds 0/1/2, measured): aligned features are more CONTROL-USEFUL
    # than random — an action-regression probe on the FROZEN features has LOWER held-out val_mse.
    # We gate the DIRECTION (probe_mse_gap = random - aligned > 0), never an absolute number.
    # NOTE: the closed-loop ROLLOUT direction does NOT reproduce at free-tier (0/12 for BOTH
    # encoders — see meta.yaml); that higher bar is the Scale Lab and is deliberately NOT gated.
    m = run_pixels(tmp_path / "out")
    assert m["probe_mse_gap"] >= RC["min_probe_advantage"], \
        f"aligned features should be more control-useful than random (lower probe val_mse): {m}"


@pytest.mark.slow
def test_ex2_trainable_encoder_memorizes(tmp_path):
    # REFRAMED off the rollout (which floors 0/12 for BOTH at free-tier — that overclaim is why the
    # chapter leads on the probe, not success). The signal that actually MOVES with --train_encoder
    # is the BC TRAINING loss: unfreezing the encoder end-to-end lets the tiny from-scratch ViT
    # MEMORIZE the demo set, driving bc_final_loss LOWER than the frozen head-only fit — the overfit
    # signature (it buys no skill; the rollout still floors for both). We gate the DIRECTION only
    # (trap reaches a lower BC train loss than frozen by >= the meta band), never an absolute loss.
    frozen = run_pixels(tmp_path / "frozen")
    trap = run_pixels(tmp_path / "trap", ["--train_encoder"])
    drop = frozen["bc_final_loss_aligned"] - trap["bc_final_loss_aligned"]
    assert drop >= RC["min_train_loss_drop"], \
        f"unfreezing the encoder should reach a LOWER BC train loss (memorizing): frozen={frozen}, trap={trap}"
