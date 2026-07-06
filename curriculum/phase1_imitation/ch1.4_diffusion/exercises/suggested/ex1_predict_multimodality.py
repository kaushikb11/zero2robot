"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch1.4.

Objective tested: the chapter's THESIS — that a diffusion model captures a
multimodal target where a squared-error regressor collapses to the average. This
is the 2D toy, before any robot, so the effect is measured, not asserted.

THE SETUP. diffusion.py builds a unit-ring target: every direction around the
circle is an equally-good "mode", and the empty CENTER is the one place no data
point sits. It then trains two models on that ring with the SAME MLP width and
the SAME data:
  - a diffusion denoiser (learns to turn noise into a SAMPLE from the ring), and
  - a one-shot MSE regressor (maps noise straight to a point).
It samples 2000 points from each and reports, in metrics.json:
  - toy_diffusion_modes_covered / toy_regress_modes_covered  (of 8 angular sectors)
  - toy_diffusion_mean_radius   / toy_regress_mean_radius     (ring radius is 1.0)

PREDICT before you run: which row describes what the metrics will show?
  A) Both cover all 8 modes — an MLP is an MLP; the objective doesn't matter.
  B) Diffusion covers all 8 modes at radius ~1 (on the ring); regression covers
     ~0 modes at radius ~0 — it lands in the empty center, the average of every
     direction, exactly where no data is.
  C) Regression covers more modes — sampling is noisy, so diffusion smears off
     the ring while the regressor nails it.

Record your answer in PREDICTION below, then run this file (a couple of minutes on
CPU — it trains the toy and a short policy at a reduced scale).
Estimated learner time: 15 minutes (mostly waiting on the run).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch1.4-diffusion",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
DIFFUSION = REPO / "curriculum/phase1_imitation/ch1.4_diffusion/diffusion.py"
RC = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text())["exercise_checks"]["exercise_config"]


def config_flags() -> list[str]:
    return ["--denoising_steps", str(RC["denoising_steps"]), "--model_dim", str(RC["model_dim"]),
            "--num_demos", str(RC["num_demos"]), "--epochs", str(RC["epochs"]),
            "--eval_episodes", str(RC["eval_episodes"])]


def run_diffusion(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(DIFFUSION), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), *config_flags(), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        m = run_diffusion(Path(tmp) / "run")
    print(f"diffusion  -> modes {m['toy_diffusion_modes_covered']}/8  mean_radius {m['toy_diffusion_mean_radius']:.2f}")
    print(f"regression -> modes {m['toy_regress_modes_covered']}/8  mean_radius {m['toy_regress_mean_radius']:.2f}")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: both models saw the same ring and have the same width. "
          "Why does one land ON the ring and the other in the empty middle?")
