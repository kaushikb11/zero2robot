"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch1.4.

Objective tested: that the number of DENOISING STEPS is not free — it is how many
times the reverse process gets to nudge pure noise back toward the data. Too few
steps and the sample never resolves. This is the `--break few_steps` misconception
("2 steps is plenty, it's faster"), measured on the toy where it is visible.

THE SETUP. diffusion.py samples the 2D ring with the reverse loop. `--break
few_steps` forces denoising_steps = 2 for the whole run (training and sampling).
The chapter default uses many more. metrics.json reports, for the toy:
  - toy_diffusion_modes_covered  (of 8 angular sectors the samples occupy)
  - toy_diffusion_ring_hit       (fraction of samples within 0.2 of radius 1.0)

You will run it TWICE: once at the exercise-config denoising_steps, once with
`--break few_steps` (2 steps), same seed.

PREDICT before you run: compared to the full step count, the 2-step run will...
  A) match it — the model is trained, so it lands the sample in one or two jumps
  B) do WORSE: with only 2 steps the reverse process can't resolve the ring, so
     ring_hit drops sharply (the samples are under-denoised — noisy blobs, not a
     ring), even though the model and data are identical
  C) do BETTER — fewer steps means less accumulated sampling noise, so a cleaner ring

Record your answer in PREDICTION below, then run this file (trains the toy twice at
a reduced scale — a few minutes on CPU).
Estimated learner time: 15 minutes (mostly waiting on the runs).
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
        full = run_diffusion(Path(tmp) / "full")
        few = run_diffusion(Path(tmp) / "few", extra=["--break", "few_steps"])
    print(f"full  ({full['denoising_steps']} steps) -> modes {full['toy_diffusion_modes_covered']}/8  ring_hit {full['toy_diffusion_ring_hit']:.2f}")
    print(f"break (  {few['denoising_steps']} steps) -> modes {few['toy_diffusion_modes_covered']}/8  ring_hit {few['toy_diffusion_ring_hit']:.2f}")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: the trained model is identical. What runs out when you "
          "only allow 2 reverse steps between pure noise and the answer?")
