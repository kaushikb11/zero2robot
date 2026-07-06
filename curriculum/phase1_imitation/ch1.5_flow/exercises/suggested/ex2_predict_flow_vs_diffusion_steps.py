"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch1.5.

Objective tested: flow matching DECOUPLES sampling from training — you integrate as
few or as many ODE steps as you like from ONE trained velocity net. Plain diffusion
(DDPM) cannot: its sampling-step count is welded to the noising schedule it trained on.
So a flow net trained once samples cleanly at a few steps, while few-step diffusion
means training a few-step schedule (which under-resolves). At MATCHED steps the two are
comparable — the win here is the DECOUPLING, not per-step superiority. Measured side by
side with ch1.4 in the few-step regime.

THE SETUP. Both chapters build the SAME 2D ring toy. We run each at a small sampling
budget and read the ring-mode coverage from metrics.json:
  - flow.py samples the SAME trained velocity net at only EFF_STEPS (=5) Euler steps
    and reports `toy_flow_lowstep_modes_covered`. Flow DECOUPLES sampling steps from
    training: integrate as few or as many as you like from one trained net.
  - ch1.4's diffusion.py cannot decouple — its schedule couples training and
    sampling steps — so the only way to sample it at a few steps is `--break
    few_steps` (denoising_steps = 2), which reports `toy_diffusion_modes_covered`.

PREDICT before you run: in the few-step regime, which model still covers the ring?
  A) Both collapse — nobody can sample a ring in a handful of steps.
  B) Flow still covers ~all 8 modes at 5 Euler steps (sampled from its one trained
     net), while the 2-step diffusion under-resolves — because diffusion's step count
     is welded to training, so "few-step" means the 2-step break. (At MATCHED steps
     diffusion matches flow; the lesson is the decoupling, not per-step superiority.)
  C) Diffusion covers more — its posterior sampler is doing more work per step, so
     it wins when steps are scarce.

Record your answer in PREDICTION below, then run this file (trains both toys at a
reduced scale — a few minutes on CPU).
Estimated learner time: 20 minutes (mostly waiting on the runs).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch1.5-flow",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
FLOW = REPO / "curriculum/phase1_imitation/ch1.5_flow/flow.py"
DIFFUSION = REPO / "curriculum/phase1_imitation/ch1.4_diffusion/diffusion.py"
RC = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text())["exercise_checks"]["exercise_config"]


def shared_flags(steps_flag: str) -> list[str]:
    # ch1.4 and ch1.5 share the same knob names except the step flag.
    return [steps_flag, str(RC["flow_steps"]), "--model_dim", str(RC["model_dim"]),
            "--num_demos", str(RC["num_demos"]), "--epochs", str(RC["epochs"]),
            "--eval_episodes", str(RC["eval_episodes"])]


def run(script: Path, steps_flag: str, out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(script), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), *shared_flags(steps_flag), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        f = run(FLOW, "--flow_steps", Path(tmp) / "flow")
        d = run(DIFFUSION, "--denoising_steps", Path(tmp) / "diff", extra=["--break", "few_steps"])
    print(f"flow  @ {f['toy_flow_lowstep_steps']} Euler steps -> modes {f['toy_flow_lowstep_modes_covered']}/8")
    print(f"diff  @ {d['denoising_steps']} DDPM steps  -> modes {d['toy_diffusion_modes_covered']}/8")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: the conditional flow path is a straight line, so Euler "
          "integration overshoots less per step. What shape is diffusion's reverse "
          "trajectory, and why does cutting it to 2 steps hurt more?")
