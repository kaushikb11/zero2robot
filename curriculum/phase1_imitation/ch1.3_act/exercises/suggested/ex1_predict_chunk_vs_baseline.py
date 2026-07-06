"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch1.3.

Objective tested: the chapter's central claim — that predicting a CHUNK of the
next K actions and training a tiny transformer to imitate the expert produces a
policy that actually moves toward the handoff, well beyond a random-init network.

THE SETUP. act.py trains one ACT policy on scripted-expert demos of the
bimanual cube transfer, then rolls out TWO policies on the same held-out reset
seeds: an untrained (random-init) network as a baseline, and the trained one.
Both use temporal ensembling. The reward is shaped (negative distance to the
target each step), so a policy that drives the cube toward the handoff earns a
much less negative mean return than one that sits still.

PREDICT before you run: on held-out mean return...
  A) the untrained baseline ties the trained policy — 50 demos and a tiny
     transformer are too little to learn a bimanual handoff at all
  B) the trained chunked policy clearly beats the baseline — it learned to drive
     the arms toward the cube and the handoff; that gap IS the chunking working
  C) the baseline wins — random exploration covers more of the workspace than a
     policy that commits to a plan

Record your answer in PREDICTION below, then run this file (a few minutes on CPU
— it trains one policy at a reduced scale).
Estimated learner time: 15 minutes (mostly waiting on the run).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch1.3-act",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
ACT = REPO / "curriculum/phase1_imitation/ch1.3_act/act.py"
CFG = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text())
RC = CFG["exercise_checks"]["exercise_config"]


def config_flags() -> list[str]:
    return ["--chunk_size", str(RC["chunk_size"]), "--model_dim", str(RC["model_dim"]),
            "--num_demos", str(RC["num_demos"]), "--epochs", str(RC["epochs"]),
            "--eval_episodes", str(RC["eval_episodes"])]


def run_act(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(ACT), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), *config_flags(), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        m = run_act(Path(tmp) / "run")
    print(f"untrained baseline -> mean_return {m['baseline_mean_return']:.3f}  (success {m['baseline_success_rate']:.2f})")
    print(f"trained ACT        -> mean_return {m['mean_return']:.3f}  (success {m['success_rate']:.2f})")
    print(f"gain = {m['mean_return'] - m['baseline_mean_return']:+.3f}   (your prediction: {PREDICTION})")
    print("\nNow explain it: the network never touched the environment during training. Where did the movement come from?")
