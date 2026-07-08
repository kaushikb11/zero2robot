"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch3.9.

Objective tested: THE MISCONCEPTION this chapter kills. Every controller you built
before this one LEARNED — BC cloned a demonstrator, PPO/SAC trained a policy over
thousands of episodes. The reflex is: "to control a robot you must LEARN a policy."
This chapter throws the network away and CONTROLS BY PLANNING — sampling action
sequences, rolling them through the model (the engine you built in ch3.3-3.6), and
acting on the best. No demonstrator, no reward to learn from, no training run.

THE EXPERIMENT: run sampling-based MPC on cartpole SWING-UP (the pole starts hanging
DOWN; the one actuator only pushes the cart), and read the upright fraction and mean
cost against the no-plan random baseline the same run prints.

PREDICT before you run: with a PERFECT model and ZERO learning, what does MPC do?
  A) nothing useful — you cannot control a robot without LEARNING a policy first; the
     pole stays down, no better than the random baseline
  B) it beats random by a little, but underactuated swing-up needs a learned policy to
     actually get the pole up and hold it
  C) it SOLVES it — the pole swings up and balances (upright_frac -> 1.0) at a cost far
     below the random baseline, with no training at all; both CEM and MPPI

Record your answer in PREDICTION below, then run this file. (~3 s per method on a CPU.)

Before you run, write one sentence: WHY can a search through a model swing the pole up
when the actuator can never push the pole directly?

Estimated learner time: 15 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch3.9-mpc",
    "choices": ["A", "B", "C"],
    "gate_before_run": True,
}

CHAPTER = Path(__file__).resolve().parents[2]
ARTIFACT = CHAPTER / "mpc.py"
REPO_ROOT = CHAPTER.parents[2]


def run_mpc(method: str, *extra: str) -> dict:
    """Run mpc.py to a temp dir; return its metrics.json."""
    with tempfile.TemporaryDirectory(prefix="z2r-ch39-ex1-") as tmp:
        cmd = [sys.executable, str(ARTIFACT), "--method", method,
               "--seed", "0", "--no-rerun", "--out", tmp, *extra]
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO_ROOT)
        return json.loads((Path(tmp) / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    for method in ("cem", "mppi"):
        m = run_mpc(method)
        print(f"{method.upper():5s}  upright_frac {m['mpc_upright_frac']:.2f}  mean_cost {m['mpc_mean_cost']:.3f}"
              f"   |  random  upright {m['random_upright_frac']:.2f}  cost {m['random_mean_cost']:.3f}")
    print(f"your prediction: {PREDICTION} — now say WHY planning beats a policy you never trained, "
          "and what it COST you (a model + a search every single tick).")
