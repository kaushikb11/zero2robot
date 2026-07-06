"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch1.6.

A success rate is only as honest as the states it was measured on. harness.py
evaluates the STRONG policy twice: once on held-out START SEEDS from the SAME
distribution it trained on (block spawned 0.10-0.24 m from the target), and once
on a LIBERO-style held-out VARIANT — the block spawned FARTHER out (0.24-0.30 m),
a start region no training demo ever visited. Same task, shifted start
distribution. The harness reports both rates and a confidence interval on their
difference.

This is the question every "our policy gets 90%" headline dodges: 90% on WHAT?
The starts it trained near, or starts it has never seen?

PREDICT before you run: what will the held-out comparison show?
  A) Held-out success is significantly BELOW train-distribution success — the diff
     CI excludes 0. The policy learned the starts it saw, not the task in general.
  B) Held-out success is about the SAME (diff CI straddles 0) — a start 6 cm farther
     out is nothing; the policy generalizes.
  C) Held-out success is significantly HIGHER — the farther starts are somehow easier.

Record your answer in PREDICTION below, then run this file (~15 s on CPU).
Estimated learner time: 15 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch1.6-harness",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
HARNESS = REPO / "curriculum/phase1_imitation/ch1.6_harness/harness.py"
RC = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text())["exercise_checks"]["exercise_config"]


def run_harness(out: Path) -> dict:
    cmd = [sys.executable, str(HARNESS), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), "--num_demos", str(RC["num_demos"]),
           "--num_demos_weak", str(RC["num_demos_weak"]), "--hidden_dim", str(RC["hidden_dim"]),
           "--epochs", str(RC["epochs"]), "--eval_episodes", str(RC["eval_episodes"]),
           "--n_seeds", str(RC["n_seeds"])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        m = run_harness(Path(tmp) / "run")
    print(f"train-distribution success {m['strong_pooled_rate']:.2f}  "
          f"[{m['strong_pooled_ci_lo']:.2f}, {m['strong_pooled_ci_hi']:.2f}]")
    print(f"held-out variant  success {m['heldout_pooled_rate']:.2f}  "
          f"[{m['heldout_pooled_ci_lo']:.2f}, {m['heldout_pooled_ci_hi']:.2f}]")
    print(f"gap significant: {m['heldout_gap_significant']}")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: nothing about the task changed — same T, same target, same "
          "physics. Why did moving the START 6 cm cost the policy so much?")
