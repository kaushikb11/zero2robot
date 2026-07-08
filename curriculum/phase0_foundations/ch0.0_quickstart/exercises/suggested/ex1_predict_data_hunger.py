"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch0.0.

Objective tested: the one honest thing this quickstart hides on purpose — WHERE
the win comes from. The policy you just trained solved ~half of its held-out
starts. It learned that from 300 scripted demonstrations. So: was it the 300, or
would a handful have done? Every one of those 300 expert episodes SUCCEEDS; if
success is what the network copies, five perfect demos ought to be plenty.

THE SETUP. This reruns quickstart.py with --demos 5 (and a short --epochs 300,
since five demos train in seconds) on the SAME held-out eval starts, seed 0. All
five demos succeed. Then it compares the trained policy's success rate to the
random-action floor of 0/25.

PREDICT before you run: on the 25 held-out starts, the 5-demo policy will...
  A) still clear the floor comfortably — the expert solved all 5, and success is
     what behavior cloning copies, so the skill carries over
  B) collapse to roughly the random floor — five trajectories don't cover the
     states the policy actually steers itself into, and it has nothing to imitate there
  C) do even BETTER than the 300-demo run — less data means less averaging over
     conflicting demonstrations, so a sharper, more decisive policy

Record your answer in PREDICTION below, then run this file (under a minute on CPU).
Estimated learner time: 10 minutes (mostly the run).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch0.0-quickstart",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
QUICKSTART = REPO / "curriculum/phase0_foundations/ch0.0_quickstart/quickstart.py"
EX = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text())["exercise_checks"]["ex1"]


def run_quickstart(out: Path, demos: int) -> dict:
    cmd = [sys.executable, str(QUICKSTART), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), "--demos", str(demos), "--epochs", str(EX["epochs"]),
           "--eval_episodes", str(EX["eval_episodes"])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        m = run_quickstart(Path(tmp) / "starved", EX["demos"])
    print(f"{m['demos']} demos ({m['expert_successes']} expert successes) -> "
          f"trained {m['successes']}/{m['eval_episodes']} solved ({m['success_rate']:.0%})")
    print(f"random floor            -> {m['random_successes']}/{m['eval_episodes']} solved ({m['random_rate']:.0%})")
    print(f"\n(the full run you already saw used {EX['reference_full_success_rate']:.0%} at 300 demos.  your prediction: {PREDICTION})")
    print("\nNow explain it: every one of the 5 demos SUCCEEDED. Why can't the policy?")
