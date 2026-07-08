"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch3.8.

Objective tested: the chapter's central caveat about reading a checkpoint with a
linear probe. probe.py fits TWO linear probes of the fused layer and compares the
TRAINED checkpoint to a RANDOM-INIT control of the same shape:
  - task-id accuracy   — can a linear map recover WHICH task the instruction selects?
  - routed-coord R^2    — can a linear map recover the VALUE of the state coordinate
                          that task routes to (a quantity the model must COMPUTE)?

The task token is a literal INPUT sitting in the sequence; the routed coordinate's
value is not — the fused token only carries it if the policy learned to combine the
instruction with the state.

PREDICT before you run (default config, seed 0): comparing the trained checkpoint to
the random-init control, which probe SEPARATES them?
  A) BOTH separate — training lifts task-id accuracy AND routed-coord R^2 far above
     the random control.
  B) ONLY the routed-coord R^2 separates them — task-id is ~1.0 for BOTH (a linear
     read of almost any projection recovers a distinct input token), while the
     routed-coord R^2 is high only after training (~0.90 vs ~0.16).
  C) NEITHER separates them — a random-init checkpoint probes the same as a trained
     one; probing tells you nothing.

Record your answer in PREDICTION, then run this file. NOTE: this runs probe.py once at
the default config on CPU (trains a ~7K-param policy for 400 steps + fits the probes) —
under a minute. Estimated learner time: 15 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch3.8-frontier",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
PROBE = REPO / "curriculum/phase3_advanced/ch3.8_frontier/probe.py"


def run_probe(out: Path) -> dict:
    cmd = [sys.executable, str(PROBE), "--seed", "0", "--device", "cpu", "--no-rerun", "--out", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        m = run_probe(Path(tmp) / "run")
    print(f"task-id accuracy  trained {m['trained_task_acc']:.2f}  vs random {m['control_task_acc']:.2f}")
    print(f"routed-coord R^2  trained {m['trained_coord_r2']:.2f}  vs random {m['control_coord_r2']:.2f}")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: WHY does the task-id probe hit ~1.0 even on a checkpoint that "
          "was never trained — and what does that tell you to distrust when a paper reports "
          "that some layer of a real VLA 'encodes' a concept? Which of the two numbers here "
          "is actual evidence that TRAINING put something into the fused representation?")
