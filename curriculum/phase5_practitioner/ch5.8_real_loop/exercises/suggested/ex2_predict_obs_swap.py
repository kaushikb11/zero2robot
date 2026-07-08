"""SUGGESTED exercise candidate (humans promote) — predict-then-run + learner-generated failure, ch5.8.

Objective tested: the bug that only shows up at DEPLOY time — the ch0.4 lesson, now on a real arm.
The recorder wrote each observation as [6 joints, box_x, box_y, box_z]. Your deploy script must
feed the policy observations in the SAME layout. `--break obs_swap` simulates the classic mistake:
the deploy code wires box_x and box_y in the OPPOSITE order than the recording did. The training
run is untouched — same dataset, same clean loss (~1e-5). Only the deployed observation is mis-wired.

You will run the loop twice and compare metrics.json: a clean deploy, and `--break obs_swap`.

PREDICT before you run: what happens to clone_success_rate under obs_swap, and what happens to
final_train_loss?
  A) BOTH collapse — a bad observation wiring corrupts training too, so the loss blows up and the
     policy never learns; the failure is visible in the loss curve.
  B) NEITHER changes — the policy is robust to which axis is which; swapping box_x/box_y is
     harmless because the reach only needs the box's distance, not its direction.
  C) final_train_loss is UNCHANGED (~1e-5, training never saw the swap) but clone_success_rate
     COLLAPSES — the arm confidently reaches the wrong way, and no loss curve could have warned you.

Record your answer in PREDICTION, then run this file. It runs the FULL loop twice (~2 min CPU).
This is the failure you GENERATE and diagnose: a train/deploy contract mismatch is invisible to
every metric you watch during training and only appears when the policy meets the world — which on
real hardware is a robot lunging at the wrong spot. It is why G1 insists your calibration `id`
matches across record, train, and rollout. Estimated learner time: 20 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch5.8-real_loop",
            "choices": ["A", "B", "C"], "gate_before_run": True, "generates_failure": True}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase5_practitioner/ch5.8_real_loop/real_loop.py"
RC = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text())["exercise_checks"]["exercise_config"]


def run_loop(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(ARTIFACT), "--seed", "0", "--device", "cpu", "--no-rerun", "--out", str(out),
           "--demos", str(RC["demos"]), "--epochs", str(RC["epochs"]),
           "--eval_episodes", str(RC["eval_episodes"]), "--steps", str(RC["steps"]), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        clean = run_loop(Path(tmp) / "clean")
        swap = run_loop(Path(tmp) / "swap", ["--break", "obs_swap"])
    print(f"clean : train_loss {clean['final_train_loss']:.2e}  clone_success_rate {clean['clone_success_rate']:.3f}")
    print(f"swap  : train_loss {swap['final_train_loss']:.2e}  clone_success_rate {swap['clone_success_rate']:.3f}")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: the two runs trained on the identical recorded dataset and reached the "
          "identical loss. Only the DEPLOY observation differed. Why is a train/deploy data-contract "
          "mismatch invisible to every training metric — and what does that tell you about how much of "
          "'getting a robot working' is watching the loss versus watching the robot?")
