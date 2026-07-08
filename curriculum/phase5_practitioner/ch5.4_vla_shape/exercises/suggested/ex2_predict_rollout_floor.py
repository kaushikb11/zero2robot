"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch5.4.

Objective tested: the chapter's HONESTY bar and ch1.8's warning. ex1 showed the two-tower's ROUTING is
load-bearing (cutting suffix->prefix collapses the held-out velocity fit). This exercise asks the other
honest question: does the two-tower, with a CORRECT full mask, actually DRIVE PushT to success in a
closed-loop rollout? The vision here is ch1.7's FROZEN RANDOM CNN — the same stand-in ch1.8 used.

PREDICT before you run: how does the full-mask two-tower do on the PushT rollout (success rate)?
  A) It solves PushT (high success) — the two-tower shape + action chunking is enough.
  B) The full mask succeeds and the cut mask fails — the rollout mirrors the flow-MSE gap exactly.
  C) It FLOORS near 0 for the full mask too — a from-scratch model on a FROZEN RANDOM vision backbone
     can't drive PushT (ch1.8's ceiling). The mechanism this chapter measures lives in the flow-MSE
     gap (ex1), NOT in a task-success %. An ALIGNED encoder (ch5.2) is what would lift the rollout.

Record your answer in PREDICTION, then run this file. It TRAINS the two-tower (~90 s CPU); the reproduce
check is slow AND deliberately NOT gated on a success number (the rollout is the higher bar and floors).
This is the point: at free-tier, report the mechanism (flow-MSE gap), report the rollout floor HONESTLY,
and name the upgrade (ch5.2 aligned encoder + a bigger tier / pretrained VLA). Estimated time: 15 min.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch5.4-vla-shape",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase5_practitioner/ch5.4_vla_shape/vla_shape.py"
RC = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text()
                    )["exercise_checks"]["exercise_config"]


def run(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(ARTIFACT), "--seed", "0", "--device", "cpu", "--no-rerun", "--out", str(out),
           "--episodes", str(RC["episodes"]), "--epochs", str(RC["epochs"]),
           "--eval_episodes", str(RC["eval_episodes"]), "--horizon", str(RC["horizon"]),
           "--model_dim", str(RC["model_dim"]), "--layers", str(RC["layers"]), "--heads", str(RC["heads"]),
           *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        full = run(Path(tmp) / "full")                    # default: full-mask rollout
        cut = run(Path(tmp) / "cut", ["--break", "cut_cross"])  # severed-expert rollout
    print(f"PushT rollout success:  full-mask {full['reported_success_rate']:.2f}   "
          f"cut-cross {cut['reported_success_rate']:.2f}   (both floor at free-tier)")
    print(f"mean_return:            full {full['reported_mean_return']:.1f}   cut {cut['reported_mean_return']:.1f}")
    print(f"the MEASURED mechanism (ex1) is the flow-MSE gap: {full['flow_mse_gap']:+.4f}  (your prediction: {PREDICTION})")
    print("\nNow explain it: ch1.1 cloned PushT from STATE and worked; here the two-tower has the state "
          "in its prefix yet the rollout floors. What is missing that a WORKING VLA has (name the encoder "
          "and the tier), and why is it honest to gate this chapter on the routing mechanism, not a success %?")
