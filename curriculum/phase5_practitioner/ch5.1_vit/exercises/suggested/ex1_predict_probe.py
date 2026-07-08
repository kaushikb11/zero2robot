"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch5.1.

Objective tested: what a LINEAR PROBE on the CLS token actually measures. vit.py trains a
from-scratch ViT to classify which quadrant the PushT block sits in, then freezes it and fits
a closed-form linear probe on the CLS feature. It reports three numbers in metrics.json:
  - probe_acc_trained   : the trained ViT's representation, read linearly
  - probe_acc_random    : a SAME-shape random-init ViT's representation, read linearly
  - majority_baseline   : always predict the most common quadrant

PREDICT before you run: how do the three order?
  A) trained ~= random ~= majority — the probe can't tell a trained ViT from a random one
     (a random projection throws away the scene).
  B) trained > random > majority — training makes the scene fact MORE linearly accessible,
     but even a random projection of patches already reads the quadrant coarsely (well above
     the majority guess), because a coarse label is nearly a bag-of-patches property.
  C) trained > majority > random — a random-init ViT is WORSE than guessing the majority.

Record your answer in PREDICTION, then run this file. It TRAINS the ViT at the default config
(~40 s on a CPU laptop). The claim to internalize is the DIRECTION (trained > random >
majority), which holds on every seed — not the exact %, which shifts with the platform's
rendering and the small held-out set (ch1.6). Estimated learner time: 15 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch5.1-vit",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
VIT = REPO / "curriculum/phase5_practitioner/ch5.1_vit/vit.py"
RC = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text())["exercise_checks"]["exercise_config"]


def run_vit(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(VIT), "--seed", "0", "--device", "cpu", "--no-rerun", "--out", str(out),
           "--episodes", str(RC["episodes"]), "--epochs", str(RC["epochs"]), "--warmup", str(RC["warmup"]),
           "--dim", str(RC["dim"]), "--depth", str(RC["depth"]), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        m = run_vit(Path(tmp) / "clean")
    print(f"probe_acc_trained  {m['probe_acc_trained']:.3f}")
    print(f"probe_acc_random   {m['probe_acc_random']:.3f}")
    print(f"majority_baseline  {m['majority_baseline']:.3f}   (chance {m['chance']:.2f})")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: a random-init ViT never learned anything, yet its probe sits well "
          "above the majority guess. What does that tell you about how much of this 'quadrant' "
          "task is really a bag-of-patches property — and what did TRAINING add on top?")
