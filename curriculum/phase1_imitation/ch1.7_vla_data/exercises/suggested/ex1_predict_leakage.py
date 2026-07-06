"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch1.7.

Objective tested: instruction LEAKAGE, the chapter's headline. A VLA is supposed
to read its instruction to know the TASK and its camera to know the current MOVE.
If the instruction template accidentally encodes the move itself, the action
becomes decodable from language ALONE — and a policy trained on it will learn to
ignore the image. This exercise makes you predict, then measure, that failure.

THE SETUP. vla_data.py builds the multi-task dataset and runs a LEAKAGE PROBE: a
linear least-squares read-out of the action from a bag-of-words of the instruction
tokens, per task, reported in metrics.json as `action_from_language_r2` (0 = words
tell you nothing about the action; ~1 = the action is fully decodable from words).
You will run it twice at a reduced config:
  - clean:        default templates ("push the t block onto the target", ...)
  - --break leak: templates that append the move direction every frame
                  ("... moving northeast")

PREDICT before you run: which row describes the two R^2 values?
  A) Both near 0 — words are just a task label; the action always comes from pixels.
  B) Clean near 0, leak much higher (~0.7-0.8) — naming the direction makes the
     action linearly decodable from language, so a policy could skip the image.
  C) Clean higher than leak — extra words only add noise and hurt the read-out.

Record your answer in PREDICTION below, then run this file (seconds on CPU — it
builds the dataset twice at a reduced scale; no training in this chapter).
Estimated learner time: 10 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch1.7-vla-data",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
VLA = REPO / "curriculum/phase1_imitation/ch1.7_vla_data/vla_data.py"
RC = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text())["exercise_checks"]["exercise_config"]


def config_flags() -> list[str]:
    return ["--episodes_per_task", str(RC["episodes_per_task"]), "--frame_stride", str(RC["frame_stride"]),
            "--feature_dim", str(RC["feature_dim"]), "--conv_width", str(RC["conv_width"])]


def run_vla(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(VLA), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), *config_flags(), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        clean = run_vla(Path(tmp) / "clean")
        leak = run_vla(Path(tmp) / "leak", ["--break", "leak"])
    print(f"clean templates -> action_from_language_r2 {clean['action_from_language_r2']:.3f}")
    print(f"--break leak     -> action_from_language_r2 {leak['action_from_language_r2']:.3f}")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: the leak template only ADDED a word naming the direction. "
          "Why does that word let a linear model read the action off the text — and what "
          "would a policy trained on this data learn to do with its camera?")
