"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch1.9.

The Break-It, as a prediction. bridge.py evaluates the official ACT on HELD-OUT
seeds (base 20000) — starts disjoint from the demo seeds (0..num_demos-1) it
trained on. `--break train_dist` swaps those for the TRAINING seeds themselves:
the exact episode starts the policy already saw. This is the 1.6 sin — grading a
policy on its own training distribution — committed on the official stack, where
it is one flag away and easy to do by accident when you are moving fast in a new
framework.

PREDICT before you run (default budget, seed 0): the official ACT evaluated on the
TRAIN seeds vs the HELD-OUT seeds will show —
  A) a HIGHER success rate on the train seeds: it has effectively memorized those
     starts, so the number inflates and you would over-credit the framework.
  B) the SAME rate: a held-out seed and a training seed are interchangeable.
  C) a LOWER rate on the train seeds.

Record your answer in PREDICTION, then run this file (trains the official ACT
twice; minutes on CPU). Estimated learner time: 20 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch1.9-bridge",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
BRIDGE = REPO / "curriculum/phase1_imitation/ch1.9_bridge/bridge.py"
RC = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text())["exercise_checks"]["exercise_config"]


def run_bridge(out: Path, extra: list[str]) -> dict:
    cmd = [sys.executable, str(BRIDGE), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), "--chunk_size", str(RC["chunk_size"]), "--model_dim", str(RC["model_dim"]),
           "--num_demos", str(RC["num_demos"]), "--epochs", str(RC["epochs"]),
           "--eval_episodes", str(RC["eval_episodes"]), "--no-compare", *extra]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        held = run_bridge(Path(tmp) / "held", [])
        train = run_bridge(Path(tmp) / "train", ["--break", "train_dist"])
    print(f"held-out seeds:   success {held['official_success_rate']:.2f}")
    print(f"train seeds:      success {train['official_success_rate']:.2f}  (--break train_dist)")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: which number would you have put in a paper, and which one is honest? "
          "This is why 1.6's held-out discipline follows you into the official stack.")
