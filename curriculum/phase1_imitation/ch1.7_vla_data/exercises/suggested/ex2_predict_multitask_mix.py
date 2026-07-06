"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch1.7.

Objective tested: what "multi-task" actually produces. The dataset is assembled
from TWO tasks with DIFFERENT episode lengths — PushT episodes end when the block
reaches the target; the ALOHA handoff is a longer pick->carry->handoff->place. With
the SAME number of episodes per task, the frame COUNTS need not match, and the
action dimensionalities differ (2 vs 6, zero-padded to a shared 6). Predicting the
mix forces you to reason about how heterogeneous demos land in one pile.

THE SETUP. vla_data.py runs both tasks for `--episodes_per_task` episodes each,
subsamples frames by `--frame_stride`, and reports in metrics.json:
  num_examples_pusht, num_examples_aloha  (frame counts per task).

PREDICT before you run (reduced config: 3 episodes/task, stride 3): which row holds?
  A) Only one task appears — the second overwrites the first in the shared arrays.
  B) Exactly equal counts — same episodes/task means same frames/task.
  C) Both tasks appear with DIFFERENT counts (episode lengths differ), and every
     example carries a task_id so the two are never confused.

Record your answer in PREDICTION below, then run this file (seconds on CPU).
Estimated learner time: 10 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch1.7-vla-data",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
VLA = REPO / "curriculum/phase1_imitation/ch1.7_vla_data/vla_data.py"
RC = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text())["exercise_checks"]["exercise_config"]


def run_vla(out: Path) -> tuple[dict, dict]:
    cmd = [sys.executable, str(VLA), "--seed", "0", "--device", "cpu", "--no-rerun", "--out", str(out),
           "--episodes_per_task", str(RC["episodes_per_task"]), "--frame_stride", str(RC["frame_stride"]),
           "--feature_dim", str(RC["feature_dim"]), "--conv_width", str(RC["conv_width"])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    metrics = json.loads((out / "metrics.json").read_text())
    ds = np.load(out / "vla_dataset.npz")
    return metrics, {"task_id": ds["task_id"], "action": ds["action"], "action_mask": ds["action_mask"]}


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        m, ds = run_vla(Path(tmp) / "run")
    tasks_present = sorted(set(ds["task_id"].tolist()))
    print(f"pusht frames {m['num_examples_pusht']}  |  aloha frames {m['num_examples_aloha']}  "
          f"|  task ids present: {tasks_present}")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow look at the action arrays: pusht rows use 2 of the 6 action dims and "
          "aloha uses all 6 — the action_mask records which are real. Why is padding + a "
          "mask the honest way to put two embodiments in one tensor?")
