"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch5.8.

Objective tested: does the LOOP actually close? real_loop.py runs the whole LeRobot pipeline on
the SO-101's real body — it DRIVES a scripted reach, RECORDS a real LeRobotDataset, TRAINS a BC
clone on that recording, and DEPLOYS the clone back into the sim. It then reports three success
rates in metrics.json, each over the SAME held-out box placements:
  - clone_success_rate    : the recorded-then-cloned BC policy
  - noop_success_rate     : hold the rest pose (do nothing)
  - random_success_rate   : random joint targets (flail)

PREDICT before you run: how do the three order?
  A) clone ~= no-op ~= random — cloning a scripted reach from 64 recorded episodes doesn't
     transfer; the deployed policy is no better than doing nothing.
  B) clone >> no-op ~ random — the clone reproduces the scripted reach (well above both
     baselines), i.e. the record -> train -> deploy loop closes end-to-end on the real arm's body.
  C) no-op > clone — holding still is closer to the box than the trained policy gets.

Record your answer in PREDICTION, then run this file. It runs the FULL loop at the default config
(~1 min on a CPU laptop; the SO-101 model is fetched once, ~17MB, then cached). The claim to
internalize is the DIRECTION (clone >> baselines), which holds on every seed — NOT the exact rate,
which shifts with the platform's contact/servo settling and the held-out sample (ch1.6). And the
honest limit: the clone matching the expert is the LOOP closing, not manipulation being solved —
the reality gap (backlash, friction, camera OOD) is what G1 and real hardware are for.
Estimated learner time: 15 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch5.8-real_loop",
            "choices": ["A", "B", "C"], "gate_before_run": True}

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
        m = run_loop(Path(tmp) / "clean")
    print(f"clone_success_rate   {m['clone_success_rate']:.3f}")
    print(f"noop_success_rate    {m['noop_success_rate']:.3f}")
    print(f"random_success_rate  {m['random_success_rate']:.3f}")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: nothing in this run touched real hardware, yet a BC clone trained ONLY "
          "on a dataset you recorded reproduces the reach on the SO-101's real morphology. Which parts "
          "of this loop are byte-for-byte what you'd run on the $150 arm (G1) — and which single thing, "
          "absent here, is the reason G1 exists at all?")
