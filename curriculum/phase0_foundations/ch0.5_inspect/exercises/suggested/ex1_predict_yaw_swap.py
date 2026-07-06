"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch0.5.

Objective tested: seeing like a robot. The block's orientation is stored as a
sin/cos pair (obs[4:6]) so it never wraps at +/-pi; the ONE correct way back to
an angle is atan2(sin, cos) — arguments in exactly that order. inspect.py has a
--break yaw-swap mode that decodes it backwards, atan2(cos, sin), which reads the
REFLECTED angle. Nothing about the recorded data changes — only how you read it.

THE DIFF UNDER STUDY (same seed, same dataset; only the yaw decode moves):

    - python inspect.py --seed 0 --episodes 3 --no-rerun
    + python inspect.py --seed 0 --episodes 3 --no-rerun --break yaw-swap

PREDICT before you run: reading the yaw backwards, the metrics.json...
  A) is identical — success is a stored column, so decoding can't change it
  B) keeps n_episodes, n_frames, episode_lengths, and the schema, but success_rate
     collapses toward 0 (every orientation now reads ~pi/2 off, so ang_err blows past ANG_TOL)
  C) changes n_episodes — a decode error drops the miscomputed episodes

Record your answer in PREDICTION below, then run this file.
Estimated learner time: 12 minutes.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

# Site metadata (the site gates the run cell on a recorded choice; the answer
# key lives in checks.py, not here).
METADATA = {
    "type": "predict-then-run",
    "chapter": "ch0.5-inspect",
    "choices": ["A", "B", "C"],
    "gate_before_run": True,
}

ARTIFACT = Path(__file__).resolve().parents[2] / "inspect.py"


def inspect_metrics(break_mode: str, out: Path) -> dict:
    """Run inspect.py --episodes 3 with a given break mode; return its metrics.json."""
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}  # lerobot reads/writes fully offline
    cmd = [sys.executable, str(ARTIFACT), "--seed", "0", "--episodes", "3",
           "--no-rerun", "--break", break_mode, "--out", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        good = inspect_metrics("none", Path(tmp) / "none")
        bad = inspect_metrics("yaw-swap", Path(tmp) / "yaw")
    print(f"none:      success_rate={good['success_rate']:g}  n_episodes={good['n_episodes']}  lengths={good['episode_lengths']}")
    print(f"yaw-swap:  success_rate={bad['success_rate']:g}  n_episodes={bad['n_episodes']}  lengths={bad['episode_lengths']}")
    print(f"your prediction: {PREDICTION} — explain to yourself why the picture and the numbers disagree before checking the key.")
