"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch0.4.

Objective tested: what an episode IS, and how the dataset scales with the number
of episodes you record. Under --smoke, record.py drives a FIXED number of control
steps per episode, so the only thing --episodes changes is how many episodes (and
therefore frames) you collect. Everything about the SCHEMA stays put.

THE DIFF UNDER STUDY (same seed, same smoke config; only the episode count moves):

    - python record.py --smoke --seed 0 --episodes 2 --no-rerun
    + python record.py --smoke --seed 0 --episodes 4 --no-rerun

PREDICT before you run: going from 2 to 4 recorded episodes, the metrics.json...
  A) doubles n_frames AND changes obs_dim/act_dim (more data means wider rows)
  B) doubles n_frames; obs_dim, act_dim, fps, and feature_keys are unchanged
  C) leaves n_frames the same (episodes are averaged into one) but adds columns

Record your answer in PREDICTION below, then run this file.
Estimated learner time: 10 minutes.
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
    "chapter": "ch0.4-record",
    "choices": ["A", "B", "C"],
    "gate_before_run": True,
}

ARTIFACT = Path(__file__).resolve().parents[2] / "record.py"


def record_metrics(episodes: int, out: Path) -> dict:
    """Run record.py --smoke with a given episode count; return its metrics.json."""
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}  # lerobot writes fully offline
    cmd = [sys.executable, str(ARTIFACT), "--smoke", "--seed", "0",
           "--episodes", str(episodes), "--no-rerun", "--out", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        two = record_metrics(2, Path(tmp) / "two")
        four = record_metrics(4, Path(tmp) / "four")
    print(f"episodes=2: n_episodes={two['n_episodes']}, n_frames={two['n_frames']}, "
          f"obs_dim={two['obs_dim']}, feature_keys={two['feature_keys']}")
    print(f"episodes=4: n_episodes={four['n_episodes']}, n_frames={four['n_frames']}, "
          f"obs_dim={four['obs_dim']}, feature_keys={four['feature_keys']}")
    print(f"your prediction: {PREDICTION} — now explain the measurement to yourself before checking the key.")
