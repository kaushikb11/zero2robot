"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch3.3.

Objective tested: energy drift as the honesty metric. A conservative system's
total energy should never change; a real integrator's does, and the SIGN and
GROWTH of that change is the whole story.

THE EXPERIMENT: integrate the same orbit under all three integrators for the
same 6 seconds of sim time (`engine.py --smoke`), and read each one's relative
energy drift at the end.

PREDICT before you run: exactly ONE of the three integrators makes the orbit's
total energy GROW without bound — its orbit spirals outward and eventually
flies apart. Which one is it?
  A) RK4 (it takes four force samples per step, so it must be the leaky one)
  B) explicit (forward) Euler
  C) semi-implicit (symplectic) Euler

Record your answer in PREDICTION below, then run this file.

Before you run: in one sentence, write WHY you expect that integrator's energy to
behave as it will — what is it about how each integrator forms its next position
(which velocity does it reuse, and when) that decides whether energy drifts or holds?

Estimated learner time: 10 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

# Site metadata (the site gates the run cell on a recorded choice; the answer
# key lives in meta.yaml exercise_checks, not here).
METADATA = {
    "type": "predict-then-run",
    "chapter": "ch3.3-engine",
    "choices": ["A", "B", "C"],
    "gate_before_run": True,
}

ARTIFACT = Path(__file__).resolve().parents[2] / "engine.py"


def run_drift(*extra: str) -> dict:
    """Run engine.py to a temp dir and return its per-integrator drift dict."""
    with tempfile.TemporaryDirectory(prefix="z2r-ex1-") as tmp:
        cmd = [sys.executable, str(ARTIFACT), "--seed", "0", "--no-rerun", "--out", tmp, *extra]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return json.loads((Path(tmp) / "metrics.json").read_text())["drift"]


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    drift = run_drift("--smoke")
    for name in ("euler", "semi_implicit", "rk4"):
        rel = drift[name]["rel_final"]
        tag = "ENERGY GROWS" if rel > 0.01 else "bounded / accurate"
        print(f"{name:<14} rel_final energy drift = {rel:>+12.4e}   ({tag})")
    print(f"your prediction: {PREDICTION} — now say WHY that one leaks and the symplectic one does not.")
