"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch3.4.

Objective tested: the double pendulum is the textbook CHAOTIC system, and this
chapter's engine reproduces it. Chaos means sensitive dependence on initial
conditions — a hair's-width difference in where you start becomes an
order-of-one difference later. The seed here nudges the launch angle by at most
0.05 rad; watch where the tip ends up 20 seconds later.

THE EXPERIMENT: run the SAME double pendulum twice — once at --seed 0, once at
--seed 1 — and compare where the tip bob finishes. Then run --seed 0 a second
time and compare THAT to the first --seed 0 run.

PREDICT before you run: what do you expect?
  A) seed 0 and seed 1 finish very close together, because 0.05 rad is tiny; and
     two seed-0 runs may differ slightly (floating point is fuzzy)
  B) seed 0 and seed 1 diverge wildly; and two seed-0 runs ALSO differ, because
     chaos destroys reproducibility
  C) seed 0 and seed 1 diverge wildly (chaos); but two seed-0 runs are BITWISE
     identical — the system is deterministic even though it is unpredictable

Record your answer in PREDICTION below, then run this file.
Estimated learner time: 15 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch3.4-constraints",
    "choices": ["A", "B", "C"],
    "gate_before_run": True,
}

ARTIFACT = Path(__file__).resolve().parents[2] / "constraints.py"


def tip_final(seed: int) -> np.ndarray:
    """Run a full double-pendulum (baumgarte) sim and return the tip's final position."""
    with tempfile.TemporaryDirectory(prefix="z2r-ex3-") as tmp:
        cmd = [sys.executable, str(ARTIFACT), "--seed", str(seed), "--system", "double",
               "--stabilization", "baumgarte", "--no-rerun", "--out", tmp]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        results = json.loads((Path(tmp) / "metrics.json").read_text())["results"]
    return np.array(results["baumgarte"]["tip_final"])


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    tip0, tip1, tip0_again = tip_final(0), tip_final(1), tip_final(0)
    divergence = float(np.linalg.norm(tip0 - tip1))
    reproduce = float(np.linalg.norm(tip0 - tip0_again))
    print(f"seed 0 tip: {np.round(tip0, 4)}")
    print(f"seed 1 tip: {np.round(tip1, 4)}")
    print(f"seed 0 vs seed 1  (0.05 rad apart at t=0):  distance = {divergence:.4f}   <- chaos")
    print(f"seed 0 vs seed 0  (same seed, rerun):        distance = {reproduce:.1e}   <- bitwise determinism")
    print(f"your prediction: {PREDICTION} — deterministic and predictable are NOT the same word.")
