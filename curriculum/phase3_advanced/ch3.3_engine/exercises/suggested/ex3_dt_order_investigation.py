"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch3.3.

Objective tested: an integrator's ORDER of accuracy. "Order p" means halving the
timestep dt shrinks the per-run error by about 2^p. Explicit Euler is first
order (p=1); RK4 is fourth order (p=4). This experiment makes you SEE the
exponents, not just read them.

THE EXPERIMENT: integrate the same orbit for the same 4 s of sim time, once at
dt = 0.01 (400 steps) and once at dt = 0.005 (800 steps), and compare each
integrator's final energy drift at the two timesteps.

PREDICT before you run: when you HALVE dt, what happens to the final drift?
  A) both Euler's and RK4's drift roughly halve
  B) neither changes — drift is set by the system, not the timestep
  C) Euler's drift roughly HALVES (~2x, first order) while RK4's drops far more
     steeply (toward ~16x, fourth order)

Record your answer in PREDICTION below, then run this file.

Before you run: in one sentence, write WHY halving dt should shrink RK4's error far more
steeply than Euler's.

Estimated learner time: 15 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch3.3-engine",
    "choices": ["A", "B", "C"],
    "gate_before_run": True,
}

ARTIFACT = Path(__file__).resolve().parents[2] / "engine.py"


def drift_at(dt: float, steps: int) -> dict:
    """Final relative energy drift per integrator for one (dt, steps) orbit run."""
    with tempfile.TemporaryDirectory(prefix="z2r-ex3-") as tmp:
        cmd = [sys.executable, str(ARTIFACT), "--seed", "0", "--no-rerun",
               "--system", "orbit", "--dt", str(dt), "--steps", str(steps), "--out", tmp]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        drift = json.loads((Path(tmp) / "metrics.json").read_text())["drift"]
    return {name: abs(drift[name]["rel_final"]) for name in ("euler", "semi_implicit", "rk4")}


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    coarse = drift_at(0.01, 400)   # dt
    fine = drift_at(0.005, 800)    # dt / 2, same 4 s of sim time
    print(f"{'integrator':<14}{'drift @ dt':>16}{'drift @ dt/2':>16}{'shrink factor':>16}")
    for name in ("euler", "semi_implicit", "rk4"):
        factor = coarse[name] / fine[name] if fine[name] > 0 else float("inf")
        print(f"{name:<14}{coarse[name]:>16.4e}{fine[name]:>16.4e}{factor:>16.1f}x")
    print(f"your prediction: {PREDICTION} — match each shrink factor to the integrator's order (2^p).")
