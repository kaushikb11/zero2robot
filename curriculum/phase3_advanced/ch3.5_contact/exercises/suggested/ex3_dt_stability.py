"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch3.5.

Objective tested: the dt-stability CLIFF — the sim artifact you have watched blow
up since ch0.1, now explained. A penalty contact is a stiff spring, and an
explicit integrator can only hold a spring stable while the timestep stays under
dt_crit ~ 2*sqrt(m/k). Cross it and the spring pumps energy faster than it can
bleed off: the body is flung to infinity. The LCP-flavored solve has no spring —
it projects the velocity — so it has no such cliff.

THE EXPERIMENT: run the DROP scene at a timestep well PAST penalty's dt_crit
(dt = 0.03 s, where dt_crit ~ 0.02 s for the default k = 1e4), with BOTH contact
models, and read the phantom energy each one gains (energy_excess, relative to the
drop energy).

PREDICT before you run: at dt past dt_crit, what happens?
  A) both blow up — no explicit contact method survives too large a timestep
  B) the PENALTY body explodes (energy_excess enormous) while the LCP body stays
     bounded (energy_excess ~ 0) — the cliff is the spring's, not contact's
  C) neither blows up — dt_crit is a myth; the body just penetrates a bit more

Record your answer in PREDICTION below, then run this file.
Estimated learner time: 12 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch3.5-contact",
    "choices": ["A", "B", "C"],
    "gate_before_run": True,
}

ARTIFACT = Path(__file__).resolve().parents[2] / "contact.py"


def energy_excess_at_dt(dt: float) -> dict:
    """Run the drop scene (both models) at timestep dt; return per-method energy_excess."""
    with tempfile.TemporaryDirectory(prefix="z2r-ex3-") as tmp:
        cmd = [sys.executable, str(ARTIFACT), "--seed", "0", "--scene", "drop",
               "--dt", str(dt), "--no-rerun", "--out", tmp]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        quality = json.loads((Path(tmp) / "metrics.json").read_text())["quality"]
    return {name: quality[name]["energy_excess"] for name in ("penalty", "lcp")}


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    safe = energy_excess_at_dt(0.002)   # under dt_crit — both fine
    over = energy_excess_at_dt(0.03)    # past dt_crit ~ 0.02 — watch penalty go
    for name in ("penalty", "lcp"):
        print(f"{name:<9} energy_excess:  dt=0.002 -> {safe[name]:>12.4e}   dt=0.030 -> {over[name]:>12.4e}")
    print(f"your prediction: {PREDICTION} — the cliff belongs to the SPRING (penalty), not to contact itself.")
