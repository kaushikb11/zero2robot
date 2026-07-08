"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch3.4.

Objective tested: constraint drift as the honesty metric. A distance constraint
says a link's length never changes; a naive acceleration-level solve enforces
that only on the SECOND derivative, so tiny per-step errors in the length itself
accumulate. The SHAPE of that accumulation is the whole story.

THE EXPERIMENT: release a double pendulum and integrate it for the smoke horizon
(`constraints.py --smoke`), once WITHOUT stabilization and once WITH Baumgarte,
and read the worst link-length error each run reaches.

PREDICT before you run: WITHOUT stabilization, what does the maximum link-length
error do over the run?
  A) it stays ~0 — enforcing the acceleration-level constraint keeps the length
     exact for free
  B) it GROWS steadily — the links visibly stretch, the pendulum coming apart in
     the arithmetic
  C) it oscillates but stays inside a fixed bounded band, like ch3.3's symplectic
     energy

Record your answer in PREDICTION below, then run this file.

Before you run: in one sentence, write WHY enforcing only the acceleration condition
g̈=0 lets the length error g itself drift.

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
    "chapter": "ch3.4-constraints",
    "choices": ["A", "B", "C"],
    "gate_before_run": True,
}

ARTIFACT = Path(__file__).resolve().parents[2] / "constraints.py"


def run_results(*extra: str) -> dict:
    """Run constraints.py to a temp dir and return its per-stabilization results."""
    with tempfile.TemporaryDirectory(prefix="z2r-ex1-") as tmp:
        cmd = [sys.executable, str(ARTIFACT), "--seed", "0", "--no-rerun", "--out", tmp, *extra]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return json.loads((Path(tmp) / "metrics.json").read_text())["results"]


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    results = run_results("--smoke")
    for name in ("none", "baumgarte"):
        r = results[name]
        grows = r["final_violation"] > 0.5 * r["max_violation"]  # near its worst at the END == still climbing
        tag = "STILL CLIMBING at the end" if grows and name == "none" else "held / bounded"
        print(f"{name:<11} max |len err| = {r['max_violation']:>10.4e}   final = {r['final_violation']:>10.4e}   ({tag})")
    print(f"your prediction: {PREDICTION} — now say WHY the acceleration-level solve lets the LENGTH itself drift.")
