"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch3.5.

Objective tested: contact quality as the honesty metric. The table is a one-sided
constraint — it may only PUSH a body up, never pull it down, and only while they
touch. A PENALTY contact fakes that with a stiff spring on penetration depth; an
LCP-FLAVORED contact solves the complementarity for the exact contact impulse.
The SHAPE of what each one gets wrong is the whole story.

THE EXPERIMENT: drop a ball onto the table and integrate for the smoke horizon
(`contact.py --smoke`), once with the PENALTY contact and once with the LCP one,
and read how deep the ball drives in on impact (max penetration, as a fraction of
its radius) and how deep it sits once at REST (rest penetration).

PREDICT before you run: what does the PENALTY contact do that the LCP one does not?
  A) nothing different — both hold the ball exactly on the surface; a contact is a
     contact
  B) the penalty ball bounces forever but never penetrates; the lcp ball penetrates
     but settles
  C) the penalty ball drives much DEEPER on impact AND comes to rest INSIDE the
     table (a static sink of mg/k); the lcp ball holds on the surface

Record your answer in PREDICTION below, then run this file.

Before you run, write one sentence: WHY — what must a stiff spring compress to in order
to hold a weight mg up, and where does that leave the ball resting?

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
    "chapter": "ch3.5-contact",
    "choices": ["A", "B", "C"],
    "gate_before_run": True,
}

ARTIFACT = Path(__file__).resolve().parents[2] / "contact.py"


def run_quality(*extra: str) -> dict:
    """Run contact.py to a temp dir and return its per-method quality metrics."""
    with tempfile.TemporaryDirectory(prefix="z2r-ex1-") as tmp:
        cmd = [sys.executable, str(ARTIFACT), "--seed", "0", "--no-rerun", "--out", tmp, *extra]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return json.loads((Path(tmp) / "metrics.json").read_text())["quality"]


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    quality = run_quality("--smoke")
    for name in ("penalty", "lcp"):
        q = quality[name]
        print(f"{name:<9} max penetration = {q['max_penetration_frac']:>8.4f} r   "
              f"rest penetration = {q['rest_penetration_frac']:>8.4f} r")
    ratio = quality["penalty"]["max_penetration_frac"] / quality["lcp"]["max_penetration_frac"]
    print(f"penalty drives {ratio:.1f}x deeper on impact; and note which one rests > 0 (inside the table).")
    print(f"your prediction: {PREDICTION} — now say WHY a stiff spring must sink to mg/k to hold up a weight.")
