"""SUGGESTED exercise candidate (humans promote) — investigation, ch4.2.

Objective tested: DAgger is not "more rounds = better." Some round peaks, and
later rounds can REGRESS — which is why Ross et al. return the BEST policy over
rounds, not the last. This exercise makes you SEE that curve, and reason out WHY
the regression happens BEFORE you read the mechanism.

THE INVESTIGATION. `dagger.py` reports `round_rates`: the success rate at BC
(round 0) and after each DAgger round. Run the default config and read the curve.

Questions to answer from the numbers (record them in FINDINGS below):
  1. Which round is the PEAK? Is it the LAST round?
  2. Is the curve monotonic, or does it dip/regress somewhere?
  3. Measured across seeds 0-2 the peak lands at round 3 or 4 — does yours?

There is no single right prediction here; the point is to SEE that the last round
is not a safe default, which is why the artifact saves the best round's
checkpoint, not the final one.

Estimated learner time: 20 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

WHY_IT_REGRESSED = None  # <- BEFORE you run: one sentence — when a still-weak policy's
#    own long failure trajectories get labeled and aggregated, what happens to the
#    regression target the next round fits?
FINDINGS = None  # <- one sentence: which round peaked, and whether the curve was monotonic

METADATA = {"type": "investigation", "chapter": "ch4.2-corrections"}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase4_capstone/ch4.2_corrections/dagger.py"


def measure(workdir: Path | None = None) -> dict:
    """Run dagger.py once (seed 0, cpu, default rounds) and return metrics.json."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-dagger-ex2-"))
    subprocess.run([sys.executable, str(ARTIFACT), "--seed", "0", "--device", "cpu",
                    "--no-rerun", "--out", str(workdir)],
                   check=True, capture_output=True, cwd=REPO)
    return json.loads((workdir / "metrics.json").read_text())


if __name__ == "__main__":
    if WHY_IT_REGRESSED is None:
        raise SystemExit("write WHY_IT_REGRESSED first — reason out the failure before you read it")
    print(f"your hypothesis for the regression: {WHY_IT_REGRESSED}\n")
    m = measure()
    rates = m["round_rates"]
    peak = max(range(len(rates)), key=lambda i: rates[i])
    print("success vs round:")
    for i, r in enumerate(rates):
        tag = "  <- BC" if i == 0 else ("  <- PEAK" if i == peak else "")
        print(f"  round {i}: {r:.3f}{tag}")
    print(f"\npeak round = {peak} (best_round reported: {m['best_round']}); "
          f"last round = {len(rates) - 1} at {rates[-1]:.3f}")
    print("Notice: the peak is not guaranteed to be the last round — that is why you "
          "evaluate every round and keep the best (Ross et al.), not just run longer.")
    print("The mechanism, now that you've committed your guess: aggregating corrections "
          "onto a still-weak reactive policy eventually FLOODS the dataset with its own "
          "long failure trajectories, and the fit regresses — that is the reason Ross et "
          "al. return the best policy over rounds, not the last.")
    if FINDINGS is None:
        print("\n(record FINDINGS in this file to complete the exercise)")
