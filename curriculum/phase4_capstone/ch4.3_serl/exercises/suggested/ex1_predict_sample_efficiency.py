"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch4.3 HIL-SERL.

Objective tested: the chapter's headline — SAMPLE EFFICIENCY. HIL-SERL starts
from an offline prior built from the corrections, so it should clear the reach
THRESHOLD in FAR FEWER online environment samples than SAC-from-scratch, which has
to discover the reach from nothing. And the RL-doctrine reading underneath (ch2.1
spike, H2): grade the signal ACROSS seeds, not on one run.

THE SETUP. Both arms run the SAME online SAC loop on pusher_reach. HIL-SERL starts
warm — actor + critics pre-trained on the corrections (AWAC), corrections
pre-loaded in the replay. From-scratch starts cold — fresh nets, empty replay,
random warmup. We measure samples-to-threshold: the first online env step at which
the eval mean final distance drops below --threshold (0.10 m; random ~0.176 m).

THE QUESTION. Does HIL-SERL reach the threshold in fewer online samples than
from-scratch SAC, on EVERY one of seeds 0, 1, 2?

PREDICT before you run: (a) yes, HIL-SERL clears the bar with far fewer online
samples on all three seeds (the corrections-as-prior are a head start); (b) they
need about the same (the prior does not really help); (c) from-scratch gets there
first. Write your choice and one sentence of why in PREDICTION.

Then run this file. It runs the full artifact on each of seeds 0, 1, 2 (each run
is bit-reproducible on CPU) and prints per-seed samples-to-threshold for both arms.
The variance you read is ACROSS seeds — which is why HIL-SERL, like all RL, is
graded on a multi-seed signal.

Estimated learner time: ~10 minutes (three full runs; each trains the prior, the
HIL-SERL fine-tune, and the from-scratch baseline).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause, e.g. "a because ..."

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch4.3-serl",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase4_capstone/ch4.3_serl/serl.py"
SEEDS = (0, 1, 2)


def run_seed(seed: int, workdir: Path) -> dict:
    """Run the full HIL-SERL vs from-scratch pipeline for one seed; return metrics
    (deterministic per seed on CPU)."""
    out = workdir / f"seed{seed}"
    subprocess.run(
        [sys.executable, str(ARTIFACT), "--seed", str(seed), "--device", "cpu",
         "--no-rerun", "--out", str(out)],
        check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


def measure(workdir: Path | None = None) -> dict[str, list]:
    """Return per-seed samples-to-threshold and success for both arms. Deterministic
    per seed, so this never flakes run-to-run — the spread is ACROSS seeds."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-serl-ex1-"))
    m = [run_seed(seed, workdir) for seed in SEEDS]
    return {
        "hil_sts": [r["hil_steps_to_threshold"] for r in m],
        "scratch_sts": [r["scratch_steps_to_threshold"] for r in m],
        "hil_success": [r["hil_success_rate"] for r in m],
        "scratch_success": [r["scratch_success_rate"] for r in m],
        "scratch_steps": [r["scratch_curve"][-1][0] for r in m],  # the online budget scratch got
    }


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    r = measure()
    for i, seed in enumerate(SEEDS):
        hil, scr, budget = r["hil_sts"][i], r["scratch_sts"][i], r["scratch_steps"][i]
        scr_str = str(scr) if scr is not None else f">{budget} (never reached)"
        print(f"seed {seed}:  HIL-SERL reaches threshold at {hil} online samples   "
              f"scratch at {scr_str}   -> HIL-SERL is "
              f"{'far more' if scr is None or (hil is not None and scr > hil) else 'NOT more'} sample-efficient")
    print("\nThe headline is the SAMPLE AXIS: HIL-SERL clears the bar almost immediately (the "
          "corrections-as-prior put it there), while scratch needs thousands of online samples. "
          "Did HIL-SERL win on every seed?")
