"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch4 offline primer.

Objective tested: the chapter's headline claim — that offline RL (AWAC) extracts
a BETTER policy than behavior cloning from the SAME fixed, mixed-quality dataset,
by using the reward BC ignores — AND the RL-doctrine reading underneath it (ch2.1
spike, H2): grade the signal ACROSS seeds, not on one run.

THE SETUP. The fixed dataset is a mix: 30% clean scripted-expert episodes, 70%
random-policy junk (the canonical "expert + random" offline mix). BC regresses
the AVERAGE action per state, so the junk drags it down. AWAC fits a Q-function
and reweights that same regression toward the actions the reward says were
above-average.

THE QUESTION. On this mixed dataset, does the offline learner BEAT BC —
higher held-out success rate AND lower mean final distance — on EVERY one of
seeds 0, 1, 2?

PREDICT before you run: (a) yes, offline RL clearly beats BC on all three seeds;
(b) they roughly tie (BC's average is already good enough); (c) BC wins (the
reward signal is too weak to help offline). Write your choice and one sentence of
why in PREDICTION.

Then run this file. It trains BOTH learners on each of seeds 0, 1, 2 (each run is
bit-reproducible on CPU) and prints per-seed success + final distance. The
variance you read is ACROSS seeds — which is why offline RL, like all RL, is
graded on a multi-seed signal.

Estimated learner time: ~4 minutes (three short training runs, each trains BC
and offline together).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause, e.g. "a because ..."

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch4-offline-primer",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase4_capstone/ch4_offline_primer/offline.py"
SEEDS = (0, 1, 2)


def train_seed(seed: int, workdir: Path) -> dict:
    """Train BC + offline (AWAC) on the default mixed dataset for one seed;
    return the metrics dict (deterministic per seed on CPU)."""
    out = workdir / f"seed{seed}"
    subprocess.run(
        [sys.executable, str(ARTIFACT), "--seed", str(seed), "--device", "cpu",
         "--no-rerun", "--out", str(out)],
        check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


def measure(workdir: Path | None = None) -> dict[str, list[float]]:
    """Return per-seed BC and offline success + final distance. Deterministic per
    seed, so this never flakes run-to-run — the spread is ACROSS seeds."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-offline-ex1-"))
    m = [train_seed(seed, workdir) for seed in SEEDS]
    return {
        "bc_success": [r["bc_success_rate"] for r in m],
        "offline_success": [r["offline_success_rate"] for r in m],
        "bc_dist": [r["bc_mean_final_dist"] for r in m],
        "offline_dist": [r["offline_mean_final_dist"] for r in m],
    }


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    r = measure()
    for i, seed in enumerate(SEEDS):
        print(f"seed {seed}:  BC success {r['bc_success'][i]:.2f} (dist {r['bc_dist'][i]:.3f} m)   "
              f"offline {r['offline_success'][i]:.2f} (dist {r['offline_dist'][i]:.3f} m)   "
              f"-> offline {'BEATS' if r['offline_success'][i] > r['bc_success'][i] else 'does NOT beat'} BC")
    bc_m = sum(r["bc_success"]) / len(SEEDS)
    off_m = sum(r["offline_success"]) / len(SEEDS)
    print(f"\nmean success: BC {bc_m:.2f} vs offline {off_m:.2f}  (random baseline dist ~0.176 m). "
          "Did offline win on every seed, or only on average? That multi-seed reading is the point.")
