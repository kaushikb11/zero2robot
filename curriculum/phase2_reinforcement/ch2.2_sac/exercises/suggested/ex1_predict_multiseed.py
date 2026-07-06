"""SUGGESTED exercise candidate (humans promote) — multi-seed predict-then-run, ch2.2.

Objective tested: the chapter's core claim — that SAC's replay + entropy + twin-Q
machinery actually LEARNS to reach on the dense-reward pusher-reach env — AND the
RL-doctrine lesson underneath it (ch2.1 spike, H2): a single training run is
NOISE. You predict a signal that must hold ACROSS seeds, then you run several
seeds and read whether the strong signal (the fingertip closing on the target)
survives seed-to-seed variance.

THE QUESTION. The random baseline leaves the fingertip ~0.176 m from the target;
the scripted IK reacher gets ~0.0001 m. At a reduced training budget
(total_steps below the chapter default), does SAC drive the held-out eval mean
final distance clearly below the random baseline on EVERY one of seeds 0, 1, 2?

PREDICT before you run: (a) yes, all three seeds land well under the random
baseline; (b) it learns on average but at least one seed stalls near random;
(c) the reduced budget is too short and none clearly beat random. Write your
choice and one sentence of why in PREDICTION.

Then run this file. It trains SAC on seeds 0, 1, 2 and prints each seed's eval
mean final distance. Notice these are DETERMINISTIC per seed (the whole pipeline
is seeded) — the variance you read is ACROSS seeds, which is exactly why RL is
graded on a multi-seed signal, not one run.

Estimated learner time: 20 minutes (mostly waiting on three short SAC runs).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause, e.g. "a because ..."

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch2.2-sac",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase2_reinforcement/ch2.2_sac/sac.py"
SEEDS = (0, 1, 2)
EXERCISE_STEPS = 15_000  # reduced from the 30k default so three seeds fit ~5 min


def train_seed(seed: int, workdir: Path) -> float:
    """Train SAC for EXERCISE_STEPS on one seed; return eval mean final distance (m)."""
    out = workdir / f"seed{seed}"
    subprocess.run(
        [sys.executable, str(ARTIFACT), "--seed", str(seed), "--device", "cpu",
         "--total_steps", str(EXERCISE_STEPS), "--no-rerun", "--out", str(out)],
        check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())["mean_eval_final_dist"]


def measure(workdir: Path | None = None) -> dict[str, list[float]]:
    """Return {"sac": [final_dist per seed]}. Deterministic per seed, so this
    never flakes run-to-run — the spread is ACROSS seeds."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-sac-ex1-"))
    return {"sac": [train_seed(seed, workdir) for seed in SEEDS]}


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    dists = measure()["sac"]
    for seed, d in zip(SEEDS, dists):
        print(f"seed {seed}: eval mean final distance {d:.4f} m  "
              f"({'beats' if d < 0.176 else 'AT/above'} random ~0.176 m)")
    print(f"\nmean over seeds: {sum(dists) / len(dists):.4f} m. "
          "Did the strong signal (closing on the target) hold on every seed, or "
          "only on average? That is the multi-seed reading RL forces.")
