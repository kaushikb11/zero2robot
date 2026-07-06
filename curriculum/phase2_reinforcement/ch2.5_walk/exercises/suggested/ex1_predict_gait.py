"""SUGGESTED exercise candidate (humans promote) — multi-seed predict-then-run, ch2.5.

Objective tested: the chapter's core claim — that a walking GAIT EMERGES from the
reward, with nobody scripting it — AND the RL-doctrine lesson underneath (ch2.1
spike, H2): a single training run is NOISE. You predict a signal that must hold
ACROSS seeds, then train several seeds and read whether the emergent gait (the
torso actually traveling forward) survives seed-to-seed variance.

THE QUESTION. A standing robot goes nowhere (~ -0.01 m of forward travel); a
random policy flails backward (~ -0.30 m); the hand-scripted trot walks +2.14 m.
At a reduced training budget (total_steps below the chapter default), does SAC
drive the held-out eval forward distance CLEARLY POSITIVE — a real gait, well past
the standing baseline — on EVERY one of seeds 0, 1, 2?

PREDICT before you run: (a) yes, all three seeds walk clearly forward (a gait
emerges every time); (b) it moves forward on average but at least one seed stalls
near the stand; (c) the reduced budget is too short and none clearly beat the
stand. Write your choice and one sentence of why in PREDICTION.

Then run this file. It trains walk.py on seeds 0, 1, 2 and prints each seed's eval
forward distance. These are DETERMINISTIC per seed (the whole pipeline is seeded)
— the variance you read is ACROSS seeds, which is exactly why gait emergence is
graded on a multi-seed signal, not one run. Note the honest ceiling: even a clear
emergent gait at this budget stays well short of the +2.14 m scripted trot.

Estimated learner time: 25 minutes (mostly waiting on three short SAC runs).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause, e.g. "a because ..."

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch2.5-walk",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase2_reinforcement/ch2.5_walk/walk.py"
SEEDS = (0, 1, 2)
EXERCISE_STEPS = 25_000  # reduced from the 60k chapter default so three seeds fit the budget


def train_seed(seed: int, workdir: Path) -> float:
    """Train walk.py for EXERCISE_STEPS on one seed; return eval mean forward distance (m)."""
    out = workdir / f"seed{seed}"
    subprocess.run(
        [sys.executable, str(ARTIFACT), "--seed", str(seed), "--device", "cpu",
         "--total_steps", str(EXERCISE_STEPS), "--no-rerun", "--out", str(out)],
        check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())["mean_eval_forward_dist"]


def measure(workdir: Path | None = None) -> dict[str, list[float]]:
    """Return {"walk": [forward_dist per seed]}. Deterministic per seed, so this
    never flakes run-to-run — the spread is ACROSS seeds."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-walk-ex1-"))
    return {"walk": [train_seed(seed, workdir) for seed in SEEDS]}


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    dists = measure()["walk"]
    for seed, d in zip(SEEDS, dists):
        print(f"seed {seed}: eval forward distance {d:+.3f} m  "
              f"({'walks' if d > 0.1 else 'STALLED near stand'}; stand ~-0.01 m, trot +2.14 m)")
    print(f"\nmean over seeds: {sum(dists) / len(dists):+.3f} m. Did a gait emerge "
          "on every seed, or only on average? And how far below the +2.14 m "
          "scripted trot did the emergent gait land? That gap is the honest cost "
          "of a free-tier training budget.")
