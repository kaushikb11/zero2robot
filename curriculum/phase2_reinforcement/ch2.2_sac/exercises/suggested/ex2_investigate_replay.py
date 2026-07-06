"""SUGGESTED exercise candidate (humans promote) — hyperparameter investigation, ch2.2.

Objective tested: the mechanism BEHIND the off-policy bargain. SAC is
sample-efficient because a big replay buffer lets it REUSE each transition across
many gradient steps. So what happens to the bargain when you starve the buffer?
This is a hyperparameter investigation, not a single bug-hunt (ch2.1 spike, H1):
you form a directional hypothesis and read it against a seed-robust signal.

THE KNOB. `--buffer_size` (default 100000) is the replay capacity. Shrink it hard
(here: 2000, ~20 episodes) and the buffer overwrites old experience fast — the
learner sees a narrow, recent, near-on-policy slice of data instead of the whole
history.

PREDICT before you run: relative to the default buffer, does a 20x-smaller
replay (a) clearly slow or break learning (final distance stays higher), (b)
make little measurable difference at this budget, or (c) HELP (recent data is
fresher)? Write your choice and a one-sentence mechanism in PREDICTION.

Then run this file. It trains the default buffer and the shrunk buffer on seeds
0, 1, 2 and prints each arm's per-seed eval final distance and mean. Read the
MEANS: the per-seed numbers move, and a real hyperparameter effect has to show up
in the average, not in one cherry-picked seed (the reason the graded check asserts
the default's strong learns-signal over seeds, and treats the buffer effect as an
observation you interpret).

Estimated learner time: 30 minutes (six short SAC runs).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause

METADATA = {
    "type": "hyperparameter-investigation",
    "chapter": "ch2.2-sac",
    "knob": "--buffer_size (replay capacity)",
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase2_reinforcement/ch2.2_sac/sac.py"
SEEDS = (0, 1, 2)
EXERCISE_STEPS = 15_000
ARMS = {"default_buffer": ["--buffer_size", "100000"],
        "small_buffer": ["--buffer_size", "2000"]}


def train_arm(flags: list[str], seed: int, workdir: Path) -> float:
    out = workdir / f"seed{seed}"
    subprocess.run(
        [sys.executable, str(ARTIFACT), "--seed", str(seed), "--device", "cpu",
         "--total_steps", str(EXERCISE_STEPS), "--no-rerun", "--out", str(out), *flags],
        check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())["mean_eval_final_dist"]


def measure(workdir: Path | None = None) -> dict[str, list[float]]:
    """Return {arm_name: [eval final distance per seed]}. Deterministic per seed."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-sac-ex2-"))
    return {name: [train_arm(flags, seed, workdir / name) for seed in SEEDS]
            for name, flags in ARMS.items()}


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    results = measure()
    for name, dists in results.items():
        mean = sum(dists) / len(dists)
        print(f"{name:15s} per-seed {[round(d, 4) for d in dists]}  mean {mean:.4f} m")
    print("\nReconcile: did shrinking the replay move the MEAN final distance, and "
          "in the direction you predicted? If the effect is small at this budget, "
          "that is itself a finding — say so honestly.")
