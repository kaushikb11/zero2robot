"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch2.1.

Objective tested: the "tricks are flags" claim from the chapter, and the harder
RL lesson underneath it — that a single training run is NOISE. You predict, then
you run across several seeds and read the AVERAGE, because one seed can tell you
anything.

THE QUESTION. Advantage normalization (the `--norm-adv` trick, on by default)
standardizes advantages inside each minibatch. Turn it off with `--no-norm-adv`.

PREDICT before you run: over three seeds at the default config, does turning off
advantage normalization (a) always hurt, (b) hurt on average but not every seed,
or (c) not matter? Write your choice and one sentence of why in PREDICTION.

Then run this file. It trains the reference and the ablation on seeds 0,1,2 and
prints both the per-seed returns and the means. Notice how much the per-seed
numbers move — that spread is the whole reason RL is graded on averages, and the
reason this chapter has no single-run "break it" bug (see the chapter's note on
variance).

Estimated learner time: 30 minutes (mostly waiting on six short runs).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause, e.g. "b because ..."

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch2.1-ppo",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase2_reinforcement/ch2.1_ppo/ppo.py"
SEEDS = (0, 1, 2)
# Default config, forced CPU for a reproducible measurement. Each run ~20 s.
COMMON = ["--device", "cpu", "--no-rerun"]
ARMS = {"reference": [], "no-norm-adv": ["--no-norm-adv"]}


def train_arm(flags: list[str], seed: int, workdir: Path) -> float:
    out = workdir / f"seed{seed}"
    subprocess.run([sys.executable, str(ARTIFACT), "--seed", str(seed), "--out", str(out), *COMMON, *flags],
                   check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())["mean_eval_return"]


def measure(workdir: Path | None = None) -> dict[str, list[float]]:
    """Return {arm_name: [eval_return per seed]}. Deterministic: same seeds ->
    same numbers, so this never flakes run-to-run — the variance is ACROSS seeds."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-ppo-ex2-"))
    return {name: [train_arm(flags, seed, workdir / name) for seed in SEEDS]
            for name, flags in ARMS.items()}


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    results = measure()
    for name, evals in results.items():
        mean = sum(evals) / len(evals)
        print(f"{name:12s} per-seed {[round(e) for e in evals]}  mean {mean:6.1f}")
    print("\nReconcile: was the ablation worse on EVERY seed, or only on average? "
          "That spread is why a single-seed 'it broke' claim is not evidence in RL.")
