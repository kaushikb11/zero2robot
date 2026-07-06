"""SUGGESTED exercise candidate (humans promote) — hyperparameter-investigation, ch2.3.

Objective tested: the THROUGHPUT-vs-GRADIENT-QUALITY tradeoff — the deeper lesson
under the wall-clock cliff. ex1 showed more parallel envs = more env-steps/sec.
So more envs is strictly better, right? Not for LEARNING at a fixed data budget.

THE SETUP. `--total_steps` is a fixed budget of env-steps. Each PPO iteration
consumes num_envs * num_steps of it, so num_iterations = total_steps / (num_envs *
num_steps) — and num_iterations is the number of GRADIENT UPDATES. Double
num_envs at a fixed total_steps and you HALVE the gradient updates.

THE QUESTION. At the default budget (total_steps 300000, num_steps 128), num_envs
64 gives 36 updates and num_envs 256 gives 9 updates. num_envs 256 runs FASTER
(higher throughput, from ex1). Predict: at this fixed env-step budget, does the
256-env run reach (a) a HIGHER eval return (more envs is just better), (b) about
the SAME (throughput is all that matters), or (c) a LOWER eval return (too few
gradient updates to learn)? Write it in PREDICTION.

Then run this file. It trains both configs at seed 0 and prints eval returns.
Measured (2026-07-06, cpu-jax): 64-env solves (eval ~407), 256-env does NOT (eval
~90) — same data, 4x fewer updates. Faster wall-clock throughput bought WORSE
learning. On a GPU you'd raise total_steps to give the 4096-env run enough
updates; the tradeoff never disappears, it just moves.

Estimated learner time: 25 minutes (two short training runs).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause, e.g. "c because ..."

METADATA = {
    "type": "hyperparameter-investigation",
    "chapter": "ch2.3-mjx",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase2_reinforcement/ch2.3_mjx/ppo_mjx.py"
# Fixed total_steps (the default 300000) — only num_envs changes, so this isolates
# "more parallel envs => fewer gradient updates". Forced CPU for reproducibility.
ARMS = {"num_envs_64": ["--num_envs", "64"], "num_envs_256": ["--num_envs", "256"]}
COMMON = ["--platform", "cpu", "--no-rerun"]


def train_arm(flags: list[str], seed: int, workdir: Path) -> float:
    subprocess.run(
        [sys.executable, str(ARTIFACT), "--seed", str(seed), "--out", str(workdir), *COMMON, *flags],
        check=True, capture_output=True, cwd=REPO)
    return json.loads((workdir / "metrics.json").read_text())["mean_eval_return"]


def measure(workdir: Path | None = None, seed: int = 0) -> dict[str, float]:
    """Return {arm_name: eval_return}. Seeded => bit-reproducible on CPU-jax, so
    this never flakes run-to-run; the effect is in the ONE comparison, not noise."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-mjx-ex2-"))
    return {name: train_arm(flags, seed, workdir / name) for name, flags in ARMS.items()}


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    results = measure()
    for name, ev in results.items():
        print(f"{name:14s}  eval return {ev:6.1f}")
    print("\nReconcile: the FASTER config (more envs, higher env-steps/sec) learned "
          "LESS at the same env-step budget — because it took fewer gradient updates. "
          "Throughput is not learning.")
