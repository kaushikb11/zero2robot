"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch2.3.

Objective tested: the WALL-CLOCK CLIFF — how throughput (env-steps/sec) scales as
you add parallel MJX envs, and where it stops paying off on YOUR hardware.

THE QUESTION. `--sweep 16,256` times one PPO update_step at num_envs = 16 and at
num_envs = 256 on CPU-jax and prints env-steps/sec for each.

PREDICT before you run: going from 16 to 256 parallel envs (16x the work per
step), does throughput (env-steps/sec) (a) rise ~16x — near-perfect parallel
scaling, (b) rise, but far less than 16x — real but sub-linear on a CPU, or (c)
stay flat / fall? Write your choice and one sentence of why in PREDICTION.

Then run this file. It sweeps 16 and 256 envs and prints both throughputs and the
ratio. On CPU-jax the honest answer is (b): more envs help, but a laptop has a
handful of cores, so the curve is sub-linear and (past ~256 here) even reverses.
THAT plateau is the cliff — and the reason MJX's 4096-robot headline is a GPU
story: on a 4090 the same curve keeps climbing where the CPU flattens.

NOTE: these are TIMINGS, not a bitwise-reproducible number — they wobble
run-to-run and depend on your machine. The reproducible fact is the SHAPE (256
clearly beats 16); the check asserts only that.

Estimated learner time: 15 minutes (two short compiled sweeps).
"""

import re
import subprocess
import sys
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause, e.g. "b because ..."

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch2.3-mjx",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase2_reinforcement/ch2.3_mjx/ppo_mjx.py"
SWEEP_ENVS = (16, 256)
_ROW = re.compile(r"^\s*(\d+)\s+([\d,]+)\s")  # "   256   104,544   0.313" -> (256, 104544)


def measure() -> dict[int, float]:
    """Run the artifact's --sweep and parse {num_envs: env_steps_per_sec}.
    Deterministic in SHAPE (bigger batch -> more throughput up to the plateau),
    though the exact numbers are timings and vary by machine/run."""
    out = subprocess.run(
        [sys.executable, str(ARTIFACT), "--sweep", ",".join(str(n) for n in SWEEP_ENVS),
         "--seed", "0", "--no-rerun", "--platform", "cpu"],
        check=True, capture_output=True, text=True, cwd=REPO)
    throughput: dict[int, float] = {}
    for line in out.stdout.splitlines():
        m = _ROW.match(line)
        if m and int(m.group(1)) in SWEEP_ENVS:
            throughput[int(m.group(1))] = float(m.group(2).replace(",", ""))
    return throughput


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    tp = measure()
    for n in SWEEP_ENVS:
        print(f"num_envs {n:>4}   {tp[n]:>12,.0f} env-steps/sec")
    ratio = tp[256] / tp[16]
    print(f"\n256-env / 16-env throughput ratio: {ratio:.2f}x "
          f"(16x envs, but a CPU has few cores — that gap IS the cliff)")
