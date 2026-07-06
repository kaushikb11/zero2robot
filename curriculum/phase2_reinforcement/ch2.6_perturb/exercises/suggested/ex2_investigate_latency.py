"""SUGGESTED exercise candidate (humans promote) — hyperparameter investigation, ch2.6.

Objective tested: the MECHANISM behind ex1's finding. If latency is what breaks
the balancer, then WHERE is the cliff? A stable controller has a delay margin —
it survives a little staleness and then, past some threshold, it does not. This
is a hyperparameter investigation (ch2.1 spike, H1): form a directional
hypothesis about the breaking point, then read it against a seed-robust sweep.

THE KNOB. `--latency_steps` delays the observation by N control steps (each 20 ms
at 50 Hz). Probed here in a fine sweep 0..8, with the OTHER perturbations held
clean, so the whole curve is attributable to delay alone.

PREDICT before you run: the balancer's success rate stays near 1.0 up to about
(a) 1 step (20 ms), (b) 4 steps (80 ms), or (c) 8 steps (160 ms), then collapses.
Write your choice and a one-sentence reason in PREDICTION.

Then run this file. It trains PPO on seeds 0, 1, 2 and evaluates each policy at
every latency in the grid, printing the per-seed success curve. Read the MEANS:
the exact breaking step wobbles seed to seed, but the SHAPE — solid, then a knee,
then dead — is the robust fact. Note that this is a cliff you cannot noise your
way across: no amount of sensor filtering buys back a policy that never learned
to act on stale state. That gap is ch2.7's job.

Estimated learner time: 30 minutes (three short PPO trains + fine latency sweeps).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause

METADATA = {
    "type": "hyperparameter-investigation",
    "chapter": "ch2.6-perturb",
    "knob": "--latency_steps (observation staleness)",
}

REPO = Path(__file__).resolve().parents[5]
PPO = REPO / "curriculum/phase2_reinforcement/ch2.1_ppo/ppo.py"
PERTURB = REPO / "curriculum/phase2_reinforcement/ch2.6_perturb/perturb.py"
SEEDS = (0, 1, 2)
LATENCIES = (0, 2, 4, 6, 8)  # control steps; each 20 ms at 50 Hz


def probe_seed(seed: int, workdir: Path) -> list[float]:
    """Train PPO once, then eval it at each latency (single-point --no-sweep runs
    with the other perturbations clean). Returns success rate per latency."""
    out = workdir / f"seed{seed}"
    ckpt = out / "ppo_agent.pt"
    subprocess.run([sys.executable, str(PPO), "--seed", str(seed), "--device", "cpu",
                    "--no-rerun", "--out", str(out)], check=True, capture_output=True, cwd=REPO)
    successes = []
    for lat in LATENCIES:
        pout = out / f"lat{lat}"
        subprocess.run([sys.executable, str(PERTURB), "--seed", str(seed), "--device", "cpu",
                        "--task", "cartpole", "--ckpt", str(ckpt), "--no-sweep",
                        "--latency_steps", str(lat), "--no-rerun", "--out", str(pout)],
                       check=True, capture_output=True, cwd=REPO)
        successes.append(json.loads((pout / "metrics.json").read_text())["single"]["success_rate"])
    return successes


def measure(workdir: Path | None = None) -> dict[str, list]:
    """Return {"latencies": [...], "success_by_seed": [[per-latency] per seed]}."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-perturb-ex2-"))
    return {"latencies": list(LATENCIES),
            "success_by_seed": [probe_seed(seed, workdir) for seed in SEEDS]}


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    r = measure()
    print("latency (steps):  " + "  ".join(f"{lat:>4d}" for lat in r["latencies"]))
    for seed, curve in zip(SEEDS, r["success_by_seed"]):
        print(f"seed {seed} success:  " + "  ".join(f"{s:>4.2f}" for s in curve))
    print("\nWhere is the knee? The exact step moves with the seed; the shape "
          "(solid -> knee -> dead) does not. That margin is the whole story.")
