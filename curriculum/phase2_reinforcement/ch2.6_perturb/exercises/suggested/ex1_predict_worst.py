"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch2.6.

Objective tested: the chapter's headline claim — that a policy trained in a clean
sim is brittle to the reality gap, and NOT uniformly. Each of the three
perturbations (sensor noise, observation latency, model/mass/gravity mismatch)
costs something DIFFERENT, and the whole skill is knowing which one to fear for a
given task. Underneath sits the RL-doctrine lesson (ch2.1 spike, H1/H2): perturbed
eval is noisy, so you predict a STRUCTURAL fact and check it holds across seeds,
not a single dramatic run.

THE QUESTION. Train the ch2.1 PPO balancer to its clean-sim 500/500, then sweep
all three perturbations. Which one degrades it MOST — pushing its success rate
down the furthest?

PREDICT before you run: (a) sensor noise — a jittery pole angle throws off the
push; (b) latency — acting on a stale observation of an unstable system; (c)
model mismatch — heavier/lighter gravity than it trained on. Write your choice
and one sentence of mechanism in PREDICTION.

Then run this file. It trains PPO on seeds 0, 1, 2 and sweeps each seed's policy.
Read whether the SAME perturbation wins on every seed — that agreement is what
makes "latency is the thing to fear here" a claim and not a coincidence. (Spoiler
you should still predict against: balance is a stability problem, and stability
problems die on delay.)

Estimated learner time: 25 minutes (three short PPO trains + sweeps).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause, e.g. "b because ..."

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch2.6-perturb",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
PPO = REPO / "curriculum/phase2_reinforcement/ch2.1_ppo/ppo.py"
PERTURB = REPO / "curriculum/phase2_reinforcement/ch2.6_perturb/perturb.py"
SEEDS = (0, 1, 2)


def sweep_seed(seed: int, workdir: Path) -> dict:
    """Train PPO (clean sim) then sweep the three perturbations; return the
    parsed metrics.json. Deterministic per seed (both scripts are fully seeded)."""
    out = workdir / f"seed{seed}"
    ckpt = out / "ppo_agent.pt"
    subprocess.run([sys.executable, str(PPO), "--seed", str(seed), "--device", "cpu",
                    "--no-rerun", "--out", str(out)], check=True, capture_output=True, cwd=REPO)
    pout = out / "perturb"
    subprocess.run([sys.executable, str(PERTURB), "--seed", str(seed), "--device", "cpu",
                    "--task", "cartpole", "--ckpt", str(ckpt), "--no-rerun", "--out", str(pout)],
                   check=True, capture_output=True, cwd=REPO)
    return json.loads((pout / "metrics.json").read_text())


def measure(workdir: Path | None = None) -> dict[str, list]:
    """Return per-seed {worst, baseline_success, latency_extreme_success}. The
    'extreme' point is the last (largest) magnitude in each family's grid."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-perturb-ex1-"))
    worst, baseline, latency8 = [], [], []
    for seed in SEEDS:
        m = sweep_seed(seed, workdir)
        worst.append(m["worst_perturbation"])
        baseline.append(m["baseline"]["success_rate"])
        latency8.append(m["sweeps"]["latency"]["points"][-1][1])  # success at max latency
    return {"worst": worst, "baseline_success": baseline, "latency_extreme_success": latency8}


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    r = measure()
    for seed, w, b, l8 in zip(SEEDS, r["worst"], r["baseline_success"], r["latency_extreme_success"]):
        print(f"seed {seed}: clean success {b:.2f}  worst perturbation = {w:<13s}  "
              f"success at max latency {l8:.2f}")
    print("\nDid the SAME perturbation win on all three seeds? If so, that is a real "
          "property of the task, not a fluke — and the reason ch2.7 randomizes the gap.")
