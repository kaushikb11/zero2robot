"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch4 offline primer.

Objective tested: the chapter's SHARPEST failure — why offline RL needs to be its
own algorithm at all. You already fit a Q-function. So why can't the policy just
MAXIMIZE it (the DDPG/SAC move) and skip the advantage-weighting bookkeeping?
Because offline, with no fresh data to correct it, the policy drifts to actions
the log never contained, where Q is an unchecked extrapolation — and the damage
scales with how NARROW the data is.

THE SETUP. `--naive` swaps AWAC's advantage-weighted regression for plain
maximize-Q policy extraction (no anchor to the dataset action). `--expert_frac`
sets how much of the fixed dataset is the clean scripted expert; the rest is a
RANDOM policy that COVERS the action space. We run naive on TWO datasets:
  narrow  --expert_frac 1.0  (expert-only — the shape of real demo/correction logs)
  broad   --expert_frac 0.3  (the default expert+random mix — random half = coverage)
and read the critic's mean |Q| over the data at the end of training.

THE QUESTION. Under naive maximize-Q, what happens to |Q|?
  (a) it blows up on BOTH — maximize-Q always diverges offline, coverage or not;
  (b) it blows up on the NARROW data but stays bounded on the broad mix — the
      random half covers the action space, so the critic keeps seeing what bad
      actions cost and can't inflate them;
  (c) it stays bounded on BOTH — the twin-Q min (clipped double-Q, ch2.2) already
      caps overestimation, so the constraint is redundant.
Write your choice and one sentence of why in PREDICTION.

Then, BEFORE you run, answer SELF_EXPLANATION in your own words: you HAVE a Q, so
why can't the policy just maximize it — what does the critic report at an action
the log never contained, and what stops a plain maximizer from walking straight
into it? Committing your reasoning first is the whole exercise.

Then run this file. It trains naive offline RL on the narrow and broad datasets
across seeds 0, 1 (each run is bit-reproducible on CPU) and prints each arm's
mean |Q| plus the eval it produces. The blowup you read is seed-robust — which is
why the graded check asserts it over N seeds, not on one run.

Estimated learner time: ~5 minutes (four short training runs).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause, e.g. "b because ..."
SELF_EXPLANATION = None  # <- one sentence, in your own words: you have a Q, so why
#    can't the policy just maximize it? What does the critic report at an action the
#    log never contained, and what (offline) stops the policy walking into it?

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch4-offline-primer",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase4_capstone/ch4_offline_primer/offline.py"
SEEDS = (0, 1)
# arm -> --expert_frac; both run with --naive (maximize-Q, no data anchor).
ARMS = {"narrow": "1.0", "broad": "0.3"}


def train_naive(expert_frac: str, seed: int, workdir: Path) -> dict:
    """Train naive offline RL (maximize-Q) at one dataset quality + seed; return the
    metrics dict. Deterministic per seed on CPU, so this never flakes run-to-run."""
    out = workdir / f"ef{expert_frac}_seed{seed}"
    subprocess.run(
        [sys.executable, str(ARTIFACT), "--seed", str(seed), "--device", "cpu",
         "--naive", "--expert_frac", expert_frac, "--no-rerun", "--out", str(out)],
        check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


def measure(workdir: Path | None = None) -> dict[str, dict[str, list[float]]]:
    """Return {arm: {"abs_q": [...], "offline_success": [...], "offline_dist": [...]}}
    per seed. The spread is ACROSS seeds; each run is bit-reproducible on CPU."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-offline-ex3-"))
    out: dict[str, dict[str, list[float]]] = {}
    for arm, ef in ARMS.items():
        rows = [train_naive(ef, seed, workdir) for seed in SEEDS]
        out[arm] = {"abs_q": [r["offline_mean_abs_q"] for r in rows],
                    "offline_success": [r["offline_success_rate"] for r in rows],
                    "offline_dist": [r["offline_mean_final_dist"] for r in rows]}
    return out


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    if SELF_EXPLANATION is None:
        raise SystemExit("write SELF_EXPLANATION first — commit WHY maximize-Q breaks "
                         "before you watch it break")
    print(f"your prediction: {PREDICTION}")
    print(f"your self-explanation: {SELF_EXPLANATION}\n")
    results = measure()
    for arm, r in results.items():
        q_m = sum(r["abs_q"]) / len(SEEDS)
        off_m = sum(r["offline_success"]) / len(SEEDS)
        dist_m = sum(r["offline_dist"]) / len(SEEDS)
        print(f"{arm:6s} (expert_frac={ARMS[arm]}):  mean |Q| {q_m:6.2f}   "
              f"naive eval: success {off_m:.2f}, dist {dist_m:.3f} m")
    q_narrow = sum(results["narrow"]["abs_q"]) / len(SEEDS)
    q_broad = sum(results["broad"]["abs_q"]) / len(SEEDS)
    print(f"\n|Q| ratio narrow/broad = {q_narrow / q_broad:.1f}x  (random-policy baseline dist ~0.176 m).")
    print("Reconcile: naive maximize-Q inflates |Q| ~9x on the NARROW expert-only log while its "
          "eval collapses toward random — the policy chases out-of-distribution actions the critic "
          "OVERESTIMATES because the log never showed what they cost. On the BROAD mix the random "
          "half COVERS the action space, so the critic stays honest (|Q|~0.8) and even naive survives. "
          "That coverage-dependence is why AWAC's advantage weight — which only ever asks the critic "
          "about actions the data contains — is the fix, and why narrow correction data (4.3's regime) needs it.")
