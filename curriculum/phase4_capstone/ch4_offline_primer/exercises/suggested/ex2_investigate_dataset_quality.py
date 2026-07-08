"""SUGGESTED exercise candidate (humans promote) — dataset-quality investigation, ch4 offline primer.

Objective tested: the MECHANISM behind the headline. Offline RL beats BC because
BC is capped by the AVERAGE quality of its data — it clones the mix. So what
happens to the BC-vs-offline gap as you clean up the dataset? This is a
hyperparameter investigation, not a bug-hunt (ch2.1 spike, H1): form a directional
hypothesis, then read it against a seed-robust signal.

THE KNOB. `--expert_frac` sets the fraction of dataset episodes that come from
the clean scripted expert (the rest are random junk). We compare a mostly-junk
dataset (0.15) against a mostly-clean one (0.6).

PREDICT before you run: as the data gets cleaner (expert_frac 0.15 -> 0.6), does
BC (a) catch up to offline RL — a cleaner average is a better clone; (b) stay far
behind — averaging ANY junk into the regression target corrupts it, so the ratio
barely matters; (c) overtake offline RL? Write your choice and a one-sentence
mechanism in PREDICTION.

Then run this file. It trains BC + offline on both dataset qualities across seeds
0, 1 and prints each arm's mean success. Read the MEANS, not one cherry-picked seed:
commit your PREDICTION first, then let the per-arm means show you whether cleaning the
data closed the BC-vs-offline gap — and reconcile the result against your mechanism.

Estimated learner time: ~5 minutes (four short training runs).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause
PREDICTION_BOUNDARY = None  # <- before you rerun at --expert_frac 1.0: one sentence —
#    with no junk left in the log, what is there for AWAC's advantage weight to
#    down-weight that BC's plain average doesn't already get right?

METADATA = {
    "type": "hyperparameter-investigation",
    "chapter": "ch4-offline-primer",
    "knob": "--expert_frac (dataset quality: fraction of clean-expert episodes)",
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase4_capstone/ch4_offline_primer/offline.py"
SEEDS = (0, 1)
ARMS = {"mostly_junk": "0.15", "mostly_clean": "0.6"}


def train_arm(expert_frac: str, seed: int, workdir: Path) -> dict:
    out = workdir / f"ef{expert_frac}_seed{seed}"
    subprocess.run(
        [sys.executable, str(ARTIFACT), "--seed", str(seed), "--device", "cpu",
         "--expert_frac", expert_frac, "--no-rerun", "--out", str(out)],
        check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


def measure(workdir: Path | None = None) -> dict[str, dict[str, list[float]]]:
    """Return {arm: {"bc": [...], "offline": [...]}} success per seed."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-offline-ex2-"))
    out: dict[str, dict[str, list[float]]] = {}
    for arm, ef in ARMS.items():
        rows = [train_arm(ef, seed, workdir) for seed in SEEDS]
        out[arm] = {"bc": [r["bc_success_rate"] for r in rows],
                    "offline": [r["offline_success_rate"] for r in rows]}
    return out


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    results = measure()
    for arm, r in results.items():
        bc_m = sum(r["bc"]) / len(SEEDS)
        off_m = sum(r["offline"]) / len(SEEDS)
        print(f"{arm:12s} (expert_frac={ARMS[arm]}):  BC mean {bc_m:.2f}  offline mean {off_m:.2f}  "
              f"gap {off_m - bc_m:+.2f}")
    print("\nReconcile: did cleaning the data close the BC-vs-offline gap? Across "
          "these two arms BC stays capped while offline stays ahead — that IS the "
          "mechanism, because as long as ANY junk is in the log BC averages it in "
          "and cannot tell a good action from a bad one.")
    if PREDICTION_BOUNDARY is None:
        print("\nNow push the knob to its limit yourself: rerun with --expert_frac 1.0 "
              "(a CLEAN expert-only log). Before you do, write PREDICTION_BOUNDARY in this "
              "file — commit your reasoning first, then run it and grade yourself against "
              "the reveal.")
    else:
        print(f"\nyour boundary prediction: {PREDICTION_BOUNDARY}")
        print("Now push the knob to its limit yourself: rerun with --expert_frac 1.0 "
              "(a CLEAN expert-only log). BC jumps to ~0.2 success and offline's edge "
              "falls INSIDE the difference-CI band — with no junk left to down-weight, "
              "AWAC has nothing to reweight and cloning is already right. That boundary "
              "is the honest headline: offline RL beats BC WHEN the data carries "
              "suboptimal actions worth the reward's attention, not unconditionally.")
