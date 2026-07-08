"""SUGGESTED exercise candidate (humans promote) — investigation, ch4.3 HIL-SERL.

Objective tested: WHERE the sample efficiency comes from. The chapter's measured
claim is that the CORRECTIONS-AS-PRIOR do the heavy lifting — so the amount of
correction data should move the prior's quality, and with it HIL-SERL's head
start. This is an investigation: you change ONE knob (how many correction
episodes feed the prior) and read the effect on the offline prior and on
HIL-SERL's samples-to-threshold.

THE KNOB. --corr_episodes controls how many scripted-expert correction episodes
build the prior. We compare a STARVED prior (few corrections) against the DEFAULT
(many). More corrections -> a better-covered offline prior -> it should clear the
threshold from closer to zero online samples.

PREDICT before you run: with far fewer corrections, does the offline prior (a)
get clearly worse (higher eval distance, later or no threshold crossing), (b)
stay about the same (the reach generalizes from a handful of episodes), or (c)
get better? Write your choice + one sentence in PREDICTION.

Then run this file. It runs the artifact at two correction budgets on seeds 0, 1
and prints the prior's eval distance and HIL-SERL's samples-to-threshold for each.

RL-doctrine note (ch2.1 spike): the seed-robust, gated claim is only that the
DEFAULT (full) corrections build a prior that clears the threshold. The direction
and size of the starvation effect is left for YOU to read and interpret — a
free-tier trend, not a guaranteed-monotone law.

Estimated learner time: ~12 minutes (four runs).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause
RECONCILE = None  # <- AFTER you read the numbers: one sentence tracing which step in the
#    pipeline (critic fit -> advantage weight -> replay contents) the missing
#    correction episodes starved.

METADATA = {
    "type": "hyperparameter-investigation",
    "chapter": "ch4.3-serl",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase4_capstone/ch4.3_serl/serl.py"
SEEDS = (0, 1)
CORR_EPISODES = {"starved": 15, "default": 60}


def run(seed: int, corr_episodes: int, workdir: Path) -> dict:
    out = workdir / f"c{corr_episodes}_s{seed}"
    subprocess.run(
        [sys.executable, str(ARTIFACT), "--seed", str(seed), "--device", "cpu",
         "--no-rerun", "--corr_episodes", str(corr_episodes), "--out", str(out)],
        check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


def measure(workdir: Path | None = None) -> dict[str, dict]:
    """For each correction budget, per-seed prior eval distance + HIL samples-to-
    threshold. Deterministic per seed on CPU."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-serl-ex2-"))
    results = {}
    for arm, ce in CORR_EPISODES.items():
        m = [run(seed, ce, workdir) for seed in SEEDS]
        results[arm] = {
            "prior_dist": [r["prior_eval_dist"] for r in m],
            "prior_success": [r["prior_success_rate"] for r in m],
            "hil_sts": [r["hil_steps_to_threshold"] for r in m],
        }
    return results


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    res = measure()
    for arm in ("starved", "default"):
        r = res[arm]
        print(f"{arm:8s} ({CORR_EPISODES[arm]} corr ep):  prior_dist {r['prior_dist']}  "
              f"prior_success {r['prior_success']}  HIL samples-to-threshold {r['hil_sts']}")
    print("\nRead it: did starving the corrections weaken the prior and delay HIL-SERL's threshold "
          "crossing? The corrections are the prior; the prior is the sample efficiency.")
    if RECONCILE is None:
        print("\n(record RECONCILE in this file: trace WHICH pipeline step the missing "
              "corrections starved — critic fit, advantage weight, or replay contents)")
