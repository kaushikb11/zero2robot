"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch3.7.

Objective tested: the chapter's thesis — "data is the policy" (ch1.2), scaled.
You have 12 PushT source demos. MimicGen-style augmentation perturbs each demo's
object/pusher pose, re-solves it with the scripted expert, and keeps the demos
the solver still finishes. Does feeding those extra, physically-valid demos to
the SAME BC policy actually make it succeed more often?

PREDICT before you run. Over the default config, does augmentation:
  (A) HELP — augmented beats source-only,
  (B) HURT — the perturbed demos confuse the policy, or
  (C) NOT MATTER — success is unchanged?
Write your choice and one sentence of why in PREDICTION.

Then run this file. It trains BC twice (source-only, then source+augmented) at
seed 0 and prints both rollout success rates. It also reminds you: one seed is
noise (ch2.1). The DIRECTION (augmented > source) is what holds seed-to-seed;
the exact numbers move, and both are modest — 12 demos is a coverage-starved
regime chosen so the DATA effect is not drowned by a saturated policy.

Now say WHY in one sentence: the policy, weights, and batch order were identical across
the two arms — so what exactly did the extra demos change that lifted success?

Estimated learner time: 30 minutes (mostly waiting on two BC trainings ~1-2 min).
"""

import json
import subprocess
import sys
from pathlib import Path

PREDICTION = None  # <- "A" | "B" | "C", plus a because-clause, e.g. "A because ..."

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch3.7-scale-data",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase3_advanced/ch3.7_scale_data/scale_data.py"


def measure(seed: int, workdir: Path) -> tuple[float, float]:
    """Run scale_data.py at `seed` (forced CPU) and return (source_rate, augmented_rate).
    Deterministic: same seed -> same numbers, so this never flakes run-to-run."""
    out = workdir / f"seed{seed}"
    subprocess.run([sys.executable, str(ARTIFACT), "--seed", str(seed),
                    "--device", "cpu", "--no-rerun", "--out", str(out)],
                   check=True, capture_output=True, cwd=REPO)
    m = json.loads((out / "metrics.json").read_text())
    return m["source_success_rate"], m["augmented_success_rate"]


if __name__ == "__main__":
    import tempfile
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    work = Path(tempfile.mkdtemp(prefix="z2r-ch37-ex1-"))
    for seed in (0, 1, 2):  # three seeds so you can SEE the spread, not just one draw
        source, augmented = measure(seed, work)
        print(f"seed {seed}:  source-only {source:.2f}  ->  source+augmented {augmented:.2f}  "
              f"(delta {augmented - source:+.2f})")
    print("\nReconcile: did augmentation help on EVERY seed, or only on average? "
          "The ordering is the lesson; the exact gain is not.")
