"""SUGGESTED exercise candidate (humans promote) — hyperparameter-investigation, ch3.7.

Objective tested: HOW MUCH augmentation, and what the returns look like. ex1
asked *whether* augmentation helps; here you turn the knob `--aug_per_demo`
(re-solved variants generated per source demo) and watch the success rate as the
augmented set grows. A real data engine (MimicGen) faces exactly this question:
generate 10x? 100x? — and where the curve bends is where you stop paying.

PREDICT before you run. As `--aug_per_demo` goes 0 -> 4 -> 8, does success:
  (A) keep CLIMBING across the whole range (you are coverage-starved, so every
      batch of valid demos still buys coverage you did not have),
  (B) PLATEAU within this range (coverage fills fast, so 4->8 buys much less
      than 0->4), or
  (C) FALL (the augmented demos drown the 12 originals)?
Write your choice and one sentence of why in PREDICTION.

Then run this file. It trains BC at aug_per_demo in {0, 4, 8} (seed 0, CPU) and
prints the success rate for each. aug_per_demo 0 is the source-only arm exactly
(no demo survives to add). Watch the SHAPE — the size of each step, not just the
endpoints.

Estimated learner time: 35 minutes (three BC-pair trainings; ~1-2 min each).
"""

import json
import subprocess
import sys
from pathlib import Path

PREDICTION = None  # <- "A" | "B" | "C", plus a because-clause

METADATA = {
    "type": "hyperparameter-investigation",
    "chapter": "ch3.7-scale-data",
    "knob": "--aug_per_demo",
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase3_advanced/ch3.7_scale_data/scale_data.py"
AUG_AMOUNTS = (0, 4, 8)


def success_at(aug_per_demo: int, workdir: Path) -> float:
    """Run scale_data.py at a given --aug_per_demo (seed 0, CPU) and return the
    augmented-arm success rate. Deterministic -> reproducible run-to-run."""
    out = workdir / f"aug{aug_per_demo}"
    subprocess.run([sys.executable, str(ARTIFACT), "--seed", "0", "--device", "cpu",
                    "--no-rerun", "--aug_per_demo", str(aug_per_demo), "--out", str(out)],
                   check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())["augmented_success_rate"]


if __name__ == "__main__":
    import tempfile
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    work = Path(tempfile.mkdtemp(prefix="z2r-ch37-ex3-"))
    prev = None
    for amount in AUG_AMOUNTS:
        rate = success_at(amount, work)
        step = "" if prev is None else f"  (step +{rate - prev:+.2f})"
        print(f"--aug_per_demo {amount}:  success {rate:.2f}{step}")
        prev = rate
    print("\nReconcile: is the 4->8 step smaller than the 0->4 step, or about the same? "
          "At free-tier scale (12 starved demos, aug<=8) coverage is still filling, so more "
          "augmentation keeps paying — the plateau is the asymptote you would only reach by "
          "pushing aug_per_demo much higher.")
