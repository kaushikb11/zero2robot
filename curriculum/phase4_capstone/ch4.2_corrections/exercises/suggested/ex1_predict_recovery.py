"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch4.2.

Objective tested: the chapter's headline — DAgger RECOVERS the covariate shift
that killed behavior cloning. You predict, then you run the real loop and read the
Wilson intervals, because a recovery you cannot separate from noise is not one.

THE SETUP. `dagger.py` trains BC on demos from a NARROW start region (the block
only starts close to the goal) and deploys on the FULL task (block anywhere). BC
never saw the far starts — that is the manufactured covariate shift. Then DAgger
rolls out the policy, has the scripted expert label the states it visits,
aggregates, and retrains, for several rounds.

PREDICT before you run: over the default config, does DAgger (a) leave BC's
success essentially unchanged (the corrections do not help), (b) recover it —
the best DAgger round's success interval clears BC's — or (c) make it worse?
Write your choice and one sentence of why in PREDICTION.

Then run this file. It runs the loop once (seed 0, ~2 min on a CPU laptop) and
prints BC's success rate and the best DAgger round's, each with its Wilson
interval, plus the difference interval that is the actual verdict.

Estimated learner time: 25 minutes (mostly waiting on the run).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause, e.g. "b because ..."

METADATA = {"type": "predict-then-run", "chapter": "ch4.2-corrections", "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase4_capstone/ch4.2_corrections/dagger.py"


def measure(workdir: Path | None = None) -> dict:
    """Run dagger.py once at the default config (seed 0, cpu) and return its
    metrics.json. Deterministic on CPU -> same numbers every run (no flake)."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-dagger-ex1-"))
    subprocess.run([sys.executable, str(ARTIFACT), "--seed", "0", "--device", "cpu",
                    "--no-rerun", "--out", str(workdir)],
                   check=True, capture_output=True, cwd=REPO)
    return json.loads((workdir / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    m = measure()
    print(f"BC           success {m['bc_rate']:.3f}  CI [{m['bc_ci_lo']:.3f}, {m['bc_ci_hi']:.3f}]")
    print(f"best {m['best_round']:<8s} success {m['best_rate']:.3f}  CI [{m['best_ci_lo']:.3f}, {m['best_ci_hi']:.3f}]")
    print(f"recovery diff CI [{m['recovery_diff_ci_lo']:+.3f}, {m['recovery_diff_ci_hi']:+.3f}]  "
          f"-> {'SIGNIFICANT' if m['recovery_significant'] else 'not significant'}")
    print("\nReconcile: did the two intervals separate? The corrections did not change "
          "the network or the reward — only which states the dataset covers.")
