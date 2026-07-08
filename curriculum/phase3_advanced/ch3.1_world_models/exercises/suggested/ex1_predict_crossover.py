"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch3.1.

Objective tested: the chapter's headline — RECONSTRUCTION is easy, PREDICTION is
the real test, and a world model beats copy-last only once the horizon is long
enough that integrating the actions beats assuming stasis.

THE QUESTION. `wm.py` rolls the learned dynamics forward `horizon` steps on
actions alone (the PRIOR — never seeing the future) and scores each step against
COPY-LAST (assume the last observed state never changes).

PREDICT before you run: across the default config, does the world model's k-step
prediction (a) beat copy-last at EVERY horizon including k=1, (b) LOSE at k=1 but
overtake copy-last as the horizon grows, or (c) never beat copy-last? Write your
choice and one sentence of why in PREDICTION.

Before you read the printed "Reconcile" line at the end, write one sentence: if the
world model wins in aggregate, why might copy-last still beat it on the object (T-block)
dimensions specifically?

Then run this file. It trains the world model on seeds 0 and 1 (~20 s each) and
reports, per seed, the horizon k where the world model first beats copy-last
(`crossover_k`) and the mean error ratio (copy-last / world-model). Only after your
prediction is locked does it print how to reconcile that outcome.

Estimated learner time: 20 minutes (mostly waiting on two short runs).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause, e.g. "b because ..."

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch3.1-world-models",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase3_advanced/ch3.1_world_models/wm.py"
SEEDS = (0, 1)
COMMON = ["--device", "cpu", "--no-rerun"]  # forced CPU => reproducible measurement


def train_seed(seed: int, workdir: Path) -> dict:
    out = workdir / f"seed{seed}"
    subprocess.run([sys.executable, str(ARTIFACT), "--seed", str(seed), "--out", str(out), *COMMON],
                   check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


def measure(workdir: Path | None = None) -> dict[int, dict]:
    """Return {seed: metrics}. Deterministic: same seed -> same numbers on CPU."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-wm-ex1-"))
    return {seed: train_seed(seed, workdir) for seed in SEEDS}


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    results = measure()
    for seed, m in results.items():
        print(f"seed {seed}: crossover_k={m['crossover_k']}  "
              f"copy-last/world-model error ratio={m['pred_ratio_copy_over_wm']:.2f}x  "
              f"(recon floor {m['val_recon']:.3f})")
    print("\nReconcile: copy-last is nearly perfect for one step because almost nothing "
          "moves in a tenth of a second — the world model's edge is that it knows where "
          "the ACTION takes the pusher next, and that edge compounds over the horizon. "
          "Did it beat copy-last at k=1, or only after a few steps? That aggregate crossover "
          "is the model learning the EASY half of the simulator — the pusher kinematics. Run "
          "wm.py directly to see the per-dim split it prints: on the T-block's own dynamics, "
          "copy-last still wins.")
