"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch3.2.

Objective tested: the chapter's headline — the IMAGINATION GAP. `dreamer.py`
trains an actor entirely inside the frozen ch3.1 world model (never touching the
real sim), then deploys that SAME policy in the dream AND in the true PushT sim.

THE QUESTION. The actor optimizes a reward computed from the block pose the world
model DECODES. But 3.1 measured that this model learned the pusher kinematics and
NOT the block/contact dynamics. So: when you deploy the imagination-trained policy
in the REAL sim, does it (a) transfer — the pushing it learned in the dream works,
real success climbs; (b) look great in imagination and FAIL in reality — a large
positive gap (imagined return >> real) with real success stuck near 0; or (c) fail
in BOTH — it never even learned to raise imagined return? Write your choice and one
sentence of why in PREDICTION.

Before you read the printed "Reconcile" line at the end, write one sentence: the actor
provably raised its imagined return, so what exactly did it get good at — and why does
that skill earn zero real success?

Then run this file. It trains the full pipeline on seeds 0 and 1 (~13 s each) and
reports, per seed, the imagined vs real return/step, the gap, the final block-to-
target distance in each world, and the real task success rate. Only after your
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
    "chapter": "ch3.2-dreamer",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase3_advanced/ch3.2_dreamer/dreamer.py"
SEEDS = (0, 1)
COMMON = ["--device", "cpu", "--no-rerun"]  # forced CPU => reproducible measurement


def train_seed(seed: int, workdir: Path) -> dict:
    out = workdir / f"seed{seed}"
    subprocess.run([sys.executable, str(ARTIFACT), "--seed", str(seed), "--out", str(out), *COMMON],
                   check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


def measure(workdir: Path | None = None) -> dict[int, dict]:
    """Return {seed: metrics}. Deterministic: same seed -> same numbers on CPU."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-dreamer-ex1-"))
    return {seed: train_seed(seed, workdir) for seed in SEEDS}


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    results = measure()
    for seed, m in results.items():
        print(f"seed {seed}: gap={m['imagination_gap']:+.3f}  "
              f"imagined={m['imagined_return_per_step']:+.3f} (tee-dist {m['imagined_final_tee_dist']:.3f} m)  "
              f"real={m['real_return_per_step']:+.3f} (tee-dist {m['real_final_tee_dist']:.3f} m)  "
              f"real success={m['real_success_rate']:.2f}")
    print("\nReconcile: the dream \"parks\" the block near the target (small imagined "
          "tee-dist) while the real block barely moves from its spawn — the policy learned "
          "to game a dream whose hard half is wrong. Imagination is only as good as your "
          "world model.")
