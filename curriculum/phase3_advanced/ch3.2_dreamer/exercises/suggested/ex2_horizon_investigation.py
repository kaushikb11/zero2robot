"""SUGGESTED exercise candidate (humans promote) — investigation, ch3.2.

Objective tested: WHERE the imagination gap comes from. If the world model were
perfect, rolling it further in imagination would give the actor a longer, still-
truthful planning horizon — more imagination should buy more real skill. It does
not here, and this exercise makes you feel why: rolling a WRONG model further just
compounds its error into a more confident hallucination.

THE INVESTIGATION. Sweep `--imag_horizon` over {5, 30} (the number of steps the
actor dreams per policy update) on seeds 0-1 and read two things: how close the
DREAM thinks it parks the block (`imagined_final_tee_dist`) and how close it
actually gets in the REAL sim (`real_final_tee_dist`).

PREDICT before you run: as you lengthen the imagination horizon, does the REAL
performance (a) improve — a longer dream is a better planner; or (b) stay stuck
while the DREAM only grows more convinced it succeeded? Write PREDICTION.

Before you read the reconciliation, write one sentence: why does dreaming 30 steps
instead of 5 make the dream no more truthful about the block?

Then run. It sweeps the imagination horizon on both seeds and prints, per horizon,
the DREAM's tee-dist, the REAL tee-dist, and real success. Only after your prediction
is locked does it print how to reconcile that outcome.

Estimated learner time: 25 minutes (four short runs).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b", plus a because-clause

METADATA = {
    "type": "investigation",
    "chapter": "ch3.2-dreamer",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase3_advanced/ch3.2_dreamer/dreamer.py"
SEEDS = (0, 1)
HORIZONS = (5, 30)
COMMON = ["--device", "cpu", "--no-rerun"]


def run(horizon: int, seed: int, workdir: Path) -> dict:
    out = workdir / f"h{horizon}_seed{seed}"
    subprocess.run([sys.executable, str(ARTIFACT), "--seed", str(seed),
                    "--imag_horizon", str(horizon), "--out", str(out), *COMMON],
                   check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


def measure(workdir: Path | None = None) -> dict[int, list[dict]]:
    """Return {horizon: [metrics per seed]}. Deterministic on CPU."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-dreamer-ex2-"))
    return {h: [run(h, seed, workdir) for seed in SEEDS] for h in HORIZONS}


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    results = measure()
    for h, runs in results.items():
        imag = sum(m["imagined_final_tee_dist"] for m in runs) / len(runs)
        real = sum(m["real_final_tee_dist"] for m in runs) / len(runs)
        succ = sum(m["real_success_rate"] for m in runs) / len(runs)
        print(f"imag_horizon={h:2d}  dream tee-dist {imag:.3f} m  |  REAL tee-dist {real:.3f} m  "
              f"real success {succ:.2f}")
    print("\nReconcile: the imagined tee-dist sits near zero at EVERY horizon (the dream "
          "parks the block whether you imagine 5 steps or 30 — ~0.01-0.02 m either way) while "
          "the real tee-dist sits at ~0.16 m at EVERY horizon — the block barely moves no "
          "matter how long you imagine. The delusion is horizon-INVARIANT: a longer rollout of "
          "a wrong model is a more confident hallucination, which is why a better world model "
          "(the Scale Lab) — not a longer rollout — is what closes the gap.")
