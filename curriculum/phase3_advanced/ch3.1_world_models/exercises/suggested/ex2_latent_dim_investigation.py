"""SUGGESTED exercise candidate (humans promote) — investigation, ch3.1.

Objective tested: the latent is the world model's memory of "everything about the
state the dynamics must carry forward." Starve it and the prior cannot represent
the next state, so PREDICTION collapses even though RECONSTRUCTION (which also gets
the deterministic GRU state) barely suffers.

THE INVESTIGATION. Sweep `--latent_dim` over {2, 16} (tiny vs the chapter default)
on seeds 0-1 and read the copy-last / world-model prediction ratio. A ratio near
1 (or below) means the world model is no better than assuming nothing moves; a
ratio well above 1 means it learned to step.

PREDICT before you run: at latent_dim=2, does prediction (a) hold up nearly as well
as latent_dim=16, or (b) collapse toward copy-last? Write PREDICTION.

Before you read the reconciliation, write one sentence: why does starving the latent to
dim 2 wreck PREDICTION far more than it hurts RECONSTRUCTION?

Then run. You should see the tiny latent barely beat (or lose to) copy-last while
the full latent wins by ~2x — capacity you can watch turn into prediction quality.
This is a hands-on taste of the chapter's thesis: better world models cost more
model, which is why the pixel version is a Scale Lab.

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
    "chapter": "ch3.1-world-models",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase3_advanced/ch3.1_world_models/wm.py"
SEEDS = (0, 1)
LATENT_DIMS = (2, 16)
COMMON = ["--device", "cpu", "--no-rerun"]


def run(latent_dim: int, seed: int, workdir: Path) -> float:
    out = workdir / f"ld{latent_dim}_seed{seed}"
    subprocess.run([sys.executable, str(ARTIFACT), "--seed", str(seed),
                    "--latent_dim", str(latent_dim), "--out", str(out), *COMMON],
                   check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())["pred_ratio_copy_over_wm"]


def measure(workdir: Path | None = None) -> dict[int, list[float]]:
    """Return {latent_dim: [ratio per seed]}. Deterministic on CPU."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-wm-ex2-"))
    return {ld: [run(ld, seed, workdir) for seed in SEEDS] for ld in LATENT_DIMS}


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    results = measure()
    for ld, ratios in results.items():
        mean = sum(ratios) / len(ratios)
        print(f"latent_dim={ld:2d}  per-seed ratio {[round(r, 2) for r in ratios]}  mean {mean:.2f}x")
    print("\nReconcile: how much prediction quality did the extra latent capacity buy? "
          "That is the compute-vs-quality trade the chapter is about.")
