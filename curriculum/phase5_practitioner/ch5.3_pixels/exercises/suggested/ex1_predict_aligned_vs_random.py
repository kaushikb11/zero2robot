"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch5.3.

Objective tested: the chapter's thesis and the payoff of ch1.8's `--break blind`.
pixels.py aligns a tiny ViT to the scene geometry (contrastive, frozen) and compares it to a
RANDOM one (the identical architecture, never aligned) on the CONTROL-USEFULNESS PROBE: freeze
each encoder and fit an action-regression MLP on its features, then read the HELD-OUT val MSE —
can a controller read the expert action off the features at all?

PREDICT before you run: how do aligned vs random features compare on the probe's val MSE?
  A) About the same — the probe can learn from either encoder's features, so the
     encoder barely matters (like ch1.1, where a random encoder would have been fine).
  B) Aligned is more control-useful (lower val MSE) — from pixels with NO state, the encoder's
     quality is the whole ballgame; a random projection doesn't carry the geometry the head needs.
  C) Random wins — alignment overfits the encoder and hurts the features.

Record your answer in PREDICTION, then run this file. NOTE: this TRAINS both encoders (a compact
contrastive alignment + two probes + a pixels-only rollout) — a few minutes on CPU; the automated
reproduce check is marked slow. The probe direction (aligned < random val MSE) is the seed-robust,
gated headline. The closed-loop ROLLOUT is a HIGHER bar that floors at free-tier for BOTH encoders
(0/12) — that is the honest ceiling and the Scale Lab, not a bug. Estimated learner time: 20 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch5.3-pixels",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
PIXELS = REPO / "curriculum/phase5_practitioner/ch5.3_pixels/pixels.py"
RC = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text()
                    )["exercise_checks"]["exercise_config"]


def run_pixels(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(PIXELS), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), "--episodes", str(RC["episodes"]), "--dim", str(RC["dim"]),
           "--depth", str(RC["depth"]), "--heads", str(RC["heads"]),
           "--align_epochs", str(RC["align_epochs"]), "--bc_epochs", str(RC["bc_epochs"]),
           "--eval_episodes", str(RC["eval_episodes"]), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        m = run_pixels(Path(tmp) / "out")
    print(f"aligned encoder -> probe val_mse {m['probe_val_mse_aligned']:.4f}   (lower = more control-useful)")
    print(f"random  encoder -> probe val_mse {m['probe_val_mse_random']:.4f}")
    print(f"probe gap (random - aligned) {m['probe_mse_gap']:+.4f}  (> 0 == aligned wins; your prediction: {PREDICTION})")
    print(f"[higher bar, Scale Lab] closed-loop rollout: aligned {m['aligned_success_rate']:.2f} vs "
          f"random {m['random_success_rate']:.2f} — floors for both at free-tier (see meta.yaml)")
    print("\nNow explain it: ch1.1 cloned PushT from STATE and a random encoder would have "
          "been fine. Here the state is GONE. Why can a controller read the action off the ALIGNED "
          "features but not the random ones — what did the contrastive alignment put into them?")
