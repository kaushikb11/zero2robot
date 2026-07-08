"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch5.4.

Objective tested: the chapter's headline mechanism. vla_shape.py trains a two-tower VLA — a PREFIX
(vision + state + instruction) and a SUFFIX action expert whose ONLY window onto the prefix is the
suffix->prefix cross-attention. The run reports the trained weights' HELD-OUT velocity fit under the
FULL mask and under the SAME weights with that cross-attention SEVERED (--break cut_cross's mask).

PREDICT before you run: what happens to the held-out flow-MSE when you cut suffix->prefix?
  A) About the same — the expert's action-query tokens already carry the state (the noised action +
     the flow time is enough), so the cross-attention barely matters.
  B) It COLLAPSES (flow_mse_cut >> flow_mse_full) — the suffix tokens carry only a noised action + a
     clock, so the cross-attention is the expert's ONLY path to the state; deny it and the fit falls
     toward the unconditional prior.
  C) It IMPROVES — dropping attention edges regularizes the expert.

Record your answer in PREDICTION, then run this file. It TRAINS the two-tower (~90 s on CPU); the
automated reproduce check is marked slow. The flow-MSE gap (cut - full > 0) is the seed-robust, byte-
reproducible, GATED headline. Note the PushT rollout it also prints: it FLOORS for both masks — that
is the higher bar and the Scale Lab, not this exercise's claim (see ex2). Estimated learner time: 15 min.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch5.4-vla-shape",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase5_practitioner/ch5.4_vla_shape/vla_shape.py"
RC = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text()
                    )["exercise_checks"]["exercise_config"]


def run(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(ARTIFACT), "--seed", "0", "--device", "cpu", "--no-rerun", "--out", str(out),
           "--episodes", str(RC["episodes"]), "--epochs", str(RC["epochs"]),
           "--eval_episodes", str(RC["eval_episodes"]), "--horizon", str(RC["horizon"]),
           "--model_dim", str(RC["model_dim"]), "--layers", str(RC["layers"]), "--heads", str(RC["heads"]),
           *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        m = run(Path(tmp) / "out")
    print(f"held-out flow-MSE:  full {m['flow_mse_full']:.4f}   cut-cross {m['flow_mse_cut']:.4f}")
    print(f"gap (cut - full) = {m['flow_mse_gap']:+.4f}   (> 0 == severing collapses the fit; your prediction: {PREDICTION})")
    print(f"[higher bar, floors] PushT rollout: {m['reported_success_rate']:.2f} success — see ex2")
    print("\nNow explain it: the suffix (action-expert) tokens are just a NOISED action + a flow clock. "
          "Trace where the STATE that determines the PushT action can possibly enter the expert. What "
          "did severing suffix->prefix take away — and why is that 'routing is load-bearing,' not just "
          "'the model got worse'?")
