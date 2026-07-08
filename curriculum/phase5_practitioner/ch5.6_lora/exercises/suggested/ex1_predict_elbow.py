"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch5.6.

Objective tested: the rank ELBOW. lora.py pretrains a compact conditioned policy on three
skills, HOLDS ONE OUT, freezes it, and then adapts to the held-out skill at a sweep of LoRA
ranks (0, 1, 2, 4, 8, 16) plus a full fine-tune. It reports, for each rank, the held-out fit
(R^2) and the % of the base weights the adapter trains.

PREDICT before you run: how does the held-out fit move as you raise the rank?
  A) It stays FLAT and low across every rank — a low-rank adapter cannot fit an unseen skill;
     only full fine-tuning (100% of the weights) can.
  B) It RISES with rank then PLATEAUS at full fine-tuning's ceiling: a small rank (r=4, ~1% of
     the weights) already recovers MOST of full-FT's held-out fit, and past the knee more
     trainable parameters buy almost nothing.
  C) It climbs LINEARLY with rank all the way to full fine-tuning — you need roughly all the
     parameters to match full-FT.

Record your answer in PREDICTION, then run this file. It runs lora.py at the default config
(a few seconds on a CPU laptop). The claim to internalize is the DIRECTION (rise-then-plateau;
a small rank recovers most), which holds on every seed — not the exact R^2. It kills the
misconception "fewer trainable parameters must mean a worse fit." Estimated learner time: 15 min.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch5.6-lora",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
LORA = REPO / "curriculum/phase5_practitioner/ch5.6_lora/lora.py"


def run_lora(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(LORA), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        m = run_lora(Path(tmp) / "clean")
    print("rank :", m["sweep_ranks"])
    print("%parm:", m["sweep_trainable_pct"])
    print("ho R2:", m["sweep_heldout_r2"])
    print(f"\nrank-4 LoRA: {m['lora_trainable_pct']:.2f}% of the weights, held-out R^2 {m['lora_heldout_r2']:.2f}, "
          f"recovering {m['lora_recovered_frac'] * 100:.0f}% of full-FT's gain (full {m['full_heldout_r2']:.2f}).")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: a rank-4 adapter trains ~1% of the weights yet recovers most of full "
          "fine-tuning's held-out fit, and rank-8 already sits on the full-FT line. What does that "
          "say about the 'fewer trainable parameters must mean a worse fit' intuition — and where, "
          "exactly, is the elbow past which more rank stops helping?")
