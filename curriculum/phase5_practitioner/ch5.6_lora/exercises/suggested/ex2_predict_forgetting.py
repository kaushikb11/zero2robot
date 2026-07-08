"""SUGGESTED exercise candidate (humans promote) — predict-then-run + the honest twist, ch5.6.

Objective tested: the tempting intuition that "LoRA freezes W, so it can't forget what the
model already knew." lora.py watches an IN-DISTRIBUTION skill (task_A, one of the three the
base was pretrained on) while it adapts the frozen policy to a HELD-OUT skill. It reports
task_A's fit (R^2) for the frozen base, the LoRA-adapted policy, and the full fine-tune.

PREDICT before you run: after adapting to the new skill, what happens to task_A?
  A) LoRA PRESERVES task_A (its W is frozen, so the old skill's mapping is untouched) while
     full fine-tuning FORGETS it — LoRA's headline advantage is protected memory.
  B) BOTH preserve task_A — adapting one skill never disturbs another, whichever method you use.
  C) task_A COLLAPSES under LoRA just as it does under full fine-tuning (here, MORE): the
     low-rank adapter is added to EVERY input and, being a single linear map, cannot gate itself
     off for the old skill — so freezing W does NOT protect it. "Frozen weights" is not "frozen
     behavior"; LoRA's real win is parameter efficiency, not free memory.

Record your answer in PREDICTION, then run this file. It runs lora.py at the default config
(a few seconds on a CPU laptop). The claim to internalize is the DIRECTION — task_A degrades
under BOTH arms, seed-robustly — not the exact R^2. This refutes a plausible, widely-repeated
intuition; the measured result is the lesson. Estimated learner time: 20 minutes.
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
    print(f"task_A R^2   frozen {m['frozen_task_a_r2']:+.2f}   ->   LoRA {m['lora_task_a_r2']:+.2f}   "
          f"|   full-FT {m['full_task_a_r2']:+.2f}")
    print(f"(LoRA forgot {m['lora_task_a_forget']:+.2f}; full-FT forgot {m['full_task_a_forget']:+.2f})")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: the LoRA adapter's W is frozen, yet task_A collapses. The adapter output "
          "(alpha/r)*B(A x) is ADDED to every input — including task_A's. Why can a single low-rank "
          "LINEAR map not switch itself off for the old skill (what would it take to gate on the "
          "skill token), and what does that tell you about when 'frozen weights' actually protect "
          "prior behavior in a real robot policy?")
