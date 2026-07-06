"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch1.8.

Objective tested: the chapter's Break-It and its thesis. A vision-language-action
policy is SUPPOSED to look at its camera. This one's camera is ch1.7's FROZEN,
RANDOM-INIT CNN — a fixed projection of the pixels, not learned perception. So: how
much does the trained policy actually USE its vision on PushT?

THE SETUP. vla.py trains the tiny VLA and evaluates PushT success. Run it twice:
  - sighted:      default (the policy sees the frozen image feature)
  - --break blind: the image feature is zeroed at BOTH train and eval — the policy
                   gets words + state, but NO vision, ever.

PREDICT before you run: what happens to PushT success under --break blind?
  A) It collapses toward the untrained 0.0 — the policy depended on its vision.
  B) It barely changes — PushT is solvable from the STATE (ch1.1 did it from state),
     and a RANDOM-init vision feature added little; the policy learned to ignore it.
  C) It improves a lot — vision was pure noise that was hurting the policy.

Record your answer in PREDICTION, then run this file. NOTE: this TRAINS the policy
twice at the full config (regenerates 60 demos/task via ch1.7, ~2 min each) — a few
minutes on CPU. That's why the automated reproduce check is marked slow.
Estimated learner time: 20 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch1.8-vla",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
VLA = REPO / "curriculum/phase1_imitation/ch1.8_vla/vla.py"


def run_vla(data: Path, out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(VLA), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--data", str(data), "--out", str(out), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"  # both runs share ONE regenerated dataset (same frozen features)
        sighted = run_vla(data, Path(tmp) / "sighted")
        blind = run_vla(data, Path(tmp) / "blind", ["--break", "blind"])
    print(f"sighted   -> PushT success {sighted['pusht_success_rate']:.2f}  (untrained {sighted['baseline_pusht_success_rate']:.2f})")
    print(f"--break blind -> PushT success {blind['pusht_success_rate']:.2f}")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: the ONLY change was zeroing the image feature. If PushT "
          "success barely moved, what was the policy actually conditioning on — and what "
          "would a PRETRAINED vision backbone (SmolVLA) have to add to make vision matter?")
