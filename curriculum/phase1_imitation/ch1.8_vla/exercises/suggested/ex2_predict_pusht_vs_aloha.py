"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch1.8.

Objective tested: the honest CEILING of a tiny from-scratch VLA, and why it is uneven
across tasks. The policy trains on ONE shared pile mixing two embodiments — PushT (a
2-D pusher, solvable from state) and ALOHA (a 6-D bimanual cube handoff, the task ACT's
action chunking was built for in ch1.3). One tiny transformer + one flow head + one
random vision encoder must serve both.

THE SETUP. vla.py trains once and reports success on BOTH tasks (PushT over N episodes,
ALOHA over N/2). You will read them off one run.

PREDICT before you run: which task does the tiny from-scratch VLA do BETTER on?
  A) PushT >> ALOHA — it learns the state-solvable single-arm push, but cannot
     coordinate ALOHA's mid-air handoff with this little capacity and no action chunking.
  B) ALOHA >> PushT — more action dimensions means more supervision, so the harder
     task trains better.
  C) About the same — both are manipulation, so one shared policy handles them alike.

Record your answer in PREDICTION, then run this file. NOTE: this TRAINS the policy at
the full config (regenerates 60 demos/task via ch1.7), a couple of minutes on CPU.
Estimated learner time: 15 minutes.
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


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        cmd = [sys.executable, str(VLA), "--seed", "0", "--device", "cpu", "--no-rerun",
               "--data", str(Path(tmp) / "data"), "--out", str(out)]
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
        m = json.loads((out / "metrics.json").read_text())
    print(f"PushT trained success {m['pusht_success_rate']:.2f}  (untrained {m['baseline_pusht_success_rate']:.2f})")
    print(f"ALOHA trained success {m['aloha_success_rate']:.2f}")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: which task's ANSWER is already in the state vector, and which "
          "needs a temporally coordinated multi-step plan the flow head samples one step at "
          "a time? What would you add (from ch1.3) to give ALOHA a chance?")
