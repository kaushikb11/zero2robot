"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch1.9.

The graduation question: now that the OFFICIAL lerobot ACT trains on your data in
~20 lines, is it BETTER than the ~380-line ACT you built by hand in 1.3? bridge.py
trains both on the SAME demos and evaluates both on the SAME held-out seeds, and —
because 1.6 taught you that a bare success number is a lie — it reports the
official policy's success WITH its Wilson interval.

PREDICT before you run (default budget, seed 0): how will the two compare?
  A) The official ACT clearly WINS — its success rate clears the from-scratch
     rate, the interval leaves no doubt. The framework's engineering pays off.
  B) They are statistically INDISTINGUISHABLE at this budget and episode count —
     the from-scratch rate sits inside the official policy's Wilson interval.
     Same algorithm, same data: the code you write differs, the ACT does not.
  C) The from-scratch ACT clearly WINS — your hand-tuned version beats the
     general-purpose framework.

Record your answer in PREDICTION, then run this file (trains both ACTs; minutes on
CPU). Estimated learner time: 20 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch1.9-bridge",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
BRIDGE = REPO / "curriculum/phase1_imitation/ch1.9_bridge/bridge.py"
RC = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text())["exercise_checks"]["exercise_config"]


def run_bridge(out: Path) -> dict:
    cmd = [sys.executable, str(BRIDGE), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), "--chunk_size", str(RC["chunk_size"]), "--model_dim", str(RC["model_dim"]),
           "--num_demos", str(RC["num_demos"]), "--epochs", str(RC["epochs"]),
           "--eval_episodes", str(RC["eval_episodes"])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        m = run_bridge(Path(tmp) / "run")
    print(f"official  ACT: success {m['official_success_rate']:.2f}  "
          f"[{m['official_ci_lo']:.2f}, {m['official_ci_hi']:.2f}]")
    print(f"from-scratch : success {m['scratch_success_rate']:.2f}  (1.3's act.py, same data/seeds)")
    inside = m["official_ci_lo"] <= m["scratch_success_rate"] <= m["official_ci_hi"]
    print(f"from-scratch rate inside official CI: {inside}")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: it is the SAME algorithm (ACT) on the SAME data, yet your 1.3 "
          "version wins. What did your 380 lines encode that the 20 lines did not — and if "
          "the framework did not buy a higher number, what DID it buy (a Hub, a community, a "
          "real-robot deploy path, a dozen other policies one import away)?")
