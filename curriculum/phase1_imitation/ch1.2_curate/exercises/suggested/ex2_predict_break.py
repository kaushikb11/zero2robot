"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch1.2.

Objective tested: why demonstrator disagreement measures DIFFICULTY, not noise
— and why using it as a quality knob backfires.

THE SETUP. Chapter 1.1 taught you that disagreement is BC's enemy: when demos
disagree, MSE averages them into mush. So here is the tempting move — curate by
KEEPING THE EPISODES THAT AGREE MOST with their neighbours and throwing out the
high-disagreement ones as "noise." That is exactly what `--break
low_disagreement` does. It keeps the same NUMBER of episodes the honest
outcome filter would, so the only thing that changes is WHICH ones.

You will run three policies: raw (everything), curated (kept by outcome), and
break (kept by lowest disagreement).

PREDICT before you run — the break policy's held-out success will land...
  A) above curated: minimizing disagreement is exactly what 1.1 said to do
  B) below raw: keeping agreeable demos is strictly worse than keeping all of them
  C) between the two — better than raw, but WORSE than honest curation, because
     the high-disagreement episodes it discarded were the HARD ones (far starts,
     rotations), and dropping them blinds the policy to exactly those states

Record your answer in PREDICTION, then run this file (~10 minutes on CPU: two
default-scale curate runs). NOTE the trap only shows at full scale — at a small
eval set the noise hides it, which is chapter 1.6's whole point.
Estimated learner time: 25 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch1.2-curate",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
CURATE = REPO / "curriculum/phase1_imitation/ch1.2_curate/curate.py"
RC = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text())["exercise_checks"]["exercise_config"]


def run_curate(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(CURATE), "--seed", "0", "--no-rerun", "--out", str(out),
           "--careful", str(RC["careful"]), "--sloppy", str(RC["sloppy"]),
           "--epochs", str(RC["epochs"]), "--eval_episodes", str(RC["eval_episodes"]),
           "--hidden_dim", str(RC["hidden_dim"]), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first.")
    with tempfile.TemporaryDirectory() as tmp:
        honest = run_curate(Path(tmp) / "honest")
        broken = run_curate(Path(tmp) / "broken", ["--break", "low_disagreement"])
    print(f"raw                    -> success {honest['raw_success_rate']:.3f}")
    print(f"curated (outcome)      -> success {honest['curated_success_rate']:.3f}")
    print(f"break (low_disagree)   -> success {broken['curated_success_rate']:.3f}")
    print(f"  disagreement kept: honest {honest['mean_disagreement_kept']:.4f}  "
          f"break {broken['mean_disagreement_kept']:.4f}  (the break's is LOWER — and it still loses)")
    print(f"\nyour prediction: {PREDICTION}")
