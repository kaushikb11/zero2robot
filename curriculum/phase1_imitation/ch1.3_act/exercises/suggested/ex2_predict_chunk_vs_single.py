"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch1.3.

Objective tested: the chapter's core claim, stated as an ablation. Does
predicting a CHUNK actually beat predicting one action? --break no_chunk forces
K=1, which turns ACT back into single-step behavior cloning through a
transformer (same architecture, same demos, same epochs — only the output shape
shrinks from K actions to one). Everything else is held fixed, so any gap is the
chunking itself.

PREDICT before you run: on held-out success rate...
  A) the chunked policy (K>1) clearly beats no_chunk (K=1) — committing to a
     short plan carries the bimanual hand-off that a single reactive step drops
  B) no_chunk (K=1) wins — one action re-decided every step reacts faster than a
     policy locked into a stale chunk
  C) they tie — chunk or no chunk, the transformer sees the same observation, so
     the output shape can't matter

Record your answer in PREDICTION below, then run this file (a few minutes on CPU
— it trains twice, K>1 and K=1).
Estimated learner time: 15 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch1.3-act",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
ACT = REPO / "curriculum/phase1_imitation/ch1.3_act/act.py"
CFG = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text())
RC = CFG["exercise_checks"]["exercise_config"]


def config_flags() -> list[str]:
    return ["--chunk_size", str(RC["chunk_size"]), "--model_dim", str(RC["model_dim"]),
            "--num_demos", str(RC["num_demos"]), "--epochs", str(RC["epochs"]),
            "--eval_episodes", str(RC["eval_episodes"])]


def run_act(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(ACT), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), *config_flags(), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        chunked = run_act(Path(tmp) / "chunked")
        single = run_act(Path(tmp) / "single", extra=["--break", "no_chunk"])
    print(f"chunked (K={chunked['chunk_size']}) -> success {chunked['success_rate']:.2f}  return {chunked['mean_return']:.1f}")
    print(f"no_chunk (K=1)         -> success {single['success_rate']:.2f}  return {single['mean_return']:.1f}")
    print(f"success delta = {chunked['success_rate'] - single['success_rate']:+.2f}   (your prediction: {PREDICTION})")
    print("\nBoth trained on the same demos for the same epochs. The only difference was the output shape.")
