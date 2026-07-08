"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch5.2.

THE LEARNER-GENERATED FAILURE. Contrastive learning has two moves in every batch:
PULL each matched (frame, caption) pair together, and PUSH every non-matching pair
apart. The push comes free — every OTHER caption in the batch is a negative for your
frame. A very common bug when you write InfoNCE yourself is to keep the pull and
forget the push: you maximize the cosine of each matched pair and never repel the
negatives. align.py's `--break noneg` is exactly that buggy loss:

    # correct: cross-entropy over the B x B cosine matrix (pull the diagonal, push the rest)
    # buggy:   (1.0 - (img_e * txt_e).sum(-1)).mean()   <- pull only, NO negatives

PREDICT before you run: you train the aligned encoder twice at a reduced config —
once with the correct symmetric InfoNCE, once with `--break noneg` — and compare
retrieval@1 (fine). Which row is right?
  A) noneg >= correct — pulling the pairs together is all retrieval needs.
  B) correct much higher; noneg lower but still ABOVE random — without negatives the
     space partially COLLAPSES (nothing keeps different scenes apart), so retrieval
     still beats chance but is measurably worse.
  C) noneg near 0 (below random) — forgetting negatives makes it worse than an
     untrained encoder.

Record your answer in PREDICTION, then run (this TRAINS twice at a reduced scale —
a few minutes on CPU). Estimated learner time: 15 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch5.2-align",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
ALIGN = REPO / "curriculum/phase5_practitioner/ch5.2_align/align.py"
RC = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text())["exercise_checks"]["exercise_config"]


def config_flags() -> list[str]:
    return ["--episodes", str(RC["episodes"]), "--epochs", str(RC["epochs"]),
            "--sup_epochs", str(RC["sup_epochs"]), "--batch_size", str(RC["batch_size"])]


def run_align(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(ALIGN), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), *config_flags(), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        correct = run_align(Path(tmp) / "correct")
        noneg = run_align(Path(tmp) / "noneg", ["--break", "noneg"])
    print(f"correct InfoNCE -> retrieval@1 (fine) {correct['retrieval_at1_aligned']:.3f}  "
          f"(random {correct['retrieval_at1_random']:.3f})")
    print(f"--break noneg    -> retrieval@1 (fine) {noneg['retrieval_at1_aligned']:.3f}")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: with no negatives, what stops the encoder from mapping EVERY frame and "
          "EVERY caption to the same point (loss 0, retrieval useless)? Why does retrieval still beat "
          "random even so — what little structure survives the collapse?")
