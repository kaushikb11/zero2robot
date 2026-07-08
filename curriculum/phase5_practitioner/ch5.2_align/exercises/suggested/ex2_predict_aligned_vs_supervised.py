"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch5.2.

Objective tested: WHY contrastive, not just supervised. align.py trains three encoders
and scores each by retrieval: type an instruction, rank held-out frames by cosine.
  - ALIGNED:     contrastive InfoNCE — no labels, only which caption rode with which frame.
  - SUPERVISED:  ch5.1's probe — a ViT trained on the 4-way QUADRANT label, then frozen and
                 aligned to text (this IS the "contrastive needs labels?" foil — it uses them).
  - RANDOM:      untrained init — the floor.
Captions name BOTH the quadrant AND near/far ("the block is FAR the top left corner").
The retrieval@1 (FINE) score demands the retrieved frame match on quadrant AND near/far.

PREDICT before you run: how do the three FINE retrieval@1 scores order?
  A) aligned > supervised > random — contrastive keeps the WHOLE caption; the supervised
     probe only kept its 4-way quadrant label, so it loses the near/far half; random is chance.
  B) supervised > aligned > random — labels beat no-labels; supervision always wins.
  C) aligned ~ supervised >> random — both learn the full caption equally well.

(Hint: also print the QUAD-only score. On quadrant alone the supervised probe ~ ties aligned —
it was TRAINED on the quadrant. The gap opens on the FINE score, where the caption says more
than the label ever did.) Record PREDICTION, then run (TRAINS three encoders at a reduced
config — a few minutes on CPU). Estimated learner time: 15 minutes.
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


def run_align(out: Path) -> dict:
    cmd = [sys.executable, str(ALIGN), "--seed", "0", "--device", "cpu", "--no-rerun", "--out", str(out),
           "--episodes", str(RC["episodes"]), "--epochs", str(RC["epochs"]),
           "--sup_epochs", str(RC["sup_epochs"]), "--batch_size", str(RC["batch_size"])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        m = run_align(Path(tmp))
    print(f"retrieval@1 FINE:  aligned {m['retrieval_at1_aligned']:.3f}  "
          f"supervised {m['retrieval_at1_supervised']:.3f}  random {m['retrieval_at1_random']:.3f}")
    print(f"retrieval@1 QUAD:  aligned {m['retrieval_quad_at1_aligned']:.3f}  "
          f"supervised {m['retrieval_quad_at1_supervised']:.3f}  random {m['retrieval_quad_at1_random']:.3f}")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: the supervised encoder was TRAINED on quadrant labels and the contrastive "
          "one saw NONE. Why does the label-free encoder still win on the FINE score — what did the "
          "4-way label throw away that the caption (and contrastive) kept?")
