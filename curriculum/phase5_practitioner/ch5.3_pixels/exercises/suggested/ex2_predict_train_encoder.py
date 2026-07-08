"""SUGGESTED exercise candidate (humans promote) — predict-then-run + THE TRAP, ch5.3.

This is the LEARNER-GENERATED deliberate failure. You have wired the encoder->policy
path. The obvious "improvement" is to stop freezing the encoder and let it learn end-to-end
with the head — surely training the whole thing beats training only an adapter? At free-tier
scale (a tiny demo set, a from-scratch ViT) it is a TRAP: the encoder has enough capacity to
memorize the frames it saw, driving the BC TRAINING loss LOWER — the classic overfit signature.
(The pixels-only rollout would show the eventual collapse, but at free-tier it floors near 0/12
for BOTH encoders, so success rate can't separate them; we read the memorization off the train
loss instead.) pixels.py exposes the trap as `--train_encoder` (default OFF = encoder frozen).

PREDICT before you run: frozen aligned encoder vs the SAME aligned encoder unfrozen
(`--train_encoder`). Compare their FINAL BC TRAINING loss (`bc_final_loss_aligned`) — remember
the rollout floors at 0/12 for BOTH at free-tier, so the signal that actually MOVES is the loss:
  A) Unfrozen reaches a LOWER BC loss — the encoder's extra capacity MEMORIZES the tiny demo set
     (the overfit signature). It buys nothing: the rollout still floors for both, so the lower
     train loss is memorization, not skill.
  B) Unfrozen reaches a HIGHER BC loss — unfreezing more parameters makes the optimization harder.
  C) They tie — freezing vs training the encoder leaves the loss curve unchanged.

Record your answer in PREDICTION, then run this file. NOTE: this TRAINS twice (the second
run backprops through the whole ViT on pixels — slower). A few minutes on CPU; the automated
reproduce check is marked slow. Estimated learner time: 20 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch5.3-pixels",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
PIXELS = REPO / "curriculum/phase5_practitioner/ch5.3_pixels/pixels.py"
RC = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text()
                    )["exercise_checks"]["exercise_config"]


def run_pixels(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(PIXELS), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), "--episodes", str(RC["episodes"]), "--dim", str(RC["dim"]),
           "--depth", str(RC["depth"]), "--heads", str(RC["heads"]),
           "--align_epochs", str(RC["align_epochs"]), "--bc_epochs", str(RC["bc_epochs"]),
           "--eval_episodes", str(RC["eval_episodes"]), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        frozen = run_pixels(Path(tmp) / "frozen")
        trap = run_pixels(Path(tmp) / "trap", ["--train_encoder"])
    print(f"frozen aligned encoder     -> BC train loss {frozen['bc_final_loss_aligned']:.4f}  "
          f"(rollout {frozen['aligned_success_rate']:.2f}, floored)")
    print(f"--train_encoder (unfrozen) -> BC train loss {trap['bc_final_loss_aligned']:.4f}  "
          f"(rollout {trap['aligned_success_rate']:.2f}, floored) — unfreezing drives train loss LOWER = memorizing")
    print(f"(your prediction: {PREDICTION})")
    print("[read the SIGNAL] the decision is the BC TRAIN LOSS: unfreezing the encoder reaches a LOWER "
          "train loss because it MEMORIZES the tiny demo set — the overfit signature. It is not skill: "
          "at free-tier both pixels-only rollouts floor near 0/12 (a Scale-Lab bar), so success rate "
          "can't separate them. Frozen keeps the transferable alignment and does NOT overfit.")
    print("\nNow explain it: ch1.1 trained its WHOLE network end-to-end and it was fine. Here "
          "unfreezing the encoder hurt. What is different? (Hint: ch1.1's input was a 10-number "
          "state that could not be memorized into a lookup; a from-scratch ViT over a tiny frame "
          "set can. The alignment already did the expensive, transferable work — leave it frozen.)")
