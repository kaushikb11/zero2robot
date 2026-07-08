"""SUGGESTED exercise candidate (humans promote) — predict-then-run + the learner-generated
deliberate failure, ch5.1.

Objective tested: the misconception "a ViT sees the image as a picture," and the footgun that
follows from it. You already wrote patchify in ex3. The single most common way to get it wrong
is to forget the permute, so the patch-grid axis INTERLEAVES with the pixel axis — every
"patch" becomes a scramble of the real ones (a global permutation of the patch set). vit.py
ships that exact bug behind `--break patch_interleave`, and the position-tag misconception
behind `--break shuffle_pos` (scramble the LEARNED position embeddings, pixels untouched).

You will run three configs and read `probe_acc_trained` from each:
  - clean                    : correct patchify
  - --break patch_interleave : the reshape bug (patches globally permuted)
  - --break shuffle_pos      : position tags scrambled, pixels identical

PREDICT before you run: which row is right?
  A) Both breaks crash the probe to chance (~0.25) — scrambling patches or positions destroys
     the scene.
  B) patch_interleave crashes the probe; shuffle_pos leaves it untouched.
  C) patch_interleave is SILENT — the coarse probe barely moves (a permuted bag is the same
     bag) — while shuffle_pos collapses the trained model's EDGE over random (toward the
     random-init level), because the learned spatial structure lived in the position TAGS.

Record your answer in PREDICTION, then run this file. It trains the ViT THREE times at the
default config (~2 min total on a CPU laptop). Estimated learner time: 20 minutes.

THE POINT: a coarse-accuracy metric CANNOT catch a patchify bug — the bug is invisible until
you look at the attention map (the toy). A ViT does not see a picture; it sees a bag of patch
vectors plus position tags, and coarse facts survive scrambling the bag.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch5.1-vit",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
VIT = REPO / "curriculum/phase5_practitioner/ch5.1_vit/vit.py"
RC = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text())["exercise_checks"]["exercise_config"]


def run_vit(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(VIT), "--seed", "0", "--device", "cpu", "--no-rerun", "--out", str(out),
           "--episodes", str(RC["episodes"]), "--epochs", str(RC["epochs"]), "--warmup", str(RC["warmup"]),
           "--dim", str(RC["dim"]), "--depth", str(RC["depth"]), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        clean = run_vit(Path(tmp) / "clean")
        interleave = run_vit(Path(tmp) / "interleave", ["--break", "patch_interleave"])
        shuffle = run_vit(Path(tmp) / "shuffle", ["--break", "shuffle_pos"])
    print(f"clean               trained {clean['probe_acc_trained']:.3f}  (gap over random {clean['probe_gap']:+.3f})")
    print(f"patch_interleave    trained {interleave['probe_acc_trained']:.3f}  (gap {interleave['probe_gap']:+.3f})")
    print(f"shuffle_pos         trained {shuffle['probe_acc_trained']:.3f}  (gap {shuffle['probe_gap']:+.3f})")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: WHY does globally permuting the patches barely move a quadrant "
          "probe (what is invariant?), and why does scrambling the position embeddings — with "
          "the pixels byte-for-byte identical — erase most of what training bought? What single "
          "sentence about 'a ViT sees a picture' does each result refute?")
