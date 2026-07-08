"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch5.5.

Objective tested: what DCT -> quantize -> BPE actually buys you. fast.py takes real PushT +
ALOHA action chunks and encodes each two ways, then reports in metrics.json:
  - fast_tokens        : DCT -> quantize -> BPE (the FAST codec)
  - naive_tokens       : per-step-per-dim binning (one token per action number = H*act_dim)
  - fast_recon_rmse    : reconstruction error of the FAST codec (normalized action units)
  - naive_recon_rmse   : reconstruction error of per-step binning at the SAME quantization step

PREDICT before you run: how does FAST compare to naive per-step binning?
  A) FAST uses about the SAME number of tokens at the same error — the DCT just relabels the
     data, so nothing is gained.
  B) FAST uses noticeably FEWER tokens at COMPARABLE error — the orthonormal DCT costs the same
     error energy as quantizing raw samples (Parseval), but in the frequency domain that error
     concentrates, most coefficients round to 0, and BPE merges the zero-runs.
  C) FAST uses fewer tokens but at MUCH WORSE error — you can only save tokens by throwing away
     reconstruction quality.

Record your answer in PREDICTION, then run this file. It runs the CODEC (pure numpy, no
training, well under a second). The claim to internalize is the DIRECTION (fewer tokens at
comparable error), which holds on every seed — the exact ratio (~2.2-2.5x here) depends on how
smooth the trajectories are. Estimated learner time: 10 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch5.5-fast",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
FAST = REPO / "curriculum/phase5_practitioner/ch5.5_fast/fast.py"
RC = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text())["exercise_checks"]["exercise_config"]


def run_fast(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(FAST), "--seed", "0", "--no-rerun", "--out", str(out),
           "--horizon", str(RC["horizon"]), "--episodes_per_task", str(RC["episodes_per_task"]),
           "--q_scale", str(RC["q_scale"]), "--num_merges", str(RC["num_merges"]), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        m = run_fast(Path(tmp) / "clean")
    print(f"fast_tokens        {m['fast_tokens']}")
    print(f"naive_tokens       {m['naive_tokens']}   -> {m['compression_ratio']:.2f}x fewer")
    print(f"fast_recon_rmse    {m['fast_recon_rmse']:.4f}")
    print(f"naive_recon_rmse   {m['naive_recon_rmse']:.4f}   (comparable)")
    print(f"coeff_zero_frac    {m['coeff_zero_frac']:.2f}   (share of quantized coeffs that are 0)")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: quantizing DCT coefficients and quantizing raw samples cost the "
          "SAME error energy (the DCT is orthonormal — Parseval). So where did the tokens go? "
          "Look at coeff_zero_frac and what BPE does to a run of zeros.")
