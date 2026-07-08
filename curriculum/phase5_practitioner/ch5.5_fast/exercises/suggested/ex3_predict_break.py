"""SUGGESTED exercise candidate (humans promote) — predict-then-run + the learner-generated
failure, ch5.5.

Objective tested: WHERE the compression comes from — the DCT basis, or the BPE? It is tempting
to think BPE is doing the work and the transform is cosmetic. This exercise makes you generate
the failure that settles it.

The FAST codec spends its token budget on a few LOW-FREQUENCY coefficients (frequency domain).
The `--break time_domain` flag spends the SAME kind of budget in the TIME domain instead: it
keeps every Nth action and zero-order-HOLDS it (a crude time-domain low-pass), then quantizes
the staircase. Same idea ("keep fewer numbers, reconstruct"), different basis.

PREDICT before you run: compared to the clean FAST codec, what does `--break time_domain` do to
the RECONSTRUCTION (fast_recon_rmse) and its SMOOTHNESS (fast_error_jerk)?
  A) Both stay about the same — a basis is a basis; keeping fewer numbers in time is as good as
     keeping fewer in frequency.
  B) It IMPROVES both — the time domain is the "natural" domain for a trajectory, so skipping
     the transform can only help.
  C) Both get much WORSE — a robot trajectory's information is spread across every timestep, so
     dropping timesteps and holding craters the error (RMSE ~15x) and turns the reconstruction
     into a jerky staircase (error-jerk ~200x). The DCT basis is load-bearing, not cosmetic.

Record your answer in PREDICTION, then run this file: it runs the codec clean AND with
`--break time_domain` (pure numpy, well under a second each) and prints both. Then GENERATE the
failure yourself and read it in rerun: run
    python curriculum/phase5_practitioner/ch5.5_fast/fast.py --seed 0 --break time_domain
and compare the toy/reconstruction stream to the clean run. Estimated learner time: 15 minutes.
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
        clean = run_fast(Path(tmp) / "clean")
        broke = run_fast(Path(tmp) / "break", ["--break", "time_domain"])
    print("                    clean        --break time_domain")
    print(f"fast_recon_rmse     {clean['fast_recon_rmse']:.4f}       {broke['fast_recon_rmse']:.4f}"
          f"   ({broke['fast_recon_rmse'] / clean['fast_recon_rmse']:.0f}x worse)")
    print(f"fast_error_jerk     {clean['fast_error_jerk']:.5f}      {broke['fast_error_jerk']:.5f}"
          f"   ({broke['fast_error_jerk'] / clean['fast_error_jerk']:.0f}x worse)")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: the DCT and the time-domain shortcut both 'keep fewer numbers'. Why "
          "does keeping low-frequency COEFFICIENTS rebuild the motion while keeping every Nth "
          "SAMPLE destroys it? What does that say about where a trajectory's information lives?")
