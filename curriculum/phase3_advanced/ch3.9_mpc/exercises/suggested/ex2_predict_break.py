"""SUGGESTED exercise candidate (humans promote) — predict-then-run + BREAK IT, ch3.9.

Objective tested: the HONEST CEILING. MPC is not magic — it works because the plan
looks far enough ahead, with enough tries, to find the energy-pumping swing. Take
either away and it fails. This is the chapter's `--break`: you GENERATE the failure
yourself and measure it, rather than taking the ceiling on faith.

`--break horizon` drops the planning horizon from 25 steps to 3. Everything else is
unchanged — same model, same sampler, same cost. The planner still optimizes; it just
cannot SEE far enough. A 3-step plan cannot tell that swinging the cart the "wrong"
way now (letting the pole fall further) is what builds the momentum to come up later.

THE EXPERIMENT: run MPC normally, then with `--break horizon`, from the same start,
and compare the upright fraction.

PREDICT before you run: what does the too-short horizon do?
  A) no change — MPC re-plans every step, so a short horizon just re-decides more often
     and still swings up
  B) it FAILS — the pole never comes up (upright_frac -> 0.0). A myopic plan greedily
     reduces cost right now and never pays the short-term cost that buys the swing
  C) it gets WORSE but still solves it eventually — upright_frac drops to ~0.5

Record your answer in PREDICTION below, then run this file. (Try `--break samples`
too — 3 samples instead of 64 — for the other way to cripple the search.)

Before you run, write one sentence: WHY is swing-up a problem where acting greedily on
a short horizon is exactly wrong?

Estimated learner time: 15 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch3.9-mpc",
    "choices": ["A", "B", "C"],
    "gate_before_run": True,
}

CHAPTER = Path(__file__).resolve().parents[2]
ARTIFACT = CHAPTER / "mpc.py"
REPO_ROOT = CHAPTER.parents[2]


def run_mpc(*extra: str) -> dict:
    """Run mpc.py to a temp dir; return its metrics.json."""
    with tempfile.TemporaryDirectory(prefix="z2r-ch39-ex2-") as tmp:
        cmd = [sys.executable, str(ARTIFACT), "--method", "cem",
               "--seed", "0", "--no-rerun", "--out", tmp, *extra]
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO_ROOT)
        return json.loads((Path(tmp) / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    full = run_mpc()
    broke = run_mpc("--break", "horizon")
    print(f"full plan   (H=25)  upright_frac {full['mpc_upright_frac']:.2f}  mean_cost {full['mpc_mean_cost']:.3f}")
    print(f"--break horizon (H=3) upright_frac {broke['mpc_upright_frac']:.2f}  mean_cost {broke['mpc_mean_cost']:.3f}")
    print(f"your prediction: {PREDICTION} — now say WHY a greedy short horizon cannot solve swing-up.")
