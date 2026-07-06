"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch1.2.

Objective tested: the chapter's central claim — that curating your data (which
THROWS EPISODES AWAY) can beat training on everything you recorded.

THE SETUP. curate.py builds a raw dataset of 500 episodes: half careful
(mostly reach the goal), half sloppy (mostly wander and fail). The honest
filter keeps only the episodes that reached the goal — fewer than half — and
re-trains chapter 1.1's BC on that smaller set. Both policies are evaluated on
the same held-out reset seeds.

PREDICT before you run: the curated policy trains on FEWER episodes than raw.
On held-out success rate...
  A) raw wins — more data is more data; you never help BC by deleting demos
  B) curated wins despite the smaller dataset — the sloppy episodes were
     poisoning the states they covered, and dropping them is worth more than
     the frames you lose
  C) they tie — BC averages, so a few bad demos wash out and change nothing

Record your answer in PREDICTION below, then run this file (~5 minutes on CPU —
it trains two policies at the chapter's default scale).
Estimated learner time: 20 minutes (mostly waiting on the runs).
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
CFG = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text())
RC = CFG["exercise_checks"]["exercise_config"]


def config_flags() -> list[str]:
    return ["--careful", str(RC["careful"]), "--sloppy", str(RC["sloppy"]),
            "--epochs", str(RC["epochs"]), "--eval_episodes", str(RC["eval_episodes"]),
            "--hidden_dim", str(RC["hidden_dim"])]


def run_curate(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(CURATE), "--seed", "0", "--no-rerun",
           "--out", str(out), *config_flags(), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        m = run_curate(Path(tmp) / "run")
    print(f"raw     ({m['n_episodes']} episodes) -> success {m['raw_success_rate']:.3f}")
    print(f"curated ({m['n_kept']} episodes)      -> success {m['curated_success_rate']:.3f}")
    print(f"delta = {m['delta_success_rate']:+.3f}   (your prediction: {PREDICTION})")
    print("\nNow explain it to yourself: the curated set is smaller. Where did the win come from?")
