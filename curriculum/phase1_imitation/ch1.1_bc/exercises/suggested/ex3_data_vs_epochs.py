"""SUGGESTED exercise candidate (humans promote) — hyperparameter-investigation, ch1.1.

Objective tested: the model region's claim that the ceiling of behavior
cloning is the data. Here you buy the same 10x compute two different ways
and see which one the success rate cares about.

THE QUESTION. Starting from a deliberately starved baseline (20 demos,
60 epochs), you may 10x exactly one axis:

    ARM A: 10x the data    (200 demos, 60 epochs)
    ARM B: 10x the epochs  (20 demos, 600 epochs)

PREDICT before you run: which arm wins on rollout success rate, and is the
loser's val loss higher or lower than the winner's? Write both parts of your
prediction in PREDICTION below, then run this file (a few minutes on CPU —
it trains three policies).

Estimated learner time: 30 minutes (mostly waiting on the runs).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- e.g. "A wins because ...; B's val loss is lower/higher because ..."

METADATA = {
    "type": "hyperparameter-investigation",
    "chapter": "ch1.1-bc",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase1_imitation/ch1.1_bc/bc.py"
GEN_DEMOS = REPO / "curriculum/common/envs/pusht/gen_demos.py"

# One compute knob held small so three runs fit in minutes: fewer eval
# episodes and a narrower net than the chapter defaults. The RANKING is what
# this exercise measures, not the chapter's headline number.
COMMON = ["--hidden_dim", "256", "--eval_episodes", "20", "--no-rerun", "--seed", "0"]
ARMS = {
    "baseline (20 demos, 60 epochs)": (20, 60),
    "A: 10x data (200 demos, 60 epochs)": (200, 60),
    "B: 10x epochs (20 demos, 600 epochs)": (20, 600),
}


def train_arm(episodes: int, epochs: int, workdir: Path) -> dict:
    """Generate `episodes` demos, train for `epochs`, return bc.py's metrics."""
    data = workdir / f"demos{episodes}"
    if not data.is_dir():
        subprocess.run([sys.executable, str(GEN_DEMOS), "--episodes", str(episodes),
                        "--seed", "0", "--out", str(data), "--no-video"],
                       check=True, capture_output=True, cwd=REPO)
    out = workdir / f"run_{episodes}ep_{epochs}epochs"
    subprocess.run([sys.executable, str(ARTIFACT), "--data", str(data), "--out", str(out),
                    "--epochs", str(epochs), *COMMON],
                   check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


def measure(workdir: Path | None = None) -> dict[str, dict]:
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-ex3-"))
    return {name: train_arm(episodes, epochs, workdir)
            for name, (episodes, epochs) in ARMS.items()}


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    results = measure()
    for name, metrics in results.items():
        print(f"{name:42s} success {metrics['success_rate']:.2f}  "
              f"val_loss {metrics['final_val_loss']:.4f}")
    print("\nNow reconcile: did the winner win the way you predicted? In one "
          "sentence, why can B's val loss look competitive while its rollouts are not?")
