"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch1.6.

This is the chapter's flagship: the harness graded by the harness. Chapters
1.3-1.5 all reported a headline like "policy A 0.30 beats policy B 0.20" over 20
rollouts. Here you decide, BEFORE running, whether such a ranking is actually
established — and then let the harness compute the confidence intervals and tell
you.

THE SETUP. harness.py trains two real BC policies — a STRONG one (more demos)
and a WEAK one (fewer demos) — and evaluates both. It reports the strong-vs-weak
success gap two ways:
  - at N = eval_episodes (one suite, e.g. 20 episodes: the number the arc reported)
  - at N = n_seeds * eval_episodes (every suite pooled, e.g. 120-200 episodes)
For each N it prints a Newcombe confidence interval on the DIFFERENCE p_strong -
p_weak. If that interval excludes 0, the ranking is significant; if it contains 0,
you have not established it, no matter how the point estimates look.

PREDICT before you run: what will the two verdicts be?
  A) Significant at BOTH N — the strong policy is clearly better, 20 episodes is plenty.
  B) NOT significant at the small N (the diff CI straddles 0), but significant once
     pooled to the large N — the gap is real; 20 episodes just could not see it.
  C) NOT significant at EITHER N — there is no real gap between the policies.

Record your answer in PREDICTION below, then run this file (~15 s on CPU at the
reduced config — it trains the two policies and evaluates them).
Estimated learner time: 15 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {"type": "predict-then-run", "chapter": "ch1.6-harness",
            "choices": ["A", "B", "C"], "gate_before_run": True}

REPO = Path(__file__).resolve().parents[5]
HARNESS = REPO / "curriculum/phase1_imitation/ch1.6_harness/harness.py"
RC = yaml.safe_load((Path(__file__).resolve().parents[2] / "meta.yaml").read_text())["exercise_checks"]["exercise_config"]


def config_flags() -> list[str]:
    return ["--num_demos", str(RC["num_demos"]), "--num_demos_weak", str(RC["num_demos_weak"]),
            "--hidden_dim", str(RC["hidden_dim"]), "--epochs", str(RC["epochs"]),
            "--eval_episodes", str(RC["eval_episodes"]), "--n_seeds", str(RC["n_seeds"])]


def run_harness(out: Path) -> dict:
    cmd = [sys.executable, str(HARNESS), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), *config_flags()]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    with tempfile.TemporaryDirectory() as tmp:
        m = run_harness(Path(tmp) / "run")
    print(f"N={m['n_small']:<4d} diff CI [{m['small_diff_ci_lo']:+.2f}, {m['small_diff_ci_hi']:+.2f}]  "
          f"significant={m['small_significant']}")
    print(f"N={m['n_pooled']:<4d} diff CI [{m['pooled_diff_ci_lo']:+.2f}, {m['pooled_diff_ci_hi']:+.2f}]  "
          f"significant={m['pooled_significant']}")
    print(f"(your prediction: {PREDICTION})")
    print("\nNow explain it: the point estimates barely moved between the two N. "
          "What shrank, and why did that flip the verdict?")
