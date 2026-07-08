"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch3.6.

Objective tested: the FULL CIRCLE and its honesty. You built a physics engine
(ch3.3-3.5). You re-created PushT in it. Now you run the ch1.1 BC policy — trained
in MuJoCo — in the engine YOU built, from bit-identical starts. What happens to a
policy when the world underneath it is a simplified, from-scratch approximation?

THE EXPERIMENT: run the SAME trained BC policy in BOTH sims for the smoke-sized
comparison (a handful of episodes), and read the success rate in MuJoCo (ground
truth) vs in your engine, plus the block-pose divergence (position and ANGLE).

PREDICT before you run: how does the policy transfer to your engine?
  A) perfectly — a policy is a function obs->action; feed it the same obs contract
     and it must reach the goal exactly as often, in any correct sim
  B) it fails completely — your engine is not MuJoCo, so success drops to ~0 and the
     block goes nowhere near the target
  C) it PARTLY works — success drops (roughly a third of MuJoCo's), and the
     trajectories start identical then diverge, MOST of all in ANGLE, because your
     frictionless two-point-mass block rotates far worse than MuJoCo's shaped one

Record your answer in PREDICTION below, then run this file (needs a trained ch1.1
policy — see find_policy; if none is found the run explains how to make one).

Before you run, write one sentence: WHY — what about a frictionless two-point-mass block
would make its ANGLE harder to reproduce than its position?

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
    "chapter": "ch3.6-compare",
    "choices": ["A", "B", "C"],
    "gate_before_run": True,
}

CHAPTER = Path(__file__).resolve().parents[2]
ARTIFACT = CHAPTER / "compare.py"
REPO_ROOT = CHAPTER.parents[2]


def find_policy() -> Path | None:
    """ch1.1's canonical trained TorchScript policy, if a REAL one exists (this chapter
    runs that exact checkpoint — the 500-demo one ch1.1 saves — not a policy of its own).

    QUALITY GUARD: ch1.1's DEFAULT --out is this very path, so a `bc.py --smoke` run
    leaves an UNTRAINED smoke checkpoint sitting right here (metrics.json: smoke=true,
    success_rate=0). Running it would silently print a meaningless ~0 transfer, so we
    read the sibling metrics.json and treat a smoke/zero-success checkpoint as NO
    policy — the caller then emits the 'train the real ch1.1 policy' guidance."""
    cand = REPO_ROOT / "outputs/ch1.1-bc/bc_policy.ts.pt"
    if not cand.is_file():
        return None
    sib = cand.parent / "metrics.json"
    if sib.is_file():
        try:
            meta = json.loads(sib.read_text())
        except (json.JSONDecodeError, OSError):
            meta = {}
        if meta.get("smoke") or meta.get("success_rate", 0) == 0:
            return None  # smoke/untrained: not a real full-circle policy
    return cand


def run_compare(policy: Path, *extra: str) -> dict:
    """Run compare.py to a temp dir with a trained policy; return its metrics."""
    with tempfile.TemporaryDirectory(prefix="z2r-ex1-") as tmp:
        cmd = [sys.executable, str(ARTIFACT), "--policy", str(policy),
               "--seed", "0", "--no-rerun", "--out", tmp, *extra]
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO_ROOT)
        return json.loads((Path(tmp) / "metrics.json").read_text())


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    policy = find_policy()
    if policy is None:
        raise SystemExit(
            "No trained policy found. Train ch1.1's canonical policy first (this chapter runs it):\n"
            "  python curriculum/common/envs/pusht/gen_demos.py --episodes 500 --seed 0 --out outputs/pusht-demos --no-video\n"
            "  python curriculum/phase1_imitation/ch1.1_bc/bc.py --data outputs/pusht-demos --out outputs/ch1.1-bc --device cpu --no-rerun")
    m = run_compare(policy, "--episodes", "20")
    print(f"MuJoCo (ground truth) BC success : {m['mj_success_rate']:.2f}")
    print(f"your-engine           BC success : {m['engine_success_rate']:.2f}")
    print(f"open-loop divergence  position   : {m['mean_pos_divergence_m']:.4f} m")
    print(f"open-loop divergence  ANGLE      : {m['mean_ang_divergence_rad']:.4f} rad  <- the dominant gap")
    print(f"your prediction: {PREDICTION} — now say WHY the angle diverges more than the position.")
