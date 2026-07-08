"""SUGGESTED exercise candidate (humans promote) — hyperparameter-investigation, ch3.6.

Objective tested: the sim-to-sim gap is a MODELING CHOICE you can measure yourself
moving. Your engine fakes PushT's planar surface friction as a viscous drag on the
block, `--block_damp`. MuJoCo's real tee is heavily damped and quasi-static (it
barely coasts). So `--block_damp` is a proxy for how well your engine matches that:
turn it up toward MuJoCo's draggy tee and the two sims should agree BETTER.

THE EXPERIMENT: replay MuJoCo's exact action sequences in your engine (open-loop)
at a LOW block drag and a HIGH block drag, and read the mean block-position
divergence at each. The divergence is the smooth, seed-robust number here (success
is a coarse 0/1 metric — do not read the gap off it).

PREDICT before you run: as `--block_damp` rises from 2 to 60, the divergence...
  A) rises — more drag means the block moves less like MuJoCo's, so they agree worse
  B) falls — more drag makes your block quasi-static like MuJoCo's tee, so the two
     sims track each other more closely (the gap narrows)
  C) does not move — friction is a detail; the divergence is set by the contact
     model alone and drag cannot touch it

Record your answer in PREDICTION below, then run this file (needs a trained ch1.1
policy — see ex1.find_policy).

Before you run, write one sentence: WHY — what does raising the block's drag do to how
far it coasts, and how does that move it toward or away from MuJoCo's quasi-static tee?

Estimated learner time: 15 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {
    "type": "hyperparameter-investigation",
    "chapter": "ch3.6-compare",
    "choices": ["A", "B", "C"],
    "gate_before_run": True,
}

CHAPTER = Path(__file__).resolve().parents[2]
ARTIFACT = CHAPTER / "compare.py"
REPO_ROOT = CHAPTER.parents[2]

LOW_DAMP, HIGH_DAMP = 2.0, 60.0


def find_policy() -> Path | None:
    # ch1.1's canonical 500-demo checkpoint — this chapter runs that exact policy.
    # QUALITY GUARD (see ex1.find_policy): ch1.1's default --out IS this path, so a
    # `bc.py --smoke` run drops an untrained smoke checkpoint here. Read the sibling
    # metrics.json and treat a smoke/zero-success checkpoint as NO policy, so the
    # caller emits the 'train the real ch1.1 policy' guidance instead of a bogus gap.
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


def divergence_at_damp(policy: Path, block_damp: float, episodes: int = 30) -> float:
    """Mean open-loop block-position divergence (m) at a given block drag."""
    with tempfile.TemporaryDirectory(prefix="z2r-ex3-") as tmp:
        cmd = [sys.executable, str(ARTIFACT), "--policy", str(policy), "--seed", "0",
               "--episodes", str(episodes), "--block_damp", str(block_damp),
               "--no-rerun", "--out", tmp]
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO_ROOT)
        return json.loads((Path(tmp) / "metrics.json").read_text())["mean_pos_divergence_m"]


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    policy = find_policy()
    if policy is None:
        raise SystemExit("No trained policy found — make one first (see ex1_predict_transfer.py).")
    low = divergence_at_damp(policy, LOW_DAMP)
    high = divergence_at_damp(policy, HIGH_DAMP)
    print(f"block_damp={LOW_DAMP:>4}  ->  mean position divergence {low:.4f} m")
    print(f"block_damp={HIGH_DAMP:>4}  ->  mean position divergence {high:.4f} m")
    print(f"raising the block drag {LOW_DAMP:g}->{HIGH_DAMP:g} narrows the gap {low / high:.2f}x")
    print(f"your prediction: {PREDICTION} — the gap is a knob, not a fact.")
