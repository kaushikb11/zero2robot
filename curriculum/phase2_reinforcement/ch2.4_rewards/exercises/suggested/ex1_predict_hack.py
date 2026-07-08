"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch2.4.

Objective tested: the reward-hacking lesson. You reason about a NAIVE reward
BEFORE running it, then you run it and measure whether the policy did what you
meant or only what you said.

THE REWARD. `r_hack` pays the quadruped for one thing: raw torso height
(`HACK_HEIGHT_W * info["height"]`). No forward term at all. A plausible-sounding
proxy — "a walking robot holds itself up, so reward height."

PREDICT before you run: over the default training, which happens?
  (a) it learns to walk forward (height correlates with a good gait, so the proxy
      works out);
  (b) its reward climbs but it does NOT walk — it games the proxy (rears/stands
      tall) and covers ~0 forward distance;
  (c) the reward never rises — PPO can't optimize height at all.
Write your choice and one sentence of why in PREDICTION.

Then run this file. It trains the `hack` design and prints the hack's forward
distance (what you MEANT: go forward) next to how much the hack's own reward rose
(what you SAID: be tall). Reconcile your prediction with the two numbers.

This trains PPO, but every RNG is seeded, so a fixed seed reproduces exactly. The
seed-ROBUST signals (asserted in checks.py) are the ORDERING — the reward rises
while forward distance stays near zero — not the exact magnitudes.

Estimated learner time: 20 minutes (mostly waiting on one training run).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause, e.g. "b because ..."

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch2.4-rewards",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase2_reinforcement/ch2.4_rewards/rewards.py"
# Reduced step budget so the check is affordable; the hack mismatch is already
# clear here (measured at the default config too — see meta.yaml provenance).
STEPS = "150000"


def measure(workdir: Path | None = None) -> dict:
    """Train the hack design once (seeded => reproducible) and return its metrics:
    forward_m (intended behaviour) and the train-return rise (its own reward)."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-rewards-ex1-"))
    subprocess.run(
        [sys.executable, str(ARTIFACT), "--design", "hack", "--seed", "0",
         "--device", "cpu", "--no-rerun", "--total_steps", STEPS, "--out", str(workdir)],
        check=True, capture_output=True, cwd=REPO,
    )
    return json.loads((workdir / "metrics.json").read_text())["designs"]["hack"]


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    hack = measure()
    rise = hack["train_return_last"] / max(1e-6, abs(hack["train_return_first"]))
    print(f"hack forward distance:  {hack['forward_m']:+.3f} m   (what you MEANT: go forward)")
    print(f"hack mean torso height: {hack['height_m']:.3f} m     (what it optimized: be tall)")
    print(f"hack reward rose:       {hack['train_return_first']:.1f} -> {hack['train_return_last']:.1f} "
          f"({rise:.1f}x)   (what you SAID)")
    print("\nBefore you read the reconcile below — in one sentence, why did a reward that "
          "never mentions forward motion still let the robot 'succeed'?")
    print(f"\nReconcile: the reward went up {rise:.1f}x while forward distance stayed a small "
          f"fraction of the shaped walk (~4.6 m): hack forward = {hack['forward_m']:+.2f} m. "
          "The policy did exactly what you SAID (be tall), not what you MEANT (walk). "
          "That is a reward hack — specification gaming.")
