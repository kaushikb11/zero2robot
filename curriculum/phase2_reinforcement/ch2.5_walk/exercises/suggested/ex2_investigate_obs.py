"""SUGGESTED exercise candidate (humans promote) — observation-design investigation, ch2.5.

Objective tested: OBSERVATION-SPACE DESIGN, the chapter's second theme. A
locomotion policy is only as good as its senses — so how much does one coordinate
matter? The env's obs[23] hands the policy its torso linear velocity (entries
20..22, entry 20 is the forward speed vx the reward pays for). The `--blind_velocity`
flag zeroes those three numbers before the policy sees them: the robot must now
walk WITHOUT a direct sense of its own forward momentum. This is a design
investigation, not a bug-hunt (ch2.1 spike, H1): you form a directional
hypothesis and read it against a seed-robust signal.

THE KNOB. `--blind_velocity` hides obs 20..22 from the policy (training AND eval).
Everything else — joint angles, joint velocities, height, up-vector — is
unchanged, so the policy keeps rich proprioception; it only loses the torso-frame
velocity.

PREDICT before you run: relative to the full observation, does blinding the torso
velocity (a) clearly hurt the emergent gait (less forward distance), (b) make
little measurable difference at this budget (joint velocities already carry enough
signal to infer motion), or (c) HELP? Write your choice and a one-sentence
mechanism in PREDICTION.

Then run this file. It trains the full-obs policy and the velocity-blinded policy
on seeds 0, 1, 2 and prints each arm's per-seed forward distance and mean. Read
the MEANS: a real observation-design effect has to show up in the average, not in
one cherry-picked seed — which is why the graded check asserts the full-obs
policy's strong emergence over seeds, and treats the blinding effect as an
observation you interpret.

Estimated learner time: 35 minutes (six short SAC runs).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause

METADATA = {
    "type": "hyperparameter-investigation",
    "chapter": "ch2.5-walk",
    "knob": "--blind_velocity (hide torso linear velocity, obs 20..22)",
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase2_reinforcement/ch2.5_walk/walk.py"
SEEDS = (0, 1, 2)
EXERCISE_STEPS = 25_000
ARMS = {"full_obs": [], "blind_velocity": ["--blind_velocity"]}


def train_arm(flags: list[str], seed: int, workdir: Path) -> float:
    out = workdir / f"seed{seed}"
    subprocess.run(
        [sys.executable, str(ARTIFACT), "--seed", str(seed), "--device", "cpu",
         "--total_steps", str(EXERCISE_STEPS), "--no-rerun", "--out", str(out), *flags],
        check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())["mean_eval_forward_dist"]


def measure(workdir: Path | None = None) -> dict[str, list[float]]:
    """Return {arm_name: [eval forward distance per seed]}. Deterministic per seed."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-walk-ex2-"))
    return {name: [train_arm(flags, seed, workdir / name) for seed in SEEDS]
            for name, flags in ARMS.items()}


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    results = measure()
    for name, dists in results.items():
        mean = sum(dists) / len(dists)
        print(f"{name:15s} per-seed {[round(d, 3) for d in dists]}  mean {mean:+.3f} m")
    print("\nReconcile: did hiding the torso velocity move the MEAN forward "
          "distance, and in the direction you predicted? If the effect is small "
          "at this budget — because the joint velocities already let the policy "
          "infer its motion — that is itself a finding about observation design. "
          "Say so honestly.")
