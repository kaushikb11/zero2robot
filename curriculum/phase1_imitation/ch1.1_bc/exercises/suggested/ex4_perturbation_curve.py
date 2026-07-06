"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch1.1.

Objective tested: diagnose covariate shift — this time by inducing it on
purpose and watching how dataset size changes the blast radius.

THE SETUP. Two policies, trained identically except for demo count (50 vs
200). During each eval episode we shove the T-block sideways by `delta`
meters at control step 25 — a state the policy has to recover from, exactly
like a visitor dragging the block in the playground demo. We sweep delta
and measure success at each size.

PREDICT before you run: sketch success-vs-delta for both policies. Which
statement matches your sketch?

  A) The shove dominates: once the block teleports, both policies are off
     their data and fail about equally — demo count stops mattering
  B) Demo count dominates: the 200-demo policy stays above the 50-demo
     policy at every shove size, shoved or not
  C) The gap WIDENS with delta — extra demos buy robustness precisely in
     the rarely-visited states shoves create
  D) The curves converge at large delta — far enough off-distribution,
     every BC policy is equally lost

Record your answer in PREDICTION, then run this file (several minutes on
CPU — it trains two policies and rolls out the whole sweep).

Estimated learner time: 35 minutes (mostly waiting on the runs).
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from curriculum.common.envs.pusht import PushTEnv  # noqa: E402

PREDICTION = None  # <- set to "A", "B", "C", or "D" BEFORE running

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch1.1-bc",
    "choices": ["A", "B", "C", "D"],
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase1_imitation/ch1.1_bc/bc.py"
GEN_DEMOS = REPO / "curriculum/common/envs/pusht/gen_demos.py"

DEMO_COUNTS = (50, 200)
DELTAS = (0.0, 0.03, 0.06, 0.10)   # shove distance in meters (block is ~0.12 across)
SHOVE_STEP = 25                    # control step at which the block teleports
EPISODES_PER_POINT = 15


def train_policy(episodes: int, workdir: Path) -> Path:
    """Train BC on `episodes` fresh demos; return the exported ONNX path."""
    data = workdir / f"demos{episodes}"
    if not data.is_dir():
        subprocess.run([sys.executable, str(GEN_DEMOS), "--episodes", str(episodes),
                        "--seed", "0", "--out", str(data), "--no-video"],
                       check=True, capture_output=True, cwd=REPO)
    out = workdir / f"bc_{episodes}"
    subprocess.run([sys.executable, str(ARTIFACT), "--data", str(data), "--out", str(out),
                    "--epochs", "300", "--hidden_dim", "256", "--eval_episodes", "5",
                    "--no-rerun", "--seed", "0"],
                   check=True, capture_output=True, cwd=REPO)
    return out / "bc_policy.onnx"


def shove_block(env: PushTEnv, delta: float, rng: np.random.Generator) -> None:
    """Teleport the block `delta` meters in a random direction (a mouse-drag
    in one instant). Pokes env internals on purpose — same spirit as ch0.1's
    xfrc_applied, but displacement is easier to dose than force."""
    import mujoco
    angle = rng.uniform(0.0, 2.0 * np.pi)
    env.data.qpos[env._jadr["tee_x"]] += delta * np.cos(angle)
    env.data.qpos[env._jadr["tee_y"]] += delta * np.sin(angle)
    mujoco.mj_forward(env.model, env.data)


def success_rate(onnx_path: Path, delta: float) -> float:
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    env = PushTEnv()
    successes = 0
    for episode in range(EPISODES_PER_POINT):
        rng = np.random.Generator(np.random.PCG64(episode))  # seeded shove direction
        obs = env.reset(seed=10_000 + episode)
        done, info = False, {}
        while not done:
            if env._step_count == SHOVE_STEP and delta > 0.0:
                shove_block(env, delta, rng)
                obs = env._obs()
            action = session.run(None, {"observation": obs[None]})[0][0]
            obs, _, done, info = env.step(action)
        successes += bool(info["success"])
    return successes / EPISODES_PER_POINT


def measure(workdir: Path | None = None) -> dict[int, list[float]]:
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-ex4-"))
    curves = {}
    for episodes in DEMO_COUNTS:
        onnx_path = train_policy(episodes, workdir)
        curves[episodes] = [success_rate(onnx_path, delta) for delta in DELTAS]
    return curves


if __name__ == "__main__":
    if PREDICTION not in METADATA["choices"]:
        raise SystemExit('set PREDICTION to "A", "B", "C", or "D" first — the run stays locked until you commit')
    print(f"your prediction: {PREDICTION}\n")
    curves = measure()
    print(f"{'delta (m)':>10s}  " + "  ".join(f"{d:.2f}" for d in DELTAS))
    for episodes, curve in curves.items():
        print(f"{episodes:4d} demos  " + "  ".join(f"{s:.2f}" for s in curve))
    print("\nCompare against your sketch. Two things to reconcile: the demo-count "
          "gap, and whatever parts of your sketch the noisy 15-episode estimate "
          "refuses to confirm (a shove is not always bad news for a weak policy — "
          "work out why before re-running with more episodes per point).")
