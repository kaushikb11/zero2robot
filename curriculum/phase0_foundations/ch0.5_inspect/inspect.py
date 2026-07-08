"""zero2robot 0.5 — Seeing Like a Robot (rerun).

You made a dataset in 0.4. But a directory of parquet files is not something you
can eyeball, and "I recorded some episodes" is not the same as knowing what is
IN them. How many of your demonstrations actually reached the target? How long
are they? Is the block's orientation what you think it is? This chapter is the
debugging tool for the rest of the book: load a LeRobot v3 dataset back, read it
like a story, and log it to rerun so you can scrub a whole episode on a timeline.

The one idea to hold onto: a dataset does NOT store success or reward — 0.4 only
wrote `observation.state` and `action`. So the "success rate hiding in your data"
is not a column you look up; you RECONSTRUCT it from the observations, with the
env's own POS_TOL / ANG_TOL, by reading whether each demonstration ended with the
block on the target. To read like a robot is to decode the numbers the robot saw
— including the yaw, stored as a sin/cos pair precisely so it never wraps.

It inspects two kinds of dataset, and always has one to show:

  (a) YOUR data   — point --dataset at the ch0.4 output you recorded.
  (b) a stand-in  — with no --dataset, it drives the shared scripted expert
      (curriculum.common, decision 004) into a small dataset and inspects THAT,
      so the chapter runs on a fresh clone and in CI with nothing recorded yet.

Run it (your data):  python inspect.py --dataset ../ch0.4_record/outputs/ch0.4-record/dataset
Run it (stand-in):   python inspect.py --episodes 6
CI smoke:            python inspect.py --smoke --seed 0 --no-rerun
Break it:            python inspect.py --break yaw-swap        # decode the yaw wrong
"""

# --- region: setup ---
import os
import sys

# This file is named inspect.py, which collides with the STDLIB `inspect`
# module: running `python inspect.py` seeds sys.path[0] with this file's own
# directory, so numpy/mujoco's `import inspect` would find THIS file instead of
# the standard library and crash. Drop our own dir from the path (FIRST, before
# any heavy import) and add the repo root so `curriculum.common` resolves — the
# same root-on-path pattern ch0.1/0.4 use, minus the self-shadowing.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

# The env is shared reference code (decision 004): we import its success
# constants so the reconstruction below uses the SAME tolerances the recorder's
# env used, never a second copy that could silently disagree.
from curriculum.common.device import banner  # noqa: E402
from curriculum.common.envs.pusht.pusht_env import PushTEnv, wrap_angle  # noqa: E402

# The three break modes are real misconceptions you can inject and MEASURE, not
# toy typos. Each corrupts how you READ the data and leaves an honest signature.
BREAK_MODES = ("none", "yaw-swap", "drop-boundaries")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=0, help="seeds the stand-in generator; two --seed 0 runs produce byte-identical metrics.json")
parser.add_argument("--smoke", action="store_true", help="tiny fixed-length stand-in for CI; two --smoke runs must match byte-for-byte")
parser.add_argument("--out", type=Path, default=Path("outputs/ch0.5-inspect"), help="run dir: metrics.json here, a provisioned stand-in in {out}/dataset")
parser.add_argument("--dataset", type=Path, default=None, help="a LeRobot v3 dataset root to inspect (e.g. your ch0.4 output); default: provision a stand-in")
parser.add_argument("--episodes", type=int, default=6, help="stand-in only: how many episodes to generate (ignored when --dataset is given)")
parser.add_argument("--break", dest="break_mode", choices=BREAK_MODES, default="none", help="inject a reading misconception: yaw-swap | drop-boundaries")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)  # replaying your episode to rerun is the default; opt OUT, not in
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip the .rrd (CI smoke)")
args = parser.parse_args()

banner("ch0.5-inspect", device="cpu")  # pure-numpy/mujoco-CPU: honest cpu tier, never the host's mps/cuda. startup contract: tier + measured wall-clock to stdout
rng = np.random.default_rng(args.seed)  # PCG64 — only used by the stand-in generator
# --- endregion ---

# --- region: provision ---
# The stand-in. When you have no --dataset yet, we manufacture one exactly the
# way 0.4 did — drive the shared scripted expert and write via lerobot — so the
# thing you inspect is a real LeRobot v3 dataset, not a mock. Noise is on so the
# expert sometimes misses: an inspector is only interesting if the data is
# imperfect, which real teleop data always is.
TASK = "Push the T-shaped block to the target pose."
STATE_NAMES = [
    "pusher_x", "pusher_y", "tee_x", "tee_y", "sin_tee_yaw", "cos_tee_yaw",
    "target_x", "target_y", "sin_target_yaw", "cos_target_yaw",
]


def provision_dataset(root: Path, episodes: int, seed: int, smoke: bool):
    """Generate a small PushT dataset to inspect. Returns nothing — it writes a
    v3 dataset at `root`, which the loader below reads back like any other."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # lazy: pulls in torch
    from curriculum.common.envs.pusht.scripted_expert import ScriptedExpert

    features = {
        "observation.state": {"dtype": "float32", "shape": (PushTEnv.OBS_DIM,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (PushTEnv.ACT_DIM,), "names": ["pusher_vx", "pusher_vy"]},
    }
    dataset = LeRobotDataset.create(repo_id="zero2robot/pusht_standin", fps=PushTEnv.CONTROL_HZ,
                                    features=features, root=root, robot_type="pusher_2d", use_videos=False)
    env = PushTEnv()
    max_len = 40 if smoke else PushTEnv.MAX_STEPS  # smoke length is capped so CI stays fast and fixed
    for i in range(episodes):
        obs = env.reset(seed + i)              # seed+i per episode: reproducible, distinct starts
        expert = ScriptedExpert(noise=0.08, seed=seed + i)  # imperfect on purpose
        for _ in range(max_len):
            action = expert.action(env)
            dataset.add_frame({"observation.state": obs, "action": action, "task": TASK})
            obs, _, done, _ = env.step(action)
            if done and not smoke:             # natural end at success/timeout; smoke stays fixed-length
                break
        dataset.save_episode()
    dataset.finalize()
# --- endregion ---

# --- region: load ---
# Reading a dataset back is two facts: the SCHEMA (what every frame is) and the
# EPISODE INDEX (where one demonstration ends and the next begins). lerobot's
# meta.episodes carries per-episode [from, to) row ranges — an episode is a
# contiguous run of frames, nothing more. We slice the frame table by those
# ranges into plain numpy so the rest of the file never touches a tensor.
def load_episodes(dataset_root: Path, break_mode: str):
    """Return (info, episodes). info: schema facts. episodes: a list of dicts
    with numpy `states` (N,10) and `actions` (N,2), one per demonstration."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id="zero2robot/inspect", root=dataset_root)
    frames = dataset.hf_dataset
    info = {
        "obs_dim": len(dataset.meta.features["observation.state"]["names"]),
        "act_dim": len(dataset.meta.features["action"]["names"]),
        "fps": int(dataset.meta.fps),
        "feature_keys": sorted(k for k in dataset.meta.features if k.startswith("observation") or k == "action"),
    }

    def slice_episode(lo: int, hi: int) -> dict:
        rows = frames[lo:hi]
        return {"states": np.stack([np.asarray(s, np.float32) for s in rows["observation.state"]]),
                "actions": np.stack([np.asarray(a, np.float32) for a in rows["action"]])}

    if break_mode == "drop-boundaries":
        # THE BUG: ignore episode_index and treat the whole table as ONE episode.
        # Everyone writes this once — you forget demonstrations have boundaries.
        episodes = [slice_episode(0, dataset.num_frames)]
    else:
        episodes = [slice_episode(int(ep["dataset_from_index"]), int(ep["dataset_to_index"]))
                    for ep in dataset.meta.episodes]
    return info, episodes
# --- endregion ---

# --- region: success ---
# Seeing like a robot: the success signal is not stored, so decode it from the
# state. tee_yaw lives as (sin, cos) at obs[4:6] to dodge the +/-pi wrap you met
# in 0.3; the ONE correct way back to an angle is atan2(sin, cos) — arguments in
# that order. Get them backwards and every orientation you read is reflected.
def decode_tee_yaw(state: np.ndarray, break_mode: str) -> float:
    sin_yaw, cos_yaw = float(state[4]), float(state[5])
    if break_mode == "yaw-swap":
        # THE BUG: atan2(cos, sin) reads the reflected angle (pi/2 - true). The
        # block will look rotated in rerun, and every ang_err below is wrong.
        return math.atan2(cos_yaw, sin_yaw)
    return math.atan2(sin_yaw, cos_yaw)


def frame_errors(state: np.ndarray, break_mode: str) -> tuple[float, float]:
    """Reconstruct the env's (pos_err, ang_err) from one stored observation.
    Target is fixed at the origin with yaw 0 (obs[6:10]); the block is obs[2:6]."""
    pos_err = float(np.hypot(state[2] - state[6], state[3] - state[7]))
    ang_err = float(abs(wrap_angle(decode_tee_yaw(state, break_mode))))  # target yaw is 0
    return pos_err, ang_err


def episode_reached(states: np.ndarray, break_mode: str) -> bool:
    """Did this demonstration reach the goal? The recorder STOPS the instant the
    env latches success (it breaks on `done`), so the last frame it stored is the
    block sitting on the target — reading 'reached' is reading whether the episode
    ENDED in tolerance. The dataset never says 'success'; this decode IS the
    reading, and it uses the env's own POS_TOL / ANG_TOL, not a second copy."""
    pos_err, ang_err = frame_errors(states[-1], break_mode)
    return pos_err < PushTEnv.POS_TOL and ang_err < PushTEnv.ANG_TOL
# --- endregion ---

# --- region: inspect ---
# The walk: turn a list of episodes into the handful of numbers you actually
# wanted — lengths, the reconstructed success rate, and the terminal error of
# each demonstration. This is what "looking at your dataset" concretely means.
def inspect_episodes(episodes: list, break_mode: str) -> dict:
    per_episode = []
    for states in (ep["states"] for ep in episodes):
        pos_err, ang_err = frame_errors(states[-1], break_mode)  # terminal frame = where the demo left the block
        per_episode.append({
            "length": int(len(states)),
            "reached": bool(episode_reached(states, break_mode)),
            "final_pos_err": round(pos_err, 6),
            "final_ang_err": round(ang_err, 6),
        })
    lengths = [rec["length"] for rec in per_episode]
    n_reached = sum(rec["reached"] for rec in per_episode)
    return {
        "n_episodes": len(per_episode),
        "n_frames": int(sum(lengths)),
        "episode_lengths": lengths,
        "reached": [rec["reached"] for rec in per_episode],
        "n_reached": int(n_reached),
        "success_rate": round(n_reached / len(per_episode), 6) if per_episode else 0.0,
        "mean_episode_length": round(float(np.mean(lengths)), 6) if lengths else 0.0,
        "final_pos_err": [rec["final_pos_err"] for rec in per_episode],
        "final_ang_err": [rec["final_ang_err"] for rec in per_episode],
    }
# --- endregion ---

# --- region: rerun ---
# Replay every stored observation onto the canonical entity paths (the SAME ones
# every chapter uses, per .claude/skills/rerun-instrument), so you scrub your own
# episodes on the sim_time timeline and watch the pusher worry the block toward
# the green target. The reconstructed pos_err/ang_err ride along as scalars — and
# under --break yaw-swap the tee visibly points the wrong way.
_TEE_CENTERS = [(0.0, 0.0, 0.0), (0.0, -0.06, 0.0)]      # the T is two boxes (see pusht.xml)
_TEE_HALF = [(0.06, 0.015, 0.015), (0.015, 0.045, 0.015)]


def log_rerun(episodes: list, fps: int, break_mode: str, path: Path):
    import rerun as rr  # lazy: --no-rerun (CI) never imports it

    rr.init("zero2robot/ch0.5-inspect", spawn=False)
    rr.save(str(path))
    gx, gy, _ = PushTEnv.TARGET_POSE
    rr.log("world/objects/target", rr.Boxes3D(centers=[(gx + c[0], gy + c[1], 0.0) for c in _TEE_CENTERS],
                                              half_sizes=_TEE_HALF, colors=(90, 205, 100, 120)), static=True)
    rr.log("world/objects/tee", rr.Boxes3D(centers=_TEE_CENTERS, half_sizes=_TEE_HALF, colors=(115, 128, 242)), static=True)
    rr.log("world/robot/pusher", rr.Cylinders3D(lengths=[0.04], radii=[0.015], colors=(230, 102, 90)), static=True)
    frame_index = 0
    for episode_index, ep in enumerate(episodes):
        for state, action in zip(ep["states"], ep["actions"]):
            rr.set_time("sim_time", duration=frame_index / fps)
            frame_index += 1
            tee_yaw = decode_tee_yaw(state, break_mode)  # the decode under test — reflected under yaw-swap
            pos_err, ang_err = frame_errors(state, break_mode)
            rr.log("world/objects/tee", rr.Transform3D(translation=(float(state[2]), float(state[3]), 0.0152),
                                                       rotation=rr.RotationAxisAngle(axis=(0, 0, 1), radians=tee_yaw)))
            rr.log("world/robot/pusher", rr.Transform3D(translation=(float(state[0]), float(state[1]), 0.02)))
            rr.log("policy/action", rr.Scalars(np.asarray(action, dtype=np.float64)))
            rr.log("eval/pos_err", rr.Scalars([pos_err]))
            rr.log("eval/ang_err", rr.Scalars([ang_err]))
            rr.log("eval/episode_index", rr.Scalars([float(episode_index)]))
# --- endregion ---

# --- region: run ---
args.out.mkdir(parents=True, exist_ok=True)
if args.dataset is not None:
    source, dataset_root = "dataset", args.dataset  # inspect the data you recorded
else:
    source, dataset_root = "provisioned", args.out / "dataset"
    provision_dataset(dataset_root, args.episodes, args.seed, args.smoke)

info, episodes = load_episodes(dataset_root, args.break_mode)
summary = inspect_episodes(episodes, args.break_mode)

# metrics.json: the determinism-checked reading. The dataset embeds a uuid and
# absolute paths (not byte-stable), but these RECONSTRUCTED facts are — two
# --smoke --seed 0 runs produce identical ones.
metrics = {"source": source, "seed": args.seed, "break_mode": args.break_mode, **info, **summary}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

print(f"inspected {summary['n_episodes']} episodes / {summary['n_frames']} frames from {dataset_root}")
print(f"schema: observation.state[{info['obs_dim']}] + action[{info['act_dim']}] @ {info['fps']} Hz  keys={info['feature_keys']}")
print(f"episode lengths: {summary['episode_lengths']}  (mean {summary['mean_episode_length']:g})")
print(f"reached target: {summary['n_reached']}/{summary['n_episodes']}  success_rate={summary['success_rate']:g}")
if args.break_mode != "none":
    print(f"[break={args.break_mode}] you are reading the data WRONG on purpose — compare this success_rate to --break none")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    rrd = args.out / "inspect.rrd"
    log_rerun(episodes, info["fps"], args.break_mode, rrd)
    print(f"recording: {rrd} — open it with: rerun {rrd}")
# --- endregion ---
