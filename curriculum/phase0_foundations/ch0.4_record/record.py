"""zero2robot 0.4 — Teleoperation & Your First Dataset.

A policy is only as good as the demonstrations you feed it, and a demonstration
is an EPISODE: a list of (observation, action) pairs recorded while something —
a human, a script, your mouse — drove the robot. This chapter records episodes
of the PushT task and writes them as a LeRobot v3 dataset, the exact format
chapter 1.1's behavior cloning trains on. The dataset you make here is the
artifact-continuity anchor of the whole book: it follows you to 0.5 (inspect
it), 1.1 (train on it), and 1.2 (curate it).

There are TWO ways to get episodes and ONE way to write them:

  (a) LOCAL teleop  — drive the PushT pusher yourself and record the rollout.
      Real interactive mouse teleop lives in the browser demo (drag to push);
      it can't be byte-reproducible, so the local path here scripts a
      deterministic stand-in for your mouse so --smoke and CI can diff two runs.
  (b) INGEST        — read the browser's `z2r-teleop-1` interchange bundle
      (interchange.json + PNG frames, per decision 008) that the playground
      hands you after a real mouse-teleop session.

Both paths funnel into the SAME canonical writer — LeRobotDataset.create ->
add_frame -> save_episode -> finalize, the identical library call as the
reference gen_demos.py. Format parity with the training datasets is therefore
free by construction, not a schema we re-implement and keep in sync.

Run it (local):   python record.py --episodes 3
Ingest a bundle:  python record.py --from-interchange /path/to/interchange
CI smoke:         python record.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as ch0.1/0.3). The env
# is shared reference code (decision 004): we import it for the obs/action
# semantics and to drive the local recording, never copy it into the chapter.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curriculum.common.device import banner  # noqa: E402
from curriculum.common.envs.pusht.pusht_env import PushTEnv  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=0, help="seeds the env reset and the teleop wobble; two --seed 0 runs match byte-for-byte")
parser.add_argument("--smoke", action="store_true", help="tiny fixed-length run for CI; two --smoke runs must produce byte-identical metrics.json")
parser.add_argument("--out", type=Path, default=Path("outputs/ch0.4-record"), help="run dir: the dataset lands in {out}/dataset, metrics in {out}/metrics.json")
parser.add_argument("--episodes", type=int, default=2, help="local mode only: how many episodes to record (ignored when --from-interchange)")
parser.add_argument("--from-interchange", dest="from_interchange", type=Path, default=None, help="INGEST mode: a z2r-teleop-1 bundle dir from the browser instead of recording locally")
parser.add_argument("--repo-id", dest="repo_id", default="zero2robot/pusht_teleop", help="repo id written into meta/info.json (local only; the bundle carries its own)")
parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=False, help="also record 96x96 top-down frames as an mp4 feature")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)  # recording your episode to rerun is the default; opt OUT, not in
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip the .rrd (CI smoke)")
args = parser.parse_args()

banner("ch0.4-record", device="cpu")  # pure-numpy/mujoco-CPU: honest cpu tier, never the host's mps/cuda. startup contract: tier + measured wall-clock to stdout, not metrics.json
rng = np.random.default_rng(args.seed)  # PCG64 — the only source of randomness in this file
# --- endregion ---

# --- region: features ---
# A LeRobot dataset is columns of typed, fixed-shape arrays. `features` is that
# schema, and it is the ONE thing chapter 1.1's BC assumes about your data: an
# `observation.state` of 10 floats and an `action` of 2. These names/shapes are
# copied verbatim from gen_demos.build_features + pusht_env.py so a teleop
# episode and a scripted-expert episode are the same shape — the whole point of
# recording your own data is that it drops straight into the same pipeline.
TASK = "Push the T-shaped block to the target pose."  # every frame carries this string
IMG_HW = 96

STATE_NAMES = [  # obs[i] meaning; sin/cos-encoded yaw avoids the wrap discontinuity
    "pusher_x", "pusher_y", "tee_x", "tee_y", "sin_tee_yaw", "cos_tee_yaw",
    "target_x", "target_y", "sin_target_yaw", "cos_target_yaw",
]


def build_features(video: bool) -> dict:
    """The dataset schema. `observation.state` f32[10] + `action` f32[2],
    plus an optional 96x96 video. lerobot adds timestamp/frame_index/... itself."""
    features = {
        "observation.state": {"dtype": "float32", "shape": (PushTEnv.OBS_DIM,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (PushTEnv.ACT_DIM,), "names": ["pusher_vx", "pusher_vy"]},
    }
    if video:
        features["observation.image"] = {"dtype": "video", "shape": (IMG_HW, IMG_HW, 3), "names": ["height", "width", "channel"]}
    return features
# --- endregion ---

# --- region: teleop ---
# The local recorder. Real teleop is a human dragging the pusher with a mouse
# (that's the browser demo); a mouse drag produces exactly an `action` — a
# target velocity [vx, vy]. We can't replay a live mouse deterministically, so
# here a tiny scripted controller stands in for your hand: get behind the block,
# then shove it toward the target. It is crude on purpose — it drives the block's
# POSITION home but never squares up its ORIENTATION, so it essentially never
# trips the env's success latch and every episode runs to the time limit. That
# imperfection is the lesson: teleop data is messy. What matters is that every
# step emits a real (obs, action) pair, and a list of those pairs is an episode.
def scripted_drive(obs: np.ndarray) -> np.ndarray:
    pusher, tee, target = obs[0:2], obs[2:4], obs[6:8]
    to_target = target - tee
    reach = float(np.linalg.norm(to_target))
    goal_dir = to_target / (reach + 1e-9) if reach > 1e-3 else np.zeros(2)
    contact = tee - goal_dir * 0.05          # the spot just behind the block to push from
    to_contact = contact - pusher
    gap = float(np.linalg.norm(to_contact))
    if gap > 0.02:                           # not in position yet: close the gap to the contact point
        return (to_contact / (gap + 1e-9)) * 0.6
    return goal_dir * 0.5                     # in contact: push the block toward the target


def record_local(episodes: int, seed: int, smoke: bool, video: bool, generator, repo_id: str):
    """Drive the env `episodes` times, collecting one frame per control step.
    Mirrors gen_demos' loop: record the pre-step obs, THEN step — so we store the
    obs we acted ON, never the terminal obs the episode ended in. Storing the
    terminal frame is a classic off-by-one that quietly corrupts a dataset."""
    env = PushTEnv()
    max_len = 40 if smoke else PushTEnv.MAX_STEPS  # smoke length is FIXED so CI can diff runs
    recorded = []
    for i in range(episodes):
        obs = env.reset(seed + i)            # seed+i per episode: each episode is a different, reproducible start
        frames = {"observation.state": [], "action": [], "observation.image": []}
        for _ in range(max_len):
            # A little seeded wobble makes the drive feel hand-driven and ties the
            # recorded data to --seed; the env clips actions to [-1, 1] anyway.
            action = np.clip(scripted_drive(obs) + generator.normal(0.0, 0.03, size=2), -1.0, 1.0).astype(np.float32)
            frames["observation.state"].append(obs.astype(np.float32))
            frames["action"].append(action)
            if video:
                frames["observation.image"].append(env.render_frame(IMG_HW, IMG_HW))
            obs, _, done, _ = env.step(action)
            if done and not smoke:           # stop when the env signals done — success latch OR the MAX_STEPS limit (this crude driver always hits the limit); smoke ignores done and stays fixed-length so CI can diff runs
                break
        recorded.append(frames)
    config = {"repo_id": repo_id, "fps": PushTEnv.CONTROL_HZ, "robot_type": "pusher_2d",
              "features": build_features(video), "use_videos": video, "task": TASK}
    return config, recorded
# --- endregion ---

# --- region: ingest ---
# The other input: a browser recording. The playground writes a `z2r-teleop-1`
# interchange (decision 008) — a format-stable JSON manifest with inline episode
# arrays plus one PNG per frame — NOT a v3 dataset. We convert it here with the
# same writer as the local path. The converter is env-agnostic on purpose: it
# reads the feature spec the browser DECLARES rather than hardcoding obs[10],
# so a future ALOHA-sim scene works without touching this code.
def load_interchange(path: Path):
    manifest = json.loads((path / "interchange.json").read_text())
    version = manifest["interchange_version"]
    assert version == "z2r-teleop-1", f"unknown interchange {version!r} (expected z2r-teleop-1)"
    features = {name: {"dtype": f["dtype"], "shape": tuple(f["shape"]), "names": f["names"]}
                for name, f in manifest["features"].items()}
    use_videos = any(f["dtype"] == "video" for f in features.values())
    episodes = []
    for ep in manifest["episodes"]:
        frames = {"observation.state": [np.asarray(s, np.float32) for s in ep["observation.state"]],
                  "action": [np.asarray(a, np.float32) for a in ep["action"]],
                  "observation.image": []}
        if use_videos:
            from PIL import Image  # only paid for when the bundle carries frames
            h, w, _ = features["observation.image"]["shape"]
            for rel in ep["observation.image"]:
                pixels = np.asarray(Image.open(path / rel).convert("RGB"))
                # A real browser frame is already HxWx3 and passes through untouched;
                # only a 1x1 stand-in (the reference writer's test PNG) is broadcast up.
                frame = pixels if pixels.shape == (h, w, 3) else np.broadcast_to(pixels[0, 0], (h, w, 3))
                frames["observation.image"].append(np.ascontiguousarray(frame, dtype=np.uint8))
        episodes.append(frames)
    config = {"repo_id": manifest["repo_id"], "fps": manifest["fps"], "robot_type": manifest["robot_type"],
              "features": features, "use_videos": use_videos, "task": manifest["task"]}
    return config, episodes
# --- endregion ---

# --- region: write ---
# The canonical write path — the ONE output both inputs share, and the reason
# format parity is free. This is byte-for-byte the LeRobotDataset.create ->
# add_frame -> save_episode -> finalize sequence from gen_demos.py. Because the
# pinned `lerobot` library does the writing, the on-disk v3 layout (parquet
# schema, meta/info.json, stats quantiles, CODEBASE_VERSION) is guaranteed to
# match the training datasets — we never re-derive it.
def write_dataset(config: dict, episodes: list, root: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # lazy: pulls in torch

    dataset = LeRobotDataset.create(
        repo_id=config["repo_id"], fps=config["fps"], features=config["features"],
        root=root, robot_type=config["robot_type"], use_videos=config["use_videos"])
    for episode in episodes:
        for i in range(len(episode["observation.state"])):
            frame = {"observation.state": episode["observation.state"][i],
                     "action": episode["action"][i], "task": config["task"]}
            if config["use_videos"]:
                frame["observation.image"] = episode["observation.image"][i]
            dataset.add_frame(frame)  # NO timestamp key — lerobot derives it from frame_index / fps
        dataset.save_episode()        # one episode = one contiguous run of frames, its own boundary
    dataset.finalize()                # compute stats, write meta/*.parquet — the dataset is now loadable
    return dataset
# --- endregion ---

# --- region: run ---
args.out.mkdir(parents=True, exist_ok=True)
if args.from_interchange is not None:
    source = "interchange"
    config, episodes = load_interchange(args.from_interchange)
else:
    source = "local-teleop"
    config, episodes = record_local(args.episodes, args.seed, args.smoke, args.video, rng, args.repo_id)

dataset_root = args.out / "dataset"  # the v3 dataset lives here; metrics.json sits beside it, out of the dataset
write_dataset(config, episodes, dataset_root)

# metrics.json: the determinism-checked artifact. The dataset itself embeds a
# uuid and absolute paths, so it is NOT byte-stable; these STRUCTURAL facts are.
all_states = [state for ep in episodes for state in ep["observation.state"]]
all_actions = [action for ep in episodes for action in ep["action"]]
metrics = {
    "source": source,
    "seed": args.seed,
    "n_episodes": len(episodes),
    "n_frames": len(all_states),
    "obs_dim": PushTEnv.OBS_DIM,
    "act_dim": PushTEnv.ACT_DIM,
    "fps": config["fps"],
    "robot_type": config["robot_type"],
    "task": config["task"],
    "feature_keys": sorted(config["features"]),
    "first_obs": [round(float(v), 6) for v in all_states[0]],
    "last_obs": [round(float(v), 6) for v in all_states[-1]],
    "first_action": [round(float(v), 6) for v in all_actions[0]],
    "last_action": [round(float(v), 6) for v in all_actions[-1]],
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def log_rerun(episodes: list, config: dict, path: Path):
    """Replay the recorded obs to rerun on canonical entity paths, so you can
    scrub your own episode. Works the same whether the obs came from local
    teleop or a browser bundle — an episode is just obs arrays either way."""
    import rerun as rr  # lazy: --no-rerun (CI) never imports it

    rr.init("zero2robot/ch0.4-record", spawn=False)
    rr.save(str(path))
    tee_centers = [(0.0, 0.0, 0.0), (0.0, -0.06, 0.0)]  # the T is two boxes (see pusht.xml)
    tee_half = [(0.06, 0.015, 0.015), (0.015, 0.045, 0.015)]
    gx, gy, _ = PushTEnv.TARGET_POSE
    rr.log("world/objects/target", rr.Boxes3D(centers=[(gx + c[0], gy + c[1], 0.0) for c in tee_centers],
                                              half_sizes=tee_half, colors=(90, 205, 100, 120)), static=True)
    rr.log("world/objects/tee", rr.Boxes3D(centers=tee_centers, half_sizes=tee_half, colors=(115, 128, 242)), static=True)
    rr.log("world/robot/pusher", rr.Cylinders3D(lengths=[0.04], radii=[0.015], colors=(230, 102, 90)), static=True)
    frame_index = 0
    for episode in episodes:
        for obs, action in zip(episode["observation.state"], episode["action"]):
            rr.set_time("sim_time", duration=frame_index / config["fps"])
            frame_index += 1
            tee_yaw = math.atan2(float(obs[4]), float(obs[5]))  # decode sin/cos back to an angle
            rr.log("world/objects/tee", rr.Transform3D(translation=(float(obs[2]), float(obs[3]), 0.0152),
                                                       rotation=rr.RotationAxisAngle(axis=(0, 0, 1), radians=tee_yaw)))
            rr.log("world/robot/pusher", rr.Transform3D(translation=(float(obs[0]), float(obs[1]), 0.02)))
            rr.log("policy/action", rr.Scalars(np.asarray(action, dtype=np.float64)))


print(f"source: {source} -> wrote {metrics['n_episodes']} episodes ({metrics['n_frames']} frames)")
print(f"dataset: {dataset_root}  (load it in ch0.5, train on it in ch1.1)")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    rrd = args.out / "record.rrd"
    log_rerun(episodes, config, rrd)
    print(f"recording: {rrd} — open it with: rerun {rrd}")
# --- endregion ---
