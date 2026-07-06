"""Generate PushT demonstration episodes with the scripted expert.

Writes a fully local LeRobot-format dataset (v3.0, parquet [+ mp4 with
--video]) via the pinned `lerobot` package -- no hub access, no network.
This is the CI/offline demo source (decision 004); the HF reference dataset
`lerobot/pusht` stays the learner-facing default in chapter prose.

Episode i uses env seed (--seed + i) for both the reset and the expert's
exploration noise, so any dataset is bit-for-bit reproducible from its CLI
arguments alone.

Usage:
    python gen_demos.py --episodes 50 --seed 0 --out /path/to/dataset
    python gen_demos.py --episodes 10 --seed 0 --out ds --noise 0.08 --video
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .pusht_env import PushTEnv
    from .scripted_expert import ScriptedExpert
except ImportError:  # running as a loose script
    from pusht_env import PushTEnv
    from scripted_expert import ScriptedExpert

TASK = "Push the T-shaped block to the target pose."
IMG_HW = 96

STATE_NAMES = [
    "pusher_x", "pusher_y", "tee_x", "tee_y", "sin_tee_yaw", "cos_tee_yaw",
    "target_x", "target_y", "sin_target_yaw", "cos_target_yaw",
]


def build_features(video: bool) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (PushTEnv.OBS_DIM,),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (PushTEnv.ACT_DIM,),
            "names": ["pusher_vx", "pusher_vy"],
        },
    }
    if video:
        features["observation.image"] = {
            "dtype": "video",
            "shape": (IMG_HW, IMG_HW, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True,
                        help="dataset root directory (must not already exist)")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="expert exploration-noise std (m/s), seeded per episode")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=False,
                        help="also record 96x96 top-down frames as mp4")
    parser.add_argument("--repo-id", default="zero2robot/pusht_scripted",
                        help="repo id written into meta/info.json (local only)")
    args = parser.parse_args(argv)

    # lazy import: heavy (torch), and only needed when actually generating
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=PushTEnv.CONTROL_HZ,
        features=build_features(args.video),
        root=args.out,
        robot_type="pusher_2d",
        use_videos=args.video,
    )

    env = PushTEnv()
    successes, total_steps = 0, 0
    for i in range(args.episodes):
        seed = args.seed + i
        obs = env.reset(seed)
        expert = ScriptedExpert(noise=args.noise, seed=seed)
        done = False
        while not done:
            action = expert.action(env)
            frame = {"observation.state": obs, "action": action, "task": TASK}
            if args.video:
                frame["observation.image"] = env.render_frame(IMG_HW, IMG_HW)
            dataset.add_frame(frame)
            obs, _, done, info = env.step(action)
            total_steps += 1
        dataset.save_episode()
        successes += bool(info["success"])
        print(f"episode {i:3d} (seed {seed}): "
              f"{'success' if info['success'] else 'FAIL'} in {env._step_count} steps")

    dataset.finalize()
    print(f"\nwrote {args.episodes} episodes ({total_steps} frames) to {args.out}")
    print(f"expert success: {successes}/{args.episodes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
