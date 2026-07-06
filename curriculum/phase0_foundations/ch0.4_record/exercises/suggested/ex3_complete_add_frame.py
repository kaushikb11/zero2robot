"""SUGGESTED exercise candidate (humans promote) — code-completion, ch0.4.

You get the schema, a couple of synthetic episodes, and the LeRobotDataset
scaffolding (create / save_episode / finalize). Complete the INNER loop: for each
frame in an episode, build the per-frame dict and hand it to `add_frame`. This is
the heart of record.py's writer — the step that turns a list of (obs, action)
pairs into a LeRobot v3 dataset.

Two things the chapter's Write region calls out, and checks.py verifies:
  - every frame dict needs a "task" string (the language label lerobot stores),
  - do NOT put a "timestamp" in the frame — lerobot derives it from frame_index
    and fps; passing one is redundant and can be wrong.

Fill in the body where marked, then run checks.py — it writes your dataset,
loads it back with LeRobotDataset, and checks the schema against a gen_demos
golden.

Run:  python ex3_complete_add_frame.py
Estimated learner time: 20 minutes.
"""

from pathlib import Path

import numpy as np

METADATA = {
    "type": "code-completion",
    "chapter": "ch0.4-record",
}

TASK = "Push the T-shaped block to the target pose."
STATE_NAMES = [
    "pusher_x", "pusher_y", "tee_x", "tee_y", "sin_tee_yaw", "cos_tee_yaw",
    "target_x", "target_y", "sin_target_yaw", "cos_target_yaw",
]


def build_features() -> dict:
    return {
        "observation.state": {"dtype": "float32", "shape": (10,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (2,), "names": ["pusher_vx", "pusher_vy"]},
    }


def synthetic_episodes() -> list:
    """Two tiny episodes (3 frames + 2 frames) of made-up but correctly-shaped
    data. The values are meaningless; only the format is under test here."""
    rng = np.random.default_rng(0)
    episodes = []
    for length in (3, 2):
        episodes.append({
            "observation.state": [rng.standard_normal(10).astype(np.float32) for _ in range(length)],
            "action": [rng.standard_normal(2).astype(np.float32) for _ in range(length)],
        })
    return episodes


def write_dataset(episodes: list, root: Path):
    """Write `episodes` as a LeRobot v3 dataset at `root`. Complete the inner loop."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id="zero2robot/ex3_teleop", fps=10, features=build_features(),
        root=root, robot_type="pusher_2d", use_videos=False)
    for episode in episodes:
        for i in range(len(episode["observation.state"])):
            # TODO(you): build the per-frame dict and add it. It needs three keys:
            #   "observation.state" -> episode["observation.state"][i]
            #   "action"            -> episode["action"][i]
            #   "task"              -> the TASK string
            # then call dataset.add_frame(frame). Do NOT include a "timestamp".
            raise NotImplementedError("complete the add_frame loop (see record.py's Write region)")
        dataset.save_episode()  # one episode boundary, once the frames are in
    dataset.finalize()
    return dataset


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ds = write_dataset(synthetic_episodes(), Path(tmp) / "ds")
        print(f"wrote {ds.num_episodes} episodes / {ds.num_frames} frames")
