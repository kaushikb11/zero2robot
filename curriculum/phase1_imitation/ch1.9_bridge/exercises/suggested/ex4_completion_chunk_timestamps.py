"""SUGGESTED exercise candidate (humans promote) — code-completion (fast), ch1.9.

In 1.3 you built the action chunk by hand: for each frame you gathered the next K
expert actions and a pad mask, with an explicit Python loop over every episode.
The official stack does it for you from ONE declaration — `delta_timestamps`. You
tell the LeRobotDataset which future action frames each sample should carry, in
SECONDS relative to the current frame, and it assembles the (K, act_dim) chunk and
the `action_is_pad` mask itself.

For a policy that predicts K actions at `fps` Hz, that list is the K offsets
0, 1/fps, 2/fps, ..., (K-1)/fps — the current frame plus the next K-1, in time.

COMPLETE `action_chunk_timestamps` so bridge.py's `delta_timestamps={"action":
action_chunk_timestamps(fps, K)}` reproduces 1.3's chunk. `checks.py` skips while
it raises, then checks it against fps/K the dataset actually uses. Estimated
learner time: 10 minutes.
"""

METADATA = {"type": "code-completion", "chapter": "ch1.9-bridge", "fast": True}


def action_chunk_timestamps(fps: int, chunk_size: int) -> list[float]:
    """The delta_timestamps list for an action chunk of `chunk_size` frames at
    `fps` Hz: the current frame (offset 0.0) plus the next chunk_size-1 frames,
    each expressed as a time offset in seconds.

    Example: fps=10, chunk_size=4 -> [0.0, 0.1, 0.2, 0.3].
    """
    raise NotImplementedError("return the chunk_size time offsets, in seconds (see the docstring)")


if __name__ == "__main__":
    for fps, k in [(10, 4), (10, 8), (50, 16)]:
        print(f"fps={fps} K={k}: {action_chunk_timestamps(fps, k)}")
