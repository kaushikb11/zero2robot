"""SUGGESTED exercise candidate (humans promote) — bug-hunt, ch1.3.

Objective tested: the padding mask that makes action chunking honest near
episode ends. For frame t, the chunk target is the expert actions [t : t+K].
When fewer than K actions remain in the episode, the tail is PADDED (we repeat
the last action) and a 0/1 mask marks those padded steps so they carry NO
gradient — the policy must never be trained to "predict" invented actions.

THE BUG. `build_chunk_targets` below reshapes per-frame actions into per-frame
chunks exactly like act.py's data region — but it marks the WHOLE chunk valid,
including the padded tail. So near every episode's end the policy is trained to
emit the repeated last action as if it were real data. Nothing crashes; the
loss curve looks fine; the end-of-episode behavior quietly rots. Find the index
range that is wrong and fix it so only the real steps are masked in.

The target slice and the pad fill are correct — do not touch them. Only the mask
range is wrong.
"""

import numpy as np

METADATA = {"type": "bug-hunt", "chapter": "ch1.3-act", "target": "build_chunk_targets"}


def build_chunk_targets(actions: np.ndarray, episode_ids: np.ndarray, K: int):
    """For each frame, the next K expert actions within ITS episode (padded).

    Returns (chunk_targets[N, K, act_dim], chunk_mask[N, K]); the mask must be
    1.0 on real actions and 0.0 on the padded tail steps near an episode's end.
    """
    act_dim = actions.shape[1]
    chunk_targets = np.zeros((len(actions), K, act_dim), dtype=np.float32)
    chunk_mask = np.zeros((len(actions), K), dtype=np.float32)
    for e in np.unique(episode_ids):
        idx = np.nonzero(episode_ids == e)[0]
        ep_actions = actions[idx]
        for j, frame in enumerate(idx):
            valid = min(K, len(idx) - j)
            chunk_targets[frame, :valid] = ep_actions[j:j + valid]
            chunk_targets[frame, valid:] = ep_actions[-1]  # pad: repeat the last action
            # BUG: marks ALL K steps valid, so the padded tail (which is invented
            # data) still gets a gradient. Only the first `valid` steps are real.
            chunk_mask[frame, :] = 1.0
    return chunk_targets, chunk_mask
