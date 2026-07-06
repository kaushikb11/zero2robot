"""SUGGESTED exercise candidate (humans promote) — code-completion, ch1.2.

Objective tested: what "demonstrator disagreement" actually IS, by implementing
it. This is the quality signal the chapter leans on hardest and the one the
Break It abuses — so you should be able to write it from the definition.

THE TASK. Fill in `frame_disagreement` below. For each frame it must:
  1. normalize the observations per-dimension to make distances comparable
     (a constant dimension has range 0 — leave it alone, do not divide by zero),
  2. for each frame, find its `k` nearest frames FROM OTHER EPISODES (neighbours
     inside the same episode are temporal near-duplicates and would trivially
     agree — exclude them), and
  3. return, per frame, the spread — POPULATION standard deviation (numpy's
     default; in torch pass correction=0) — of those neighbours' actions,
     averaged over the action dimensions.

A high value means: near this state, demonstrators chose visibly different
actions. The chapter artifact does this with a chunked `torch.cdist`; you may
use torch or plain numpy. checks.py compares your output to the reference on a
small fixed fixture.

Run the check:
    pytest curriculum/phase1_imitation/ch1.2_curate/exercises/suggested/checks.py -k ex4
Estimated learner time: 30 minutes.
"""

from __future__ import annotations

import numpy as np


def frame_disagreement(obs: np.ndarray, actions: np.ndarray,
                       episode_ids: np.ndarray, k: int) -> np.ndarray:
    """(N, obs_dim), (N, act_dim), (N,) -> (N,) per-frame neighbour-action spread.

    See the module docstring for the three steps. Return a float array of
    length N; entry i is the mean over action dims of the std of frame i's k
    nearest out-of-episode neighbours' actions.
    """
    raise NotImplementedError("implement frame_disagreement — that's the exercise")


if __name__ == "__main__":
    # A trivial sanity fixture: two episodes that visit the same state but act
    # oppositely should register high disagreement there.
    obs = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    actions = np.array([[1.0, 0.0], [-1.0, 0.0], [0.5, 0.5], [0.4, 0.6]], dtype=np.float32)
    episode_ids = np.array([0, 1, 0, 1])
    print(frame_disagreement(obs, actions, episode_ids, k=1))
