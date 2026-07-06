"""SUGGESTED exercise candidate (humans promote) — code-completion, ch1.3.

Objective tested: the arithmetic at the heart of temporal ensembling. At eval
step t, several chunks predicted at earlier steps all voted on the action for t.
`ensemble_action` must combine those votes into one action with EXPONENTIAL
weights: `votes` is ordered OLDEST-first (row 0 is the earliest prediction), and
the vote at row i gets weight exp(-m * i) — so the OLDEST vote (row 0) is weighted
MOST and larger m concentrates weight on it — with the weights normalized to sum
to 1. This matches act.py's eval region and the original ACT implementation.

Fill in `ensemble_action` so its output matches the reference on the fixture in
checks.py. This is the exact line act.py runs every control step.
"""

import numpy as np

METADATA = {"type": "code-completion", "chapter": "ch1.3-act", "target": "ensemble_action"}


def ensemble_action(votes: np.ndarray, m: float) -> np.ndarray:
    """Exponentially-weighted average of overlapping chunk votes for one step.

    Args:
        votes: (n, act_dim) predicted actions for the CURRENT step, oldest first.
        m: decay rate; larger m = concentrate weight on the OLDEST vote (row 0); ~0 = uniform.
    Returns:
        (act_dim,) the single action to execute this step.
    """
    raise NotImplementedError(
        "Implement temporal ensembling: weight vote i (age i, oldest = 0) by "
        "exp(-m * i), normalize the weights to sum to 1, return the weighted "
        "sum over the votes. One line is enough; see act.py's eval region."
    )
