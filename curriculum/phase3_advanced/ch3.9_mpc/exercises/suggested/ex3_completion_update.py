"""SUGGESTED exercise candidate (humans promote) — code-completion, ch3.9.

Objective tested: the ~15-line diff that IS this chapter. Sampling-based MPC is one
loop — sample N action sequences, roll each through the model, score them — and the
ONLY thing CEM and MPPI disagree on is how the scores update the sampling mean. Fill
in both updates and feel how little separates them.

You are given N sampled action sequences `samples` (shape (N, H)) and their costs
`costs` (shape (N,), lower is better). Return the new mean plan (shape (H,)).

  * CEM  — keep the ELITE fraction (the lowest-cost sequences) and set the new mean to
           their average. A hard cutoff: elites in, everyone else out.
  * MPPI — keep EVERYONE, but weight sequence i by exp(-(cost_i - min_cost) / temperature)
           (a softmax over negative cost), and set the new mean to that weighted average.
           A soft cutoff: good plans dominate smoothly, nothing is discarded.

Fill in the two functions below (replace the `raise NotImplementedError`). Then run
`pytest checks.py` in this directory — the checks compare your updates against the
reference on fixed (samples, costs), and confirm CEM and MPPI agree when the elite
fraction is tiny and the temperature is low (both then collapse to 'trust the best').

Before you code, PREDICT: as temperature -> 0, which single sample does the MPPI mean
approach? (Answer in a comment. It is the same one CEM keeps when n_elite == 1.)

Estimated learner time: 25 minutes.
"""

import numpy as np

METADATA = {
    "type": "code-completion",
    "chapter": "ch3.9-mpc",
    "blanks": ["cem_update", "mppi_update"],
}

STD_FLOOR = 0.05  # (used by the full planner's std refit; not needed for the mean update here)


def cem_update(samples: np.ndarray, costs: np.ndarray, elite_frac: float) -> np.ndarray:
    """CEM: new mean = average of the elite (lowest-cost) fraction of `samples`.

    samples: (N, H), costs: (N,), returns: (H,).
    Steps: n_elite = max(1, int(elite_frac * N)); take the n_elite lowest-cost rows
    (np.argsort(costs)[:n_elite]); return their mean over axis 0.
    """
    raise NotImplementedError("fill in the CEM elite refit")


def mppi_update(samples: np.ndarray, costs: np.ndarray, temperature: float) -> np.ndarray:
    """MPPI: new mean = softmax(-cost / temperature)-weighted average of ALL `samples`.

    samples: (N, H), costs: (N,), returns: (H,).
    Steps: weights = exp(-(costs - costs.min()) / temperature); normalize to sum 1;
    return the weighted sum of the sequences (weights[:, None] * samples).sum(0).
    Subtracting costs.min() before exp is for numerical stability — it cancels in the
    normalization, so it does not change the result.
    """
    raise NotImplementedError("fill in the MPPI weighted average")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    samples = rng.uniform(-1, 1, size=(64, 25))
    costs = rng.uniform(0, 10, size=64)
    try:
        print("CEM  mean[:3] :", cem_update(samples, costs, 0.1)[:3])
        print("MPPI mean[:3] :", mppi_update(samples, costs, 0.3)[:3])
    except NotImplementedError as e:
        print(f"not filled yet: {e}")
