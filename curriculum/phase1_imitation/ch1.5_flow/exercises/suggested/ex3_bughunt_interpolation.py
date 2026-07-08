"""SUGGESTED exercise candidate (humans promote) — bug-hunt, ch1.5.

Flow matching is defined against ONE geometric object: the straight line from a
noise point to a data point, evaluated at a random time t in [0, 1],

    x_t = (1 - t) * noise + t * data        (noise ~ N(0, I))

The endpoints are the whole contract: at t = 0 the path must sit exactly on the
NOISE the sampler starts from, and at t = 1 it must sit exactly on the DATA. The
velocity the net is trained to predict is this line's derivative, the constant
data - noise. Get the interpolation endpoints backwards and you train the net to
drive noise toward noise (and "data" toward data at the wrong end): the loss still
falls, but the sampler integrates the wrong line and never reaches the data.

THE BUG. `flow_interpolate` below has its two coefficients SWAPPED. It still runs,
still returns the right shape, and training still produces a falling loss — the
model just learns the wrong path and samples badly.

Before you unswap them, write one sentence: with the endpoints backwards the
loss still falls — so what did the net learn to predict perfectly, and why does
the sampler then walk the wrong way?

Find it and fix it so the check
passes (the endpoints must be: t=0 -> noise, t=1 -> data).

    pytest curriculum/phase1_imitation/ch1.5_flow/exercises/suggested/checks.py -k ex3
"""

import numpy as np

METADATA = {"type": "bug-hunt", "chapter": "ch1.5-flow"}


def flow_interpolate(data: np.ndarray, noise: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Point on the straight noise->data line at time t. Shapes: data, noise are
    (B, D); t is (B,). Return (B, D).

    x_t = (1 - t) * noise + t * data

    BUG: the two time coefficients are swapped, so the path runs data->noise instead
    of noise->data. At t=0 it must equal noise; at t=1 it must equal data.
    """
    return t[:, None] * noise + (1.0 - t)[:, None] * data
