"""SUGGESTED exercise candidate (humans promote) — code-completion, ch1.4.

Sampling is the forward process run backwards, one step at a time. Given the
current noisy x_t and the model's guess x0 of the clean sample, the next (less
noisy) point x_{t-1} is drawn around the DDPM posterior mean of q(x_{t-1} | x_t, x0):

    mean =  beta_t * sqrt(acp_prev) / (1 - acp_t) * x0
          + (1 - acp_prev) * sqrt(alpha_t) / (1 - acp_t) * x_t

where alpha_t = 1 - beta_t, acp_t is alpha-bar at t, and acp_prev is alpha-bar at
t-1 (acp_prev = 1 at the LAST reverse step, t=0). This one line is the entire reverse step in
diffusion.py's sampler — it blends "where the model thinks the clean answer is"
(x0) with "where we already are" (x_t), weighted by how much noise is left.

YOUR JOB: implement `reverse_posterior_mean` from the formula above. Then:

    pytest curriculum/phase1_imitation/ch1.4_diffusion/exercises/suggested/checks.py -k ex4
"""

import numpy as np

METADATA = {"type": "code-completion", "chapter": "ch1.4-diffusion"}


def reverse_posterior_mean(x_t: np.ndarray, x0: np.ndarray, beta_t: float,
                           acp_t: float, acp_prev: float) -> np.ndarray:
    """Posterior mean of q(x_{t-1} | x_t, x0). Shapes: x_t, x0 are (B, D); the
    schedule scalars beta_t, acp_t, acp_prev are floats. Return an (B, D) array.

    Replace the NotImplementedError with the two-term blend from the docstring:
    a coefficient on x0 plus a coefficient on x_t.
    """
    raise NotImplementedError("implement the DDPM posterior mean (see the module docstring)")
