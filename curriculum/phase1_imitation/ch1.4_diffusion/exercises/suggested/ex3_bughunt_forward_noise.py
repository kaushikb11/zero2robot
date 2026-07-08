"""SUGGESTED exercise candidate (humans promote) — bug-hunt, ch1.4.

The forward process is the one equation everything else is defined against: to
train, we jump a clean sample x0 to noise level t in a single shot,

    x_t = sqrt(acp_t) * x0 + sqrt(1 - acp_t) * noise        (noise ~ N(0, I))

The two coefficients are not free knobs. They are chosen so that x_t keeps unit
variance at every level (acp_t + (1 - acp_t) = 1): the signal fades as sqrt(acp_t)
and the noise grows as sqrt(1 - acp_t), and the sum stays calibrated to the
N(0, I) that sampling starts from. Get the noise coefficient wrong and you train
the denoiser against noise levels that don't match the ones the sampler will use.

THE BUG. `forward_noise` below has ONE wrong coefficient. It still runs, still
returns the right shape, and training still produces a falling loss — the model
just learns the wrong noising and samples badly.

Before you fix the coefficient, write one sentence: why does the loss keep
falling while the samples get worse — what is the denoiser cheerfully learning
to do instead?

Find it and fix it so the check
passes (do NOT touch the signal term; only one coefficient is wrong).

    pytest curriculum/phase1_imitation/ch1.4_diffusion/exercises/suggested/checks.py -k ex3
"""

import numpy as np

METADATA = {"type": "bug-hunt", "chapter": "ch1.4-diffusion"}


def forward_noise(x0: np.ndarray, acp_t: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Noise a clean batch x0 (B, D) to the level whose alpha-bar is acp_t (B,).

    x_t = sqrt(acp_t) * x0 + <noise coefficient> * noise

    BUG: the noise coefficient is wrong. The variance-preserving forward process
    needs the signal and noise powers to sum to 1 at every level.
    """
    signal = np.sqrt(acp_t)[:, None] * x0
    noise_term = (1.0 - acp_t)[:, None] * noise
    return signal + noise_term
