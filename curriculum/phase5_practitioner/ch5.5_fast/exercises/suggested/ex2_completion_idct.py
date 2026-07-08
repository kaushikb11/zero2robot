"""SUGGESTED exercise candidate (humans promote) — code-completion, ch5.5.

Objective tested: the inverse half of the transform — the thing that turns coefficients back
into a trajectory. fast.py builds the forward DCT as an ORTHONORMAL matrix D (dct_matrix): a
chunk's coefficients are `D @ chunk`. Your job is the inverse, `idct`, which the decoder needs
to go coefficients -> trajectory.

The whole point of using the orthonormal DCT is that the inverse is trivial: because D has
orthonormal rows, D @ D.T = I, so the inverse transform is JUST the transpose — no scaling to
undo, no schedule (contrast ch1.4's DDPM reverse loop). Get this one line right and the codec
round-trips (BPE-decode -> dequantize -> idct) to machine precision; get the scaling wrong and
every reconstruction is silently stretched.

Implement `idct` below (pure numpy), then run the checks:
    pytest curriculum/phase5_practitioner/ch5.5_fast/exercises/suggested/checks.py -k ex2
Estimated learner time: 10 minutes.
"""

import numpy as np


def dct_matrix(n: int) -> np.ndarray:
    """The forward transform, given (same as fast.py): an (n, n) orthonormal DCT-II matrix.
    coeffs = dct_matrix(n) @ signal."""
    k = np.arange(n)[:, None]
    t = np.arange(n)[None, :]
    d = np.cos(np.pi * (2 * t + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    d[0] *= 1.0 / np.sqrt(2.0)
    return d


def idct(coeffs: np.ndarray, dct_mat: np.ndarray) -> np.ndarray:
    """Invert the DCT: (H, act_dim) coefficients -> (H, act_dim) trajectory, given the SAME
    orthonormal matrix `dct_mat` the forward transform used. Remove the NotImplementedError
    and write it. HINT: the matrix is orthonormal, so the inverse is one matmul with its
    transpose — no division, no scaling."""
    raise NotImplementedError("write idct: the inverse of an orthonormal transform is its transpose")
