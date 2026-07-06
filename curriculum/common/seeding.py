"""Seed every RNG a chapter touches, in one call.

Determinism tiering (root CLAUDE.md, invariant 2 — be honest about this):

- BITWISE: env resets on CPU MuJoCo are bitwise-reproducible, and CI enforces
  it (same seed -> identical bytes, twice-run hash match).
- STATISTICAL: training is *statistically* reproducible. Same seed -> same
  qualitative result and metrics within the recorded seeded-run band on the
  same tier. Bitwise training reproducibility is NOT promised on GPU: cuDNN
  and MPS kernels have nondeterministic implementations that
  `use_deterministic_algorithms(warn_only=True)` warns about but does not
  forbid. Chapter prose never claims bitwise GPU reproducibility.

Usage in a chapter artifact:

    rng = set_seed(args.seed)   # `--seed` is mandatory everywhere
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> np.random.Generator:
    """Seed python, numpy, and torch RNGs. Returns a fresh numpy Generator.

    Seeds, in order:
    - `random` (python stdlib)
    - numpy's legacy global RNG (`np.random.seed`) — some third-party code
      still draws from it
    - torch's CPU RNG, plus CUDA (all devices) and MPS when present
    - flips torch into deterministic-algorithms mode (warn_only=True: GPU
      kernels without a deterministic implementation warn instead of crash —
      that is the statistical tier, documented above)
    - when CUDA is present, pins cuDNN to deterministic kernels and disables
      benchmark autotuning (autotuning picks different kernels run-to-run)

    Returns `np.random.Generator(PCG64(seed))` — prefer drawing from this
    generator over the global RNGs; it is explicit and cannot be reseeded
    behind your back by library code.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return np.random.Generator(np.random.PCG64(seed))
