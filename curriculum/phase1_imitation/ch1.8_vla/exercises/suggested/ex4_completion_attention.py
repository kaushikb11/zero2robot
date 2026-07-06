"""SUGGESTED exercise candidate (humans promote) — code-completion, ch1.8.

The tiny VLM backbone is built FROM SCRATCH — no transformers, no einops. Its heart is
one operation: scaled dot-product self-attention, which lets the vision token, the
state token, and each instruction word token exchange information so the CLS token can
read out one fused representation. This exercise asks you to write that operation.

The contract (the same math vla.py's Block runs):

    scores = (q @ kᵀ) / sqrt(head_dim)          # (B, heads, L, L)
    scores[padded keys] = -inf                   # never attend to <pad> tokens
    attn   = softmax(scores, over the last dim)  # each query's weights sum to 1
    out    = attn @ v                            # (B, heads, L, head_dim)

Complete `attention` below so the check passes (it compares against a reference and
verifies that padded key positions get zero weight).

    pytest curriculum/phase1_imitation/ch1.8_vla/exercises/suggested/checks.py -k ex4
"""

import math  # noqa: F401  — you'll need math.sqrt for the scale

import torch

METADATA = {"type": "code-completion", "chapter": "ch1.8-vla"}


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_pad: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product attention.

    q, k, v : (B, heads, L, head_dim)
    key_pad : (B, L) bool — True marks a key position to IGNORE (a <pad> token)
    returns : (B, heads, L, head_dim)

    TODO: implement the four lines from the docstring. Use math.sqrt for the scale and
    masked_fill(key_pad[:, None, None, :], float("-inf")) to drop padded keys before the
    softmax.
    """
    raise NotImplementedError("implement scaled dot-product attention (see the docstring)")
