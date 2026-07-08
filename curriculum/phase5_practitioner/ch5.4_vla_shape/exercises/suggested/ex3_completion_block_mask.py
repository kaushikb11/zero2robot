"""SUGGESTED exercise candidate (humans promote) — code-completion, ch5.4 (FAST, self-contained).

Objective tested: the block-attention mask that IS the two-tower architecture. Implement `block_mask`
so the four blocks are exactly right. The sequence is [prefix (0..P-1) | suffix (P..P+H-1)]; the entry
(query i, key j) is True when token i is ALLOWED to attend to token j. The four rules (pi0 / SmolVLA):

    prefix <-> prefix   ALLOWED   the VLM fuses vision + state + language (bidirectional)
    suffix -> prefix    ALLOWED   the action expert READS the whole VLM (the cross-attention) —
                                  unless cut_cross, which severs exactly this block (the --break)
    suffix <-> suffix   ALLOWED   the H chunk steps coordinate (intra-chunk)
    prefix -> suffix    BLOCKED   the prefix NEVER reads the actions, so it is action-independent and
                                  KV-CACHEABLE. Forget this and you make the mask fully bidirectional —
                                  the classic bug the check below catches.

Return a boolean numpy array of shape (P+H, P+H), True = allowed. The checks in checks.py compare it to
a reference AND verify it is NOT the "fully bidirectional" bug (prefix reading the suffix). This gate is
fast + deterministic and runs in `make check`.
"""

import numpy as np


def block_mask(prefix_len: int, horizon: int, cut_cross: bool = False) -> np.ndarray:
    """(P+H, P+H) bool, True = query (row) may attend to key (col). See the module docstring."""
    raise NotImplementedError("implement block_mask: fill the four blocks (prefix->suffix stays BLOCKED)")
    # HINT: start from a full-False (S, S) array, then set the ALLOWED blocks True with slicing:
    #   allowed[:P, :P]  (prefix<->prefix), allowed[P:, P:] (suffix<->suffix),
    #   and allowed[P:, :P] (suffix->prefix) ONLY when not cut_cross. Leave allowed[:P, P:] False.


if __name__ == "__main__":
    P, H = 14, 8
    full = block_mask(P, H, cut_cross=False)
    cut = block_mask(P, H, cut_cross=True)
    print(f"full mask: suffix->prefix allowed? {bool(full[P:, :P].all())}  (want True)")
    print(f"cut  mask: suffix->prefix allowed? {bool(cut[P:, :P].any())}  (want False — that IS --break cut_cross)")
    print(f"prefix->suffix blocked in both? {not full[:P, P:].any() and not cut[:P, P:].any()}  "
          f"(want True — the KV-cacheable asymmetry)")
