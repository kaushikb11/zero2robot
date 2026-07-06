"""SUGGESTED exercise candidate (humans promote) — code-completion, ch3.7.

Objective tested: the cross-embodiment wrangling mechanism at the heart of the
chapter (and of OXE-scale training). Two embodiments with DIFFERENT action
dimensions — PushT's 2-D pusher velocity and ALOHA's 6-D bimanual command — must
land in ONE action tensor a shared policy can emit. You reimplement the zero-pad
+ action_mask that scale_data.py's `wrangle` region builds (the same trick you
first saw in ch1.7): pad every embodiment up to the widest action dim, and carry
a mask marking which dims are REAL so the loss never trains on padding.

This check is DETERMINISTIC: it runs on fixed toy arrays, no training, so it
never flakes. The mechanism is checkable even though the data-scale metric it
enables (ex1, ex3) is noisy.

Run:  python ex2_completion_wrangle.py     (prints the mixed tensor + mask)
Estimated learner time: 20 minutes.
"""

import numpy as np

METADATA = {
    "type": "code-completion",
    "chapter": "ch3.7-scale-data",
    "blanked_region": "wrangle: zero-pad + action_mask",
}


def mix_embodiments(pusht_act: np.ndarray, aloha_act: np.ndarray,
                    pad_dim: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Stack two embodiments' actions into one (N, pad_dim) tensor + a mask.

    Args:
      pusht_act:  (Np, 2) PushT actions — pusher [vx, vy].
      aloha_act:  (Na, 6) ALOHA actions — [r_vx, r_vy, r_grip, l_vx, l_vy, l_grip].
      pad_dim:    the widest action dim across embodiments (6 here).

    Return (mixed, mask), both (Np + Na, pad_dim) float32, PushT rows first:
      - mixed[i, :d] = that row's action (d = its embodiment's real dim), rest 0.
      - mask[i, :d]  = 1.0 on the real dims, 0.0 on the padding.

    The mask is the whole point: a shared policy always emits `pad_dim` numbers,
    but a PushT example only constrains 2 of them — so a masked loss must ignore
    the 4 padded dims, or the model learns to predict PushT's zeros as if they
    were real ALOHA commands. Mirror scale_data.py's `wrangle` region exactly.

    Rough size: 8-10 lines.
    """
    # YOUR CODE HERE (delete the next line once you start)
    raise NotImplementedError("zero-pad each embodiment up to pad_dim and set the action_mask")


if __name__ == "__main__":
    # 3 PushT rows (2-D) + 2 ALOHA rows (6-D). If your padding is right, each
    # PushT row uses 2 of 6 dims (mask density 1/3) and each ALOHA row uses all 6.
    pusht = np.array([[0.5, -0.2], [0.1, 0.9], [-0.4, 0.0]], np.float32)
    aloha = np.array([[1, 0, -1, 0, 0, 1], [0, 1, 1, -1, 0, -1]], np.float32)
    mixed, mask = mix_embodiments(pusht, aloha)
    print("mixed:\n", np.round(mixed, 2))
    print("mask:\n", mask)
    print(f"pusht rows use {mask[:3].mean():.3f} of the padded dims "
          f"(expect 0.333 — 2 of 6); aloha rows use {mask[3:].mean():.3f} (expect 1.0)")
