"""SUGGESTED exercise candidate (humans promote) — bug-hunt, ch1.8.

The VLA trains on a MULTI-TASK pile where the embodiments have different action
dimensionalities: a PushT frame has 2 real action dims, an ALOHA frame has 6 (the
rest are zero-padded, marked by an action_mask). The flow-matching loss must weigh a
PushT frame and an ALOHA frame EQUALLY — otherwise the 6-dim task contributes 3x the
per-frame gradient and DOMINATES training, starving the other task (measured: with the
buggy loss, PushT collapses to 0.0 success while ALOHA learns; balancing flips it).

The contract: reduce the per-dim squared error to ONE number per example (the MEAN
over that example's VALID dims), THEN average over examples. A 2-dim example and a
6-dim example each contribute exactly one equally-weighted term.

Before you read why, write one sentence: with `.sum() / mask.sum()`, does a
6-dim ALOHA frame push harder or softer on the gradient than a 2-dim PushT
frame — and which task does that starve to 0.0?

THE BUG. `masked_flow_loss` below SUMS the masked error over the whole batch and
divides by the total number of valid dims — so an example with more valid dims pulls
harder. Find it and fix it to the per-example average (the check pins a fixture where
the two are numerically different).

    pytest curriculum/phase1_imitation/ch1.8_vla/exercises/suggested/checks.py -k ex3
"""

import torch

METADATA = {"type": "bug-hunt", "chapter": "ch1.8-vla"}


def masked_flow_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Velocity MSE over VALID action dims, weighing every EXAMPLE equally.

    pred/target/mask are all (B, ACT_DIM); mask is 1.0 where the action dim is real for
    that example's embodiment, 0.0 where it is zero-padding.

    BUG: this sums the masked error across the whole batch and divides by mask.sum(),
    which weights each example by its number of valid dims — the higher-DOF embodiment
    dominates. It must instead average each example's error over ITS valid dims first,
    then take the mean over examples.
    """
    se = (pred - target) ** 2
    return (se * mask).sum() / mask.sum()
