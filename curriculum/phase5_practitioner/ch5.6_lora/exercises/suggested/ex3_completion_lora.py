"""SUGGESTED exercise candidate (humans promote) — code-completion + the learner-generated
failure, ch5.6.

Objective tested: the LoRA update rule AND why B is zero-initialized. LoRA wraps a FROZEN
linear layer y = W x + b with a thin trainable low-rank bypass:

    lora_forward(x) = W x + b  +  (alpha / r) * B (A x)      A: (r, in)   B: (out, r)

Only A and B train; W and b are frozen. The single most important initialization detail:
B is ZEROED, so B(A x) = 0 at step 0 and the wrapped layer is BITWISE the frozen one — you
begin fine-tuning from the exact model you pretrained. Get that wrong (init B like any other
Linear) and the adapter perturbs the frozen model before a single gradient step.

Your job:
  (1) implement `lora_delta(x, A, B, scaling)` = scaling * B (A x)   [the low-rank bypass], and
  (2) implement `init_B(out_features, r)` returning the CORRECT initial B (the one that makes
      the adapter a no-op at step 0).

Then run the checks:
    pytest curriculum/phase5_practitioner/ch5.6_lora/exercises/suggested/checks.py -k ex3

THE FAILURE YOU GENERATE: the checks also build B with `init_B_buggy` (kaiming, the "I forgot
to zero it" mistake) and confirm the adapter is NO LONGER a no-op — the frozen output moves at
step 0. That is exactly what `lora.py --break rand_init_B` does. Estimated learner time: 20 min.
"""

import math

import torch
import torch.nn.functional as F  # noqa: F401  (the learner completes the loss using F)


def lora_delta(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, scaling: float) -> torch.Tensor:
    """The low-rank bypass added to the frozen layer's output: scaling * B (A x).

    x: (batch, in)   A: (r, in)   B: (out, r)   ->   returns (batch, out).
    HINT: project x down with A (F.linear(x, A) -> (batch, r)), up with B, then scale.
    Remove the NotImplementedError and write it.
    """
    raise NotImplementedError("write the low-rank bypass: scaling * B (A x)")


def init_B(out_features: int, r: int) -> torch.Tensor:
    """Return the CORRECT initial B so the adapter is a NO-OP at step 0 (adapted == frozen).
    Remove the NotImplementedError and write it (one line)."""
    raise NotImplementedError("what value of B makes (alpha/r) * B (A x) exactly zero for all x?")


def init_B_buggy(out_features: int, r: int) -> torch.Tensor:
    """The natural mistake: init B like any other Linear weight (kaiming), NOT zeroed. The
    checks use this to show the adapter is no longer a no-op — the learner-generated failure."""
    B = torch.empty(out_features, r)
    torch.nn.init.kaiming_uniform_(B, a=math.sqrt(5))
    return B
