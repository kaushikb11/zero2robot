"""SUGGESTED exercise candidate (humans promote) — code-completion, ch5.7.

The one function the whole chapter turns on: symmetric INT8 quantization of a weight matrix.
INT8 is not a rounding; it is a SCALE. Map a real range [-r, r] onto the integer grid
[-127, 127] with a scale s = r / 127, store q = round(W / s) clamped to the grid, and recover
W_hat = q * s. Your job: implement `quantize_weight` for both granularities.

  per-tensor  : r = max|W| over the WHOLE matrix -> ONE scalar scale.
  per-channel : r = max|W[o]| per OUTPUT ROW -> one scale each, shape (out, 1), so it
                broadcasts back over the input axis.

The trap this refutes: "just round the weights to int8." Round with NO scale and a policy
whose weights live in (-0.5, 0.5) collapses — every weight rounds to 0. The checks prove
your per-channel round-trip error is strictly smaller than per-tensor, AND that naive
no-scale rounding annihilates the tensor.

Implement `quantize_weight` below (pure numpy), then run the checks:
    pytest curriculum/phase5_practitioner/ch5.7_quantize/exercises/suggested/checks.py -k ex2
Estimated learner time: 20 minutes.
"""

import numpy as np

QMAX = 127  # symmetric signed int8 grid is [-127, 127]


def quantize_weight(W: np.ndarray, per_channel: bool):
    """(out, in) weight -> (q int8 same shape, scale). Dequantize is q * scale.

    Remove the NotImplementedError and write it. HINT:
      r = np.abs(W).max(axis=1, keepdims=True) if per_channel else np.abs(W).max()
      scale = np.maximum(np.asarray(r, np.float32) / QMAX, 1e-8)   # never divide by zero
      q = np.clip(np.round(W / scale), -QMAX, QMAX).astype(np.int8)
    Return (q, scale). For per-channel, scale has shape (out, 1) so q * scale broadcasts.
    """
    raise NotImplementedError("write quantize_weight: a per-tensor OR per-row max -> scale = r/127 -> round & clamp")
