"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch5.7.

The deployment triangle. You will run `quantize.py` and read three numbers off the FP32,
per-tensor INT8, and per-channel INT8 rows:
  - action error vs fp32 (MSE) for each int8 config,
  - the size ratio fp32 / int8,
  - whether int8 is FASTER than fp32 on this CPU.

PREDICT FIRST (set PREDICTION below), THEN run and check:
    python curriculum/phase5_practitioner/ch5.7_quantize/quantize.py --seed 0
    pytest curriculum/phase5_practitioner/ch5.7_quantize/exercises/suggested/checks.py -k ex1

Which single statement is true?
  A) per-TENSOR INT8 has the lowest action error (a coarser scale is more accurate),
     it is ~4x smaller than fp32, and int8 is ~4x FASTER.
  B) per-CHANNEL INT8 has the lowest action error and is ~4x smaller, and int8 is FASTER —
     fewer bits always means fewer nanoseconds.
  C) per-CHANNEL INT8 has the lowest action error (its per-row scale recovers most of what
     per-tensor lost), it is ~4x smaller than fp32, and int8 is NOT faster than fp32 on this
     CPU (naive int8 pays a dequantize overhead with no fused kernel).
  D) FP32 and both int8 configs have identical action error — quantization is lossless.

Estimated learner time: 15 minutes (a full run trains the MLP, ~20 s CPU).
"""

# Record your prediction as the string "A", "B", "C", or "D". Leave None to SKIP the gate.
PREDICTION = None
