"""SUGGESTED exercise candidate (humans promote) — predict-then-run + learner-generated
failure, ch5.7.

You generate the failure yourself. `--break bad_calib` calibrates the activation scales on a
NARROW slice of states — only the frames where the block already sits near the target, where
the policy barely moves and activations are small. Then it deploys the full-integer policy on
the real distribution.

PREDICT FIRST (set PREDICTION below), THEN generate the failure and check:
    python curriculum/phase5_practitioner/ch5.7_quantize/quantize.py --seed 0
    python curriculum/phase5_practitioner/ch5.7_quantize/quantize.py --seed 0 --break bad_calib
    pytest curriculum/phase5_practitioner/ch5.7_quantize/exercises/suggested/checks.py -k ex3

What happens under `--break bad_calib`, relative to the good (representative) calibration?
  A) Nothing changes — activation scales are derived from the weights, not the data, so the
     calibration set does not matter.
  B) The full-INTEGER path's action error EXPLODES: the narrow calibration underestimates the
     real activation range, so at deployment the activations run off the end of the int8 grid
     and SATURATE at +-127. The weight-only triangle (per-tensor / per-channel) is UNAFFECTED —
     the break only attacks the activation path.
  C) The per-tensor and per-channel weight round-trip errors both blow up — bad calibration
     corrupts the stored int8 weights.
  D) The model gets SMALLER — clamped activations compress better.

Then answer for yourself: does switching per-tensor -> per-channel weights RESCUE the broken
run? (No — when activations saturate, weight granularity is irrelevant. The fix is a
representative calibration set, plus a percentile clip for the single-outlier flavor.)

Estimated learner time: 20 minutes (two full runs, ~20 s CPU each).
"""

# Record your prediction as the string "A", "B", "C", or "D". Leave None to SKIP the gate.
PREDICTION = None
