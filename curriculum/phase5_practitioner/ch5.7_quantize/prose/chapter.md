# 5.7: Quantize a Policy by Hand — INT8 Is a Scale, Not a Rounding

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## The wrong mental model

You have a trained policy. Someone says "ship it in int8, it'll be four times smaller and
faster." The obvious thing to try is a cast: take each weight, round it to the nearest whole
number, store it as a signed byte. Try it on the ch1.1 behavior-cloning MLP, whose weights all
live in `(-0.5, 0.5)`, and the policy **evaporates** — `round(0.31) = 0`, `round(-0.12) = 0`,
about 95% of the weights round to zero, and the robot goes limp.

That failure is the whole lesson in miniature. INT8 quantization is **not** a rounding. It is a
**scale**. The int8 grid is a fixed ruler with 255 evenly spaced ticks from `-127` to `+127`;
the same ruler for every tensor in every model. Quantizing means choosing a real number `s` —
the spacing between ticks in your tensor's units — and storing `q = round(w / s)`. You recover
`w ≈ s · q`. Rounding threw away the scale and measured a policy in the wrong units. Open
`quantize.py`. Six regions: **setup**, **data**, **train**, **quantize**, **deploy**, **report**.

Everything in this chapter runs on the **CPU in numpy**, and that is a feature: int8 and int32
integer arithmetic is **exact and bitwise-reproducible**. Same seed, same bytes — no cuDNN
nondeterminism to caveat. This is the one chapter where "deterministic" needs no asterisk.

## A scale, per tensor and per channel

```
[include-by-region: quantize.py#quantize]
```

`quantize_weight` is the whole idea in four lines. Take the range `r = max|W|`, set the scale
`s = r / 127`, and round `W / s` onto the grid. Dequantize is a single multiply, `s · q`. The
**only** decision is what `r` ranges over:

- **per-tensor**: one `r` for the whole matrix. Simple — and wasteful. One fat-tailed output
  channel with a big weight sets the tick spacing for *every* channel, so a row whose weights
  are all tiny gets crushed onto three or four ticks and loses almost all its resolution.
- **per-channel**: one `r` per **output row**. Every channel gets a tick spacing matched to its
  own range. The fat channel no longer taxes the small ones.

Measure the round-trip error `mean|W − s·q|` both ways and per-channel is **guaranteed** smaller
— a per-row scale is a strict refinement of a single scale. On the trained policy it is ~1.7–2.0×
smaller, every seed. That refinement is the recovery you will see propagate all the way to the
policy's actions. And the one-liner right below it, `NAIVE_ROUND_ZERO_FRAC`, is the misconception
from the opening, measured: round with no scale and ~95% of the weights collapse to zero.

## Activations don't hold still

Weights you can quantize offline — they are fixed numbers, you know their range. **Activations
you cannot**: their range depends on the input, and you don't know it until data flows through
the net. So you *calibrate*.

```
[include-by-region: quantize.py#quantize]
```

`calibrate` runs a **calibration set** — a handful of held-out states — through the fp32 network
and records the range of each layer's *input* activations. From that range you get an activation
scale, exactly like a weight scale. Two rules for reading the range:

- **min-max**: `r = max|a|`. Honest, until one freak state produces a giant activation and drags
  the scale up so far that ordinary activations lose their resolution.
- **percentile**: `r =` the 99.9th percentile of `|a|`. Ignores the top 0.1%, so a single outlier
  can't blow up the scale. This is the robust default (it is exactly what onnxruntime's percentile
  calibrator does).

With activation scales in hand, the forward pass becomes **all integers**: quantize the input to
int8, do an `int8 @ int8 → int32` matmul (int32 because a sum of products of bytes overflows a
byte), then dequantize the accumulator back to real units with `s_x · s_w` and add the fp32 bias.

```
[include-by-region: quantize.py#deploy]
```

## The deployment triangle

There is no free lunch in deployment; there is a **triangle** — size, accuracy, latency — and you
have to look at all three corners at once. `quantize.py` measures them for FP32, per-tensor INT8,
and per-channel INT8 (the triangle uses **weight-only** int8: int8 weights, dequantized on the
fly, fp32 matmul — the dominant real-world mode, what llama.cpp's `Q8` and HuggingFace int8 do).

**Size.** Guaranteed. An int8 weight is one byte where fp32 was four. The measured ratio is
~3.6×, a hair short of 4× only because the biases and the per-channel scales stay fp32.

**Accuracy.** This is the headline, and it reproduces on every seed: **per-tensor INT8 spikes the
action error; per-channel INT8 recovers most of it** — 2–7× lower action-MSE-vs-fp32, at the same
~4× size saving. The per-row scale you built in region three is the entire difference. We grade
accuracy with ch1.6 rigor: each config is also rolled out for a success rate with a **Wilson
interval**, and at this eval budget those intervals *overlap fp32* — so we make **no** task-success
claim. Quantization here is task-indistinguishable from fp32 at N=24. The claim we stand behind is
the deterministic action-error direction and the size win, not a success-rate drop that never
cleared the band.

**Latency.** Here is the honesty the marketing skips. The size win is guaranteed; the **latency
win is not**. On a laptop CPU with no fused int8 kernel, quantizing and dequantizing around a
plain matmul is *overhead*, and naive int8 comes out **~6× slower** than fp32. That is not a bug
in the code — it is the reason production deployment reaches for TensorRT or a fused QLinear
kernel, where the integer matmul is actually faster than the float one. We measure it and print
whichever wins; on this CPU, int8 loses.

## Break it: calibrate on the wrong distribution

```
python curriculum/phase5_practitioner/ch5.7_quantize/quantize.py --seed 0 --break bad_calib
```

The fragile corner of static quantization is the calibration set. `--break bad_calib` calibrates
the activation scales on a **narrow slice** — only the frames where the block already sits near
the target, where the policy barely moves and activations are small. The scales come out too
tight. At real deployment the activations run right off the end of the int8 grid and **saturate**
at `±127`, and the full-integer action error **explodes** — 4.5–24× worse than a representative
calibration, every seed.

Two things to notice. First, the **weight-only triangle is untouched** — the break only attacks
the activation path, so a bug in your calibration data hides completely from the weight-quant
numbers. Second, switching per-tensor → per-channel **weights does not rescue it**: when
activations clamp, weight granularity is irrelevant. These are two independent failure modes.
The fix for saturation is a calibration set that matches deployment (plus a percentile clip for
the single-outlier flavor) — not a finer weight scale.

## What you built

A complete post-training quantization pipeline, from scratch, in numpy you can read: a symmetric
int8 scale, per-tensor and per-channel weight quantization, static activation calibration with a
percentile clip, a full-integer forward pass, and an honest three-corner measurement of what it
costs. The **read-the-real-thing** segment opens onnxruntime's `quantization/` tools — `calibrate.py`,
`onnx_quantizer.py`, `quantize.py` — and you will recognize every piece, because you just wrote
the toy version of each. The **scale lab** is the corner this CPU can't show you: the same policy
under a fused int8 kernel, where the latency finally turns in int8's favor.
