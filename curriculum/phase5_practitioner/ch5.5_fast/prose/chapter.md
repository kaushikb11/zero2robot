# 5.5: FAST — Turning Torques into Tokens

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## Two ways to say an action

By now you have a policy that can emit an action *chunk* — `H` timesteps of `act_dim` numbers,
the coordinated little burst of motion ch1.3 argued for. The question this chapter asks is
narrow and mechanical: **how do you get that chunk out of a model?**

You have already built one answer. ch1.5's **flow head** treats the chunk as a point in a
continuous space and *integrates* it out of noise in a few Euler steps — the whole chunk at
once, no vocabulary anywhere. This chapter builds the other answer, the one pi0-FAST and
OpenVLA use: **turn the chunk into a short string of discrete tokens** and let a language model
emit them one at a time, so a robot's motion rides the exact same cross-entropy machinery as
text. To do that you need a *codec* — something that maps a real-valued `H × act_dim` chunk to
integers and back. That codec is the whole chapter. No policy is trained, no environment is
stepped; like ch1.7, the subject is the data pipeline itself. Open `fast.py`. Eight regions:
**setup**, **dct**, **quantize**, **bpe**, **data**, **codec**, **measure**, **report**. It is
pure `numpy` — no `torch`, no model.

## The naive tokenizer, and why it hurts

The obvious way to tokenize an action chunk is **per-step-per-dim binning** (this is what RT-2
does): chop each scalar's range into bins and emit one token per number. A chunk is then
`H × act_dim` tokens, always — for `H = 24` and a 6-DoF arm that is 144 tokens for a single
burst of motion, and the language model has to *generate all of them*, autoregressively, every
control cycle. Tokens are the currency of an LM's compute; per-step binning spends them
recklessly. FAST's whole claim is that you can say the same motion in far fewer.

## Step 1 — the DCT: change the basis

```
[include-by-region: fast.py#dct]
```

A smooth trajectory wastes the time domain. Neighboring samples are nearly equal, so most of
the `H` numbers per dimension are redundant. The **discrete cosine transform** re-expresses the
same chunk in a basis of *cosine waves* — a slow one (the average level), a slightly faster one,
and so on up to the fastest wiggle. We build it as a plain `H × H` matrix, one cosine per row,
with no `scipy` and no FFT trick: `coeffs = D @ chunk`. The one design choice that matters is
the **orthonormal** scaling, which makes `D @ Dᵀ = I`. That buys two things we lean on hard:
the inverse transform is *just the transpose* (`idct` is one matmul, no schedule to undo —
contrast ch1.4's DDPM reverse loop), and, by **Parseval's theorem**, the transform preserves L2
energy exactly. Hold onto that second fact.

## Step 2 — quantize: where the zeros come from

```
[include-by-region: fast.py#quantize]
```

`round(x / step)` is the entire quantizer. Applied to the DCT coefficients, something useful
happens: a smooth motion has almost no energy in the high-frequency cosines, so those
coefficients are already near zero and **round to exactly 0**. The coefficient grid comes out
mostly zeros. That is the compressible structure the next step feeds on — and it is *created* by
changing basis, not by the rounding itself.

Here is the sharp version of the Parseval point. Because `D` is orthonormal, quantizing the
coefficients injects the **same error energy** as quantizing the raw samples would. So the DCT
does not buy you a better reconstruction — it buys you the *same* reconstruction with the error
repackaged into a form you can throw a compressor at.

## Step 3 — BPE, on actions instead of text

```
[include-by-region: fast.py#bpe]
```

This is Karpathy's `minbpe` move, re-derived on action tokens. Start from an alphabet of the
distinct integers; repeatedly find the most frequent adjacent **pair** across the whole corpus
and merge it into one new token id; record the merge. Runs of zeros collapse first — `(0,0) →
A`, then `(A,A) → B` — so a coefficient block's long zero tail becomes one or two tokens. BPE is
**lossless**: it is entropy coding stacked on top of the lossy DCT-and-quantize, so it changes
the token *count* and never the reconstruction. (The artifact `assert`s the round-trip, because
a codec you cannot invert is not a codec.)

## Putting it together, and measuring it

```
[include-by-region: fast.py#codec]
```

The `data` region pulls **real robot action chunks** — the same PushT and ALOHA scripted demos
ch1.7's dataset is built from, replayed in-process for their *actions only* (no rendering, so
nothing here is platform-sensitive), sliced into non-overlapping `H = 24` chunks. We encode
every chunk, learn **one** BPE over the pooled corpus, and count. The measured headline, seed 0:

| tokenizer                       | tokens | reconstruction RMSE |
| ------------------------------- | ------ | ------------------- |
| per-step-per-dim binning        | 1968   | 0.012               |
| **FAST** (DCT → quantize → BPE) | **888**| 0.014               |

**About 2.2× fewer tokens at the same reconstruction error** — and it holds on every seed (2.2 /
2.3 / 2.5×). Read the two columns together: the RMSE barely moved (Parseval promised that), and
yet the token count more than halved. Where did the tokens go? Into the ~17% of coefficients
that quantized to zero and the BPE merges that ate their runs. That is the entire trick, and it
is honest: no policy, no rollout, no success rate — a codec, measured as a codec.

## Scout the data before you trust the method

Textbook FAST goes further: it *truncates*, keeping only a handful of low-frequency coefficients
and discarding the rest, on the theory that "the robot never jerks." That works on smooth **human
teleop**. It does **not** work here, and the honest reason is worth internalizing. Our scripted
experts switch phases — approach, grasp, carry — and each switch is a step-*discontinuity* in the
action, which is exactly the high-frequency content a low-pass throws away. Truncating our chunks
to 6 of 24 coefficients drives RMSE to 0.21 — a fifth of the whole action range, unusable. So on
*this* data the compression comes from Parseval + BPE-on-zeros, not from truncation. The toy — a
deliberately **smooth synthetic chunk** — is where truncation is cheap (keep 6 of 24 → RMSE ~0.04)
and where the demo lets you watch the mechanism at its best. Same method, different data, honestly
different result: check the structure your method assumes before you believe a number.

## Break it: is it the DCT, or the BPE?

```
[include-by-region: fast.py#measure]
```

It is tempting to think BPE is doing the real work and the transform is decoration. Settle it by
generating the failure. `--break time_domain` spends the *same kind of budget* — keep fewer
numbers, reconstruct — but in the **time domain**: keep every Nth action and zero-order-hold it,
then quantize the staircase. Predict what that does to the reconstruction, then run it.

| metric               | clean FAST | `--break time_domain` |
| -------------------- | ---------- | --------------------- |
| reconstruction RMSE  | 0.014      | **0.220** (~15×)      |
| error jerk           | 0.0012     | **0.264** (~200×)     |

The motion falls apart. A trajectory's information is spread across *every* timestep — there is
no basis in the time domain where energy concentrates — so dropping timesteps and holding is a
jerky staircase with fifteen times the error. Keep the same number of *coefficients* and the
smooth motion comes back; keep the same number of *samples* and it does not. **The DCT basis is
load-bearing.** (One honest note the pilot's playbook forces: the *naive* comparison in the
contract — plain per-step binning giving "an order of magnitude more tokens" — does not survive
measurement, because BPE happily compresses a slow time-domain signal too. The claim that
reproduces, and the break that actually degrades, are the ones above.)

## The fork: FAST vs flow

You now hold both decoders, so name the tradeoff plainly. **FAST** decodes autoregressively — one
LM step per token, ~35 sequential steps for one of these chunks — but every step is a plain
cross-entropy over a vocabulary, so it reuses a language model's whole stack and gives you exact
per-token likelihoods for free. **Flow** (ch1.5) decodes the entire chunk continuously in a few
Euler steps flat, no vocabulary and no autoregression, but it needs a bespoke sampler and gives
no discrete likelihood. Tokens buy language-model reuse; flow buys a short, constant decode. pi0
famously ships *both* — flow for real-time control, FAST for pretraining — and now you know why
neither is simply "better."

## Read the real thing

Everything here has a production form in `openpi`'s `pi0_fast.py` and its FAST processor: the
cosine transform, the scale-and-round quantizer, the byte-pair merges, and the autoregressive
detokenize that runs your `bpe_decode → dequantize → idct` in reverse. You will recognize every
line. What the real processor adds is scale (a real action vocabulary, real teleop) and the wiring
into an LM cross-entropy head — the fork this chapter only sketched.

## What's next

You turned continuous motion into a short string of integers and back, and measured — honestly,
on real robot chunks — exactly what the DCT, the quantizer, and the BPE each contribute. That is
the last piece of the modern VLA data stack: a policy can now speak actions the way it speaks
words, and the same transformer that reads an instruction can write the motion that answers it.
