# 5.3: Control From Pixels: Visuomotor Behavior Cloning

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## The escape hatch, removed

Way back in **chapter 1.1** you cloned PushT from a ten-number state vector (pusher xy,
block pose, target pose) and it just *worked*. A three-layer MLP, plain MSE, done. Then in
**chapter 1.8** you built a vision-language-action policy, gave it a camera, and ran
`--break blind`: zeroing the image feature changed PushT success by *nothing*. The policy
never used its eyes. We told you why: PushT is solvable from the state, and a random-init
encoder had nothing to add, but that was an argument, not a measurement. This chapter turns
it into one.

Here is the whole move: **take away the state.** The policy gets one input, a live 64×64
top-down frame, and must produce the pusher velocity from pixels alone. No coordinates, no
cheating. The same BC loop from chapter 1.1 (load pairs, fit a network, roll it out, export
to ONNX), but the observation is now an image. And the moment the state is gone, one thing
you could ignore in chapter 1.8 becomes the entire chapter: **the quality of the encoder that
turns pixels into features.**

`pixels.py` runs that BC loop *twice*, changing exactly one part, the encoder:

- **ALIGNED**: a tiny ViT that has been contrastively pre-aligned to the scene geometry
  (the chapter 5.2 recipe, re-contained and compact). Frozen; only a thin adapter and the
  policy head train.
- **RANDOM**: the *identical* ViT architecture, never aligned. A fixed random projection of
  the pixels, exactly the kind of encoder chapter 1.8's VLA looked through and ignored.

Everything downstream (the adapter, the head, the BC loss, the rollout) is held fixed. The
only variable is what the encoder learned. That is the experiment.

## The misconception this kills

> *"Pixels → action is just behavior cloning with more inputs."*

It is not, and the reason is the whole lesson. In chapter 1.1 the input was a state vector
that already *was* the answer: block position, target position, a straight line between them.
Any encoder, even the identity, hands the MLP everything it needs. Swap in pixels and that
free lunch disappears. Raw pixels do not linearly encode "where is the block relative to the
target"; *something* has to extract that. From pixels with no state, **the encoder's quality
is not a detail of the input: it is the entire policy.** A random projection scrambles the
geometry; the BC head, no matter how well trained, is cloning from noise.

## The tiny ViT (the shared backbone, re-contained)

Every chapter re-contains its own backbone. The repetition is the lesson. The encoder is the
same block shape chapter 1.8's VLA used: a stride-8 conv patch-embed turns the 64×64 frame
into an 8×8 grid of 64 patch tokens, a learned `CLS` token leads the sequence, learned
positions are added, and a few pre-norm self-attention blocks let every patch talk to every
other. The `CLS` row that comes out is the image feature the policy conditions on.

```
[include-by-region: pixels.py#model]
```

Read `PixelPolicy`. Its `forward` takes the **flat image** as the observation (the encoder
lives *inside* the policy), so the entire pixels→action path exports as one ONNX graph under
tensor contract v1 (the encoder reshapes `[1, 12288]` back into a frame). That is the honest
consequence of chapter 1.8's lesson that a frozen encoder is part of the policy's input
contract: here we carry it literally into the deployed graph.

## Aligning the encoder (the ch5.2 recipe, compact)

Chapter 5.2 taught contrastive alignment in full. We re-contain a compact version, because
this chapter needs the *product* (an aligned encoder) more than a second full treatment. The
objective is a symmetric InfoNCE: for each frame in a batch, pull its image feature onto its
own scene geometry and push it off every other example's. The encoder is never told the
action, only that *these pixels and this geometry belong together*, and that is enough to
make its features carry where the block and pusher are.

```
[include-by-region: pixels.py#align]
```

A note on faithfulness to 5.2: there the partner is a **text** tower; here it is the
**state**, because on single-instruction PushT the geometry, not the words, is the signal
that makes pixel features control-relevant. The *mechanism* is identical (a tiny ViT, a
partner tower, symmetric InfoNCE); only the partner differs. And, exactly like chapter 1.8's
frozen encoder, the aligned encoder is **rebuilt deterministically from the seed**: CPU
alignment is reproducible, so re-running it reproduces the weights with no checkpoint binary
in git.

## Cloning from frozen features

With the encoder aligned and frozen, the rest is chapter 1.1 with the observation swapped.
We featurize every frame once, standardize, and fit the adapter + head with plain MSE. The
random-encoder run is byte-for-byte the same loop over a ViT that skipped alignment.

```
[include-by-region: pixels.py#train]
```

## What the measurement says

Run it: `--seed 0 --device cpu`, wall-clock **~2.95 min** (measured on cpu-laptop; see
`curriculum/common/wallclock.csv`, and the banner prints it at startup). Both policies
roll out from **pixels alone**, and we report a **Wilson 95% interval** (chapter 1.6), because
pixel-BC success is noisy and a bare percentage lies:

```
[include-by-region: pixels.py#eval]
```

The thing to watch is the **direction**, not an absolute number. The reproducible one is a
**probe on the frozen features**: fit a tiny linear map from the encoder's features to the
expert's action, and compare the held-out error for the *aligned* encoder against a *random* one
(`probe_val_mse`, aligned vs random). When the alignment has made the features carry the geometry,
that action-probe error is lower. Aligned beats random on **every seed: +0.028 / +0.040 /
+0.054.** That is the gated headline, and it stands on its own: aligned features are measurably
more *control-useful* than random ones.

**What floors, and why (said once, then trusted for the rest of the chapter).** The closed-loop
`success_rate` is a harder bar than the probe, and at free-tier scale it **floors at 0/12 for
both** encoders: the rollout gap is 0.0. That is not the experiment failing. It is the ceiling of
a tiny from-scratch encoder aligned to a few hundred frames, nowhere near SigLIP quality, cloning
a non-Markovian expert one action at a time from a single 64×64 frame. Two things follow, and we
will not repeat them each section. First, the reproducible signal is the **probe**, not a rollout
win, so that is what the chapter gates (chapter 1.6). Second, absolute pixel-BC success is small
and platform-sensitive: MuJoCo rasterization is **not bitwise across CPU architectures** (chapter
1.8), so we report a **direction** under a Wilson interval, never a promised number. Driving PushT
end to end from a scaled backbone is the **Scale Lab**. That honest ceiling *is* the lesson: a
from-scratch encoder is a fixed, weak projection of the pixels, which is the whole reason the real
answer (the read-the-real-thing) is a backbone aligned at internet scale.

## The payoff to `--break blind`

This is the measured answer to chapter 1.8's cliffhanger. There, zeroing vision was a no-op,
because the state made vision redundant and the random encoder added nothing. Here the state
is gone, and the two encoders sit at opposite ends of the same experiment: the *aligned* features
carry enough of the geometry that a linear probe reads the expert's action out of them better than
from *random* features, on every seed. That gap **is** the thing chapter 1.8 could only assert:
that a pretrained, aligned backbone is what makes vision actually matter. When the state cannot
bail you out, the encoder is the whole policy, and the quality of its features is the whole
ballgame.

## Break it yourself: the trainable-encoder trap

You wired the encoder→policy path. The obvious next thought: *why freeze the encoder? Won't
training the whole thing end-to-end beat training a thin adapter?* Predict the answer before
you run `--train_encoder` (that is **ex2**), then run it. At free-tier scale it is a trap: a
from-scratch ViT has enough capacity to memorize the handful of frames it trained on, so the
pixels→action map overfits: a tiny training loss sitting over a worse held-out fit (the tell is
that overfit gap, not a rollout number). Then explain it to yourself: chapter 1.1 trained
its whole network end-to-end and was fine. Why is this different? (A ten-number state cannot
be memorized into a lookup table; a tiny frame set can. The alignment already did the
expensive, transferable work. Leave it frozen.)

## Read the real thing

The definitive version of "a pretrained, *aligned* vision backbone feeds the action decoder"
is **OpenVLA** (`openvla/openvla`; the author pins the verified commit). Where our policy runs
one frozen `CLS` feature through a thin `nn.Linear` adapter into a BC head, OpenVLA runs images
through **Prismatic's fused SigLIP + DINOv2 backbone** (vision features aligned to objects and
language at internet scale), projects them into an LLM's token space, and decodes actions. Read
it for the one structural echo of this chapter: the backbone is **frozen** (or only lightly
adapted), because the alignment is the expensive, transferable part: the same reason our
`--train_encoder` trap loses to the frozen aligned encoder, and the same aligned-vs-random gap
this chapter measures, scaled up a million-fold. (Alternative: `Physical-Intelligence/openpi`'s
pi0, a PaliGemma/SigLIP image path into a flow-matching action expert.)

## What we cut

- **A real aligned backbone.** Ours is a tiny from-scratch ViT aligned to sim state on a few
  hundred frames: *modestly* better than random, not SigLIP. The whole point of OpenVLA /
  pi0 is that the alignment is pretrained at scale. That is the read-the-real-thing.
- **Action chunking and history.** We clone a single action from a single frame (chapter 1.1's
  shape). ACT (chapter 1.3) and real visuomotor policies predict a chunk from a short history,
  likely what a harder task would need.
- **Language.** PushT has one instruction, so alignment here is image↔state. Chapter 5.2's
  image↔text alignment is what you would reach for on a multi-instruction task.

## Exercises

- **ex1 (predict-then-run):** aligned vs random encoder: whose frozen features are more
  control-useful? (The gated signal is the action-probe direction; the encoder is load-bearing
  once the state is gone.)
- **ex2 (predict-then-run + the trap):** frozen aligned encoder vs `--train_encoder`. Predict,
  run, and self-explain why unfreezing overfits here but not in chapter 1.1.
- **ex3 (code-completion):** write the symmetric InfoNCE at the heart of the alignment.
