# 5.1: Patches & Attention: A ViT From Scratch

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## The frozen stand-in, revisited

Every VLA you built in Phase 1 saw the world through `FrozenVisionEncoder`, a small,
random-init, never-trained conv stack (ch1.7). We were honest about it: a frozen random CNN
is a fixed nonlinear projection of the pixels, not perception. It preserved coarse layout and
nothing more, and `--break blind` proved the policy barely used it. The whole reason to reach
for a *pretrained* backbone (SigLIP, DINOv2) was to close that gap.

Before you can adapt a pretrained vision tower, you have to know what one *is*. This chapter
builds the architecture underneath all of them, from scratch, over cached PushT frames: a
**Vision Transformer**. No convolutions, no `transformers`, no `einops`: a patchify reshape,
a CLS token, learned positions, and the exact pre-norm attention `Block` you fused with in
ch1.8, re-derived here. Open `vit.py`. Seven regions: **setup**, **data**, **patches**,
**model**, **train**, **probe**, **report**.

## A ViT sees no picture

Here is the single idea the rest of the chapter defends. A ViT does not process an image the
way a CNN does, sliding receptive fields over a grid. It **cuts the image into patches,
flattens each into a plain vector, and treats the result as a SEQUENCE**, a bag of tokens.
The only thing that tells the model two patches were neighbors is a **learned positional
embedding** we add to each. Order is not baked in; it is a parameter.

```
[include-by-region: vit.py#patches]
```

`patchify` is the whole trick, and the whole trap. To turn a `(B, 64, 64, 3)` image into
`(B, 64, 192)` (64 patches, each an 8×8×3 = 192-vector), you split `H → (grid_row,
patch_row)` and `W → (grid_col, patch_col)`, then **permute the two grid axes in front of the
two pixel axes** before flattening, so each output row is one spatially-contiguous patch. Skip
the permute and the grid axis interleaves with the pixel axis: every "patch" becomes a
scramble. The shape is still `(B, 64, 192)`, so nothing errors. Hold that thought: it is the
deliberate failure at the end.

## Cache the frames, label them for free

```
[include-by-region: vit.py#data]
```

We replay the scripted PushT expert (the ch1.7 pattern) and cache ~740 frames **once**, at
the free-tier floor of 64×64, half the pixels of ch1.7's 96×96. Each frame carries a label
that costs nothing: **which quadrant the block sits in**, read straight off the simulator
state (`obs[2:4]` is the block's x, y). No annotation, no human, a deterministic function of
the state that produced the pixels. We split by *episode*, not by frame, so near-duplicate
frames from one episode never straddle train and test (ch1.6).

## The ViT itself

```
[include-by-region: vit.py#model]
```

The `Block` is ch1.8's, re-derived: pre-norm multi-head self-attention (Q/K/V are three
`nn.Linear`, a scaled-dot-product softmax mixes tokens, an output projection), then a per-token
MLP, both on residuals. `TinyViT` embeds the patches with a single `nn.Linear` (a `Conv2d`
with `stride=8` is the *exact same operation*, which is how the real towers write it), prepends
a learned CLS token, adds the learned position table, runs the blocks, and reads the **CLS
row** as the pooled scene representation. Free-tier dims are deliberately tiny: `dim 96, depth
2, heads 3`. This is a `nanoGPT`-tiny, in pixels.

## Training a transformer is not free

```
[include-by-region: vit.py#train]
```

A from-scratch transformer does not just train: it cold-starts badly. Two things keep it
honest here: a **modest learning rate** and a **linear LR warmup**. Drop the warmup
(`--warmup 1`) and some seeds pin at chance for the *entire* run: the loss never leaves
`ln 4`. This is not a detail to hide; it is a real, measured property of small transformers,
and it is why every production recipe warms up. Feel it before you trust the numbers.

## The measurement: does the representation carry the scene?

```
[include-by-region: vit.py#probe]
```

We do not grade the ViT by its training accuracy (of course it fits its own labels). We grade
its **representation** with a **linear probe**: freeze the backbone, extract the CLS feature,
and fit a closed-form least-squares read-out of the quadrant (ch1.7's `lstsq` trick, no SGD).
Probe accuracy is how *linearly accessible* the scene fact is in the pooled vector. We run it
on the trained ViT **and** on a same-shape random-init ViT, and compare to the majority guess.

The measured direction, seed 0 (and it holds on every seed):

| representation            | probe accuracy |
| ------------------------- | -------------- |
| trained ViT (CLS)         | ~0.89          |
| random-init ViT (CLS)     | ~0.69          |
| majority guess            | ~0.33          |
| chance (1/4)              | 0.25           |

Read this carefully, because it is more honest than "the ViT learned to see." A **random-init**
ViT, never trained, already probes to ~0.69, far above the majority guess. That is the
point: *which quadrant* is nearly a **bag-of-patches** property, and a random projection of
patches preserves it. What training adds is real but layered on top: it lifts the probe another
~0.20, **and** it teaches the attention to concentrate: the CLS token's attention-rollout
peaks ~10–16× over uniform on the patches that actually contain the block, where the
random-init ViT's attention is flat. That concentration is the toy: hover a frame and watch the
trained map find the block while the random map washes out. Report the **order** (trained >
random > majority), never the exact %: these are rendered images (not bitwise across CPU
arches) read through a small held-out set (ch1.6).

## Break it: the bug your accuracy can't see

Now collect on the patchify trap. You wrote the reshape in the exercise; the classic mistake is
to forget the permute, globally permuting the patch set. Predict what it does to the probe,
then run `--break patch_interleave`.

If you predicted the probe crashes to chance, you are in good company, and wrong. The probe is
**silent**: ~0.89 clean, ~0.92 interleaved. A coarse label is a permutation-invariant property
of the bag, and a *shape-preserving reshape bug is exactly a permutation*, so the ViT relearns
the fixed reordering and the accuracy shrugs. **That silence is the danger.** A patchify bug
will not show up in your metric; it shows up only in the attention map, which no longer lines
up with the image. The lesson of the ViT is also its footgun: patches are a bag.

The companion misconception, "a ViT sees the image as a *picture*", falls to
`--break shuffle_pos`, which permutes the learned position embeddings with the **pixels
byte-for-byte identical**. The coarse probe survives (a bag barely needs order for a quadrant),
but the trained model's *edge over random* nearly vanishes (the gap collapses from ~0.20 toward
zero) and its attention map scrambles. The pixels never changed; the position **tags** did. The
model was never looking at a picture. It was looking at tokens wearing position tags, and the
spatial structure it learned lived in those tags.

## Read the real thing

Everything here has a production form. The read-the-real-thing segment pairs this chapter with
a real SigLIP ViT vision tower (`google-research/big_vision`, `models/vit.py`, or the SmolVLA
`vision_model` in the already-pinned `huggingface/lerobot`). You will recognize every piece:
the patch embedding is a `Conv2d(stride=patch)` (our flattened-patch `Linear`), the class/pool
token is our CLS, the learned positional embeddings are our `pos` table, and the pre-norm
attention stack is our `Block`. What the real tower adds is scale and, above all, **pretraining
on image–text pairs**, which is what makes its features object- and language-aligned and
transferable. That is the Scale Lab, and the reason to prefer adapt-pretrained when performance
matters.

## What's next

You built the architecture a vision backbone is made of and measured, honestly, what a tiny
from-scratch one carries: a coarse scene fact, more accessible after training, with attention
that localizes the object, and a random-init baseline closer than you'd like, because the task
is easy. The gap between this and a policy-ready visual representation is exactly the gap
pretraining fills. The next chapters put a real, pretrained tower to work.
