# 5.2: Why Aligned: Contrastive Vision-Language Pretraining

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## From "what do I see?" to "does this match what I said?"

Chapter 5.1 trained a tiny ViT with a **supervised probe**: you handed it a label,
which quadrant the block sits in, and a linear head learned to read that label off the
pixels. It works, and it is the honest way to ask *does my encoder see anything?* But a
label is a thin description. "Top-left" is one of four buckets; the caption *"the block
is far in the top-left corner"* says more, and there is no fixed list of captions to
enumerate. This chapter asks the harder question a real robot needs: can an **image**
and the **words** that describe it land in the *same* space, so that a sentence you have
never turned into a class can still pull up the right picture?

That is **contrastive vision-language pretraining**, the idea behind CLIP, and behind
the vision backbones real VLAs run on (SigLIP inside SmolVLA and OpenVLA). The surprise,
and the thing to hold onto, is the supervision: **there is none, in the usual sense.**
We never tell the model what a frame contains. We only tell it *which caption came with
which frame*, the pairing, and let it sort out the rest. Open `align.py`. Seven
regions: **setup**, **data**, **language** (the text tower), **vision** (the image ViT),
**contrastive** (the loss), **eval** (retrieval), and **report**.

## The data: cheap captions from sim state

```
[include-by-region: align.py#data]
```

We reuse PushT. Replaying the scripted expert sweeps the block across the whole table,
so a few thousand cached 64×64 frames span every quadrant and every distance from the
center. The **caption** for each frame is generated from the block's position in the
simulator state, a *cheap label*, no human annotation: a quadrant word (`top left`, …)
from the sign of the block's coordinates, and a `near`/`far` word from its radius. A
handful of paraphrase templates keeps the wording from being a single fixed string.

Read the boundary carefully, because it is the whole point of the chapter: the
contrastive model **never sees these quadrant or near/far labels**. It only ever sees
*this frame goes with this sentence*. The labels exist for two jobs that stand *outside*
the contrastive learner: to train the supervised baseline we will race against, and to
*score* retrieval at the end. Pairing is not labeling, and the difference is the lesson.

## Two towers into one space

An image and a sentence are different kinds of object; to compare them we need to map
both into a single vector space where "close" means "these belong together." That takes
two encoders, **towers**, that meet in the middle.

The text tower is deliberately trivial (the interesting one is the image ViT):

```
[include-by-region: align.py#language]
```

We re-contain ch1.7's word-level **tokenizer** (string in, fixed-length id array out,
no `transformers`, no BPE), then embed the ids, run *one* attention block so the words
can talk, masked-mean-pool to a single vector, and project. That is all a caption this
simple needs. A real CLIP text encoder is a full Transformer over a 30,000-token subword
vocab; ours is a couple dozen words. The *mechanism* is the same; the scale is not.

The image tower is the tiny ViT from 5.1, re-derived here (chapters re-contain their
backbones; we do not import 5.1):

```
[include-by-region: align.py#vision]
```

Patch-embed the frame into 8×8 = 64 tokens, prepend a CLS token, add learned positions,
run a few pre-norm attention blocks, read the fused scene off CLS, and project. One
detail earns its line and is easy to skip: **background subtraction.** A 64×64 top-down
PushT frame is ~98% constant table: the block is a handful of pixels. Feed that raw to a
tiny ViT and the CLS token is dominated by the blank background; every frame maps to
nearly the same point and nothing learns. (Measure it: a linear probe on raw pixels sits
at chance; on background-subtracted pixels it is near-perfect: the signal was always
there, drowned by the constant.) So we subtract the mean frame first. It is not a trick;
it is what makes a tiny encoder on near-blank sim frames learnable at all, and worth
remembering the next time a "working" architecture refuses to train on flat inputs.

## The loss: pull the pair, push everything else

Cross-entropy down the rows (each image picks its caption) *plus* cross-entropy
across the columns (each caption picks its frame), over the $B\times B$ matrix of
cosines scaled by a learned temperature $\tau$, and here it is in code:

$$
\mathcal{L} = -\frac{1}{2B}\sum_{i=1}^{B}\left[ \log\frac{\exp(s_{ii}/\tau)}{\sum_{j}\exp(s_{ij}/\tau)} + \log\frac{\exp(s_{ii}/\tau)}{\sum_{j}\exp(s_{ji}/\tau)} \right],
\qquad s_{ij} = z^{\mathrm{img}}_i \cdot z^{\mathrm{txt}}_j
$$

```
[include-by-region: align.py#contrastive]
```

Here is the engine. Take a batch of B (frame, caption) pairs, L2-normalize every
embedding onto the unit sphere so a dot product *is* a cosine, and form the B×B matrix of
image-to-caption cosines. If the space were perfect, that matrix would be bright on its
diagonal, each frame matches *its* caption, and dark everywhere else. **Symmetric
InfoNCE** makes it so: a cross-entropy down the rows (each image should pick its caption
out of all B) *plus* a cross-entropy across the columns (each caption should pick its
frame), scaled by a **learned temperature**. Two forces, every step: *pull* each matched
pair together, *push* every mismatched pair apart.

Where do the "mismatched pairs" come from? For free: every *other* caption in the batch
is a negative for your frame. That is why contrastive learning wants a **big batch**
(128–256): more captions in the batch means more negatives, a harder and more informative
lesson. It fits the free-tier floor here only because the towers are tiny. Flag that as
a real constraint, not a detail. And it is why the "no labels" claim holds: the negatives
are handed to you by the *pairing*, not by any annotation.

## Alignment is a number: retrieval

A shared space is a claim; retrieval is the measurement that makes it honest.

```
[include-by-region: align.py#eval]
```

Split the cache into a disjoint held-out **gallery** (images) and **query** (captions).
For each query, *"the block is near the top left corner"*, rank every gallery frame by
cosine and take the top one: is it actually a top-left, near frame? Average over queries
and you have **retrieval@1**. We score it two ways: **fine** (the retrieved frame must
match on quadrant *and* near/far, the whole caption) and **quad** (quadrant only, the
coarser bar). Chance is ~1/8 and ~1/4.

Then we run the same measurement over **three** encoders:

- **aligned**, the contrastive towers above.
- **random**, the same two towers, never trained. The floor.
- **supervised**, 5.1's probe: a ViT trained on the 4-way quadrant *label*, then frozen
  and aligned to text. This is the "contrastive needs labels?" foil: it *uses* them.

The result (measured, seed-robust in **direction**, never trust the exact number on
rendered frames across machines): **aligned ≫ random** by a wide margin: contrastive
built a real shared space from pairing alone, while random init retrieves noise. That is
the rock. On the finer question, **aligned > supervised**: both nail the quadrant (on the
quad score they roughly tie, the probe was *trained* on quadrant, so say so), but the
supervised encoder compressed the scene to a 4-way label and **lost the near/far half of
the caption**. Contrastive kept the whole sentence. The encoder that saw *no labels* wins
on the richer question, which is the misconception, dead on the page.

> **Misconception: "contrastive learning needs labels."** It does not. The only signal is
> which caption rode along with which frame. Our supervised baseline literally *does* use
> quadrant labels and still loses. Negatives, the thing that makes contrastive work, come
> from the batch, not from an annotator.

## Break it: forget the negatives

You will write this loss yourself, and there is one bug almost everyone writes on the
first try. Contrastive learning is *pull* and *push*. It is tempting to write only the
pull: maximize the cosine of each matched pair and stop. `--break noneg` is exactly that,
drop the negatives, keep the positive term:

```python
# correct: cross-entropy over the B x B matrix  (pull the diagonal, push the rest)
# noneg:   (1.0 - (img_e * txt_e).sum(-1)).mean()   # pull only — no push
```

Predict before you run (exercise 1). Then measure: retrieval@1 falls sharply, in the
default run from ~0.79 to ~0.29, yet stays *above* random (~0.07). (Read these for their
*direction*, not the exact percentages: retrieval over rendered frames shifts across
machines, but the collapse reproduces every seed.) That "plausible but worse"
signature is the tell. Ask yourself what stops the buggy model from mapping **every**
frame and **every** caption to the *same* vector: the loss would be zero and retrieval
useless. Nothing stops it. That is representational **collapse**, and it is what the
negatives were preventing. (Retrieval clings above chance only because the tiny surviving
structure from initialization and the shared background subtraction is not fully washed
out.) Two forces, not one; the push is not optional.

## Read the real thing

Open `openai/CLIP`, `model.py`. You will recognize it immediately: two towers projecting
to a shared, L2-normalized space; `logits_per_image` and `logits_per_text`; two
cross-entropies against `arange` labels; and a **learned `logit_scale`**, the same
clamped log-temperature we use, `exp()`'d. Everything you built here is there at scale: a
big ViT and a real Transformer text encoder instead of our two toys, hundreds of millions
of image-text pairs instead of a few thousand PushT frames, but the *objective is
line-for-line ours*.

Then look at what modern VLAs actually ship: `google-research/big_vision`,
`models/proj/image_text/two_towers.py`. **SigLIP** swaps the softmax InfoNCE for a per-pair
**sigmoid** loss. The reason is exactly the batch constraint we hit: a softmax over
in-batch negatives couples the whole batch, while the sigmoid scores each pair
independently, which decouples the loss and scales to enormous batches far better. That is
why SmolVLA's vision backbone is a SigLIP model, and why the "big batch" note in this
chapter is not a toy detail but the thing the next design decision turns on.

## What's next

You now have an encoder whose space *language can point into*. That is the missing piece
under a real VLA: 5.1 asked whether an encoder sees, 5.2 aligned what it sees to what you
*say*, and the perception arc closes (5.3) by putting these features to work: a
pretrained, language-aligned backbone is precisely what lets a policy follow an
instruction it was never explicitly trained on. The from-scratch towers here are a
mechanism demo on near-blank frames; the payoff is understanding, cheaply and completely,
the objective that a web-scale CLIP/SigLIP spends millions of pairs to earn.
