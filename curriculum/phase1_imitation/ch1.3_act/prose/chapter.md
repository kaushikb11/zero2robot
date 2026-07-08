# 1.3: ACT — Commit to the Chunk

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## See it work

The right arm reaches into its half of the table, closes on the cube, and
carries it to the middle. The left arm is already waiting there. For a moment
both grippers hold the cube; then the right one opens and backs off and the left
one carries it the rest of the way to a target the right arm could never reach.
Neither arm can do this task alone — the reach is split down the middle — so the
whole thing turns on a hand-off that has to happen at the right place and the
right instant. Watch the policy do it. Then remember that a single-step policy —
chapter 1.1's whole model class — has no place to keep a plan across the release,
and drops the cube at this hand-off far more often, for a reason this chapter is
about.

What changed is not the data and not the loss. It is the *shape of the output*.
A behavior-cloning policy predicts one action, then looks again, then predicts
one action, at 10 Hz forever. ACT predicts the next eight actions at once — a
*chunk* — and commits to them. That single change is enough to turn a policy that
averages itself into paralysis at every fork into one that picks a plan and
follows it through the hand-off. Measured on this env, that one change takes
held-out success from 0.6 (single-step) to 0.9 (chunked).

## The problem

Chapter 1.2 left us at a wall we could name but not climb. Behavior cloning fits
the average action, and at a state where two futures are equally good, the
average of "go left" and "go right" is "drive straight into the middle." We
curated the data until the wall was as low as data could make it, and the policy
still topped out well short of the expert. The diagnosis at the end of 1.2 was
blunt: it is the model class. A function that commits to one action per state,
re-evaluated every 0.1 seconds, *cannot* represent a temporally extended plan.

The bimanual hand-off makes this concrete. Consider the instant the right arm
arrives at the middle with the cube. The correct behavior is a little sequence:
hold, wait for the left gripper to close, *then* open and retreat. A single-step
policy sees only the current frame and has to re-derive that whole intention from
scratch every step — and near the hand-off, tiny errors in "am I holding or
releasing" compound into dropping the cube. Nothing about predicting one action
at a time lets the policy say "I am three steps into a five-step release." That
memory has to live somewhere, and in ACT it lives in the chunk.

## Build

`act.py` is one file, about 380 lines, in six regions: setup, data, model,
train, eval, report. It generates scripted-expert demos of the cube transfer,
reshapes them into action chunks, builds a small transformer from scratch,
trains it with plain L1, and rolls it out with temporal ensembling. There is a
real transformer in here and nothing is imported to hide it — no `transformers`,
no `timm`, no attention you cannot read.

### Setup

```
[include-by-region: act.py#setup]
```

The scale knobs are the story of the chapter as flags. `--chunk_size` (K) is how
far ahead the policy commits; `--model_dim` is the width of the transformer.
Every default is free-tier first — the whole run fits on a CPU laptop in a few
minutes. A note the env forces on us: aloha_cube episodes are *short*, a median
of about 27 control steps, because the env is a deliberately tame planar model
with a weld-constraint grasp (see its README). So keep K well under the episode
length — a chunk longer than the task is mostly padding. Eight (about a third of
a median episode) is the default here, and it is not arbitrary: measured, K=8
beats K=16, which in turn crushes K=1 — the "tune the chunk down for short
episodes" rule, as a number. One hundred, which real ACT uses on its
several-hundred-step episodes, would be absurd on 27.

### Data

```
[include-by-region: act.py#data]
```

The demos are the same scripted expert every Phase-1 chapter uses; the only new
step is the reshape. For each frame `t`, the *target* is no longer one action —
it is the next K actions the expert took, `actions[t : t+K]`. Near the end of an
episode there are fewer than K actions left, so we pad the tail by repeating the
last action and record a 0/1 mask that zeroes those padded steps out of the loss.
Get that mask wrong — mark the padding as real — and you quietly train the policy
to predict invented actions at the end of every episode (that is exercise 3). The
observations themselves are unchanged from chapter 1.1: ten numbers, normalized
into the model as buffers so the checkpoint carries its own stats.

### Model

```
[include-by-region: act.py#model]
```

This is the part worth slowing down for. A transformer attends over a *set of
tokens*, but our observation is a flat vector of ten numbers — so we make a set
out of it. We split the ten numbers into four **entity tokens**: the right arm
(x, y, grip), the left arm, the cube, and the target. Self-attention over those
four tokens is exactly where the policy gets to reason about *relationships* —
how far is the cube from each gripper, which arm is in position — instead of
staring at ten anonymous floats. That is the encoder.

The decoder is where the chunk comes from. We hold K learned **query tokens**,
one per step of the chunk. Each query first attends to the other queries (so the
steps of the plan can coordinate — "if step 3 releases, step 4 retreats") and
then *cross-attends* to the encoder's memory of the scene. A linear head turns
each finished query into one six-dimensional action. Eight queries in, eight
actions out, in a single forward pass. The blocks are hand-rolled from
`nn.MultiheadAttention` with pre-norm residuals — the same structure as the DETR
decoder real ACT borrows, shrunk until you can read all of it.

### Train

```
[include-by-region: act.py#train]
```

The loop is chapter 1.1's, almost unchanged: Adam, cosine-decayed lr, a shuffled
permutation for batches. The one difference is the loss. We use **L1** on the
chunk (real ACT's choice — it is sharper than MSE on multimodal action data),
averaged over the valid, unpadded steps. That is the entire training story: map
each observation to the expert's next eight actions, and measure the miss in
absolute value.

### Eval — temporal ensembling

```
[include-by-region: act.py#eval]
```

Here is the subtle idea, and it only exists because we chunk. At execution step
`t`, the policy predicts a fresh chunk covering steps `t … t+K-1`. But step `t`
was *also* predicted by the chunk from step `t-1`, and `t-2`, all the way back K
steps — each of those older chunks reached forward and made a prediction for
right now. **Temporal ensembling** averages all of those overlapping predictions
for the current step, weighting them exponentially so the policy blends its
plans into one smooth, committed trajectory instead of lurching each time a fresh
chunk disagrees with the old one at the seam. It is a running vote across time,
and it costs nothing but a buffer.

The `--break` flag ablates each piece so you can feel what it was worth; the
"Break it" section below reads the numbers.

## Run it

```
python curriculum/phase1_imitation/ch1.3_act/act.py --seed 0 --device cpu
```

<!-- wall-clock table renders from wallclock.csv -->

The result at the default config, seed 0 on CPU, over 25 held-out eval episodes:

| | held-out success | mean return |
|---|---|---|
| untrained (random init) | 0% | −320 |
| trained ACT (chunk + ensembling) | **88%** | **−46** |

The network never touched the environment during training — it only ever saw the
expert's chunks — and it comes out driving the cube through the hand-off on nearly
9 of 10 held-out starts. That gap is the chunking and the transformer doing their
job. Open the recording and scrub the two arms through the hand-off band:

```
rerun outputs/ch1.3-act/act.rrd
```

It is not the expert's 100%, and it should not be: this is a tiny transformer on
fifty demonstrations trained for a few minutes so it fits on a laptop. Scale the
demos, the width, and the epochs and the number climbs — but at *this* budget the
chunked policy already clears a bar single-step behavior cloning never reached on
this task.

## What we cut

This is a real transformer trained the real way, but it is **not** the full ACT,
and the missing pieces matter enough to name:

- **No CVAE.** Real ACT is a conditional variational autoencoder: a second
  encoder reads the expert's *action sequence* and compresses it into a latent
  `z` that the decoder conditions on, trained with a KL term. That latent is how
  real ACT models *multimodality* — a demonstrator who sometimes goes left and
  sometimes right. We dropped it entirely. Our policy is deterministic: obs in,
  one chunk out. It works here because the scripted expert is essentially
  unimodal (one clean way to do the hand-off), so there is little multimodality
  for a latent to capture. On messy human demos, the missing CVAE is exactly
  what you would add back.
- **No images.** Real ACT sees the scene through cameras and a ResNet backbone.
  We train on the ten-number state vector, so there is no vision at all — the
  entity tokens *are* our perception. The env can render a top-down image
  (`--video` demos), and wiring an image encoder onto the front of this same
  transformer is the natural next step.

Neither cut is an approximation that quietly degrades a number; each is a whole
capability left for later, on purpose, so the chunk-and-ensemble core is legible.
The "read the real thing" segment for this chapter walks the original ACT repo
so you can see precisely what these two paragraphs left out.

## Break it

Three ablations, each a real ACT misconception, all measured at the default
config (seed 0, CPU, 25 eval episodes). The first has a large, robust signature;
the other two teach something subtler and more honest about what temporal
ensembling actually buys.

**`--break no_chunk` — "why not just predict one action?"** This forces K=1: the
transformer now predicts a single action, which is chapter 1.1's behavior cloning
wearing an attention costume. Held-out success drops from **0.88 to 0.6** — and
the gap holds across seeds and eval sizes. The chunk was not decoration; it was
the mechanism. A policy that re-decides everything every step cannot hold the
hold-wait-release intention through the hand-off, and the cube gets dropped. This
is the chapter's thesis stated as an ablation, and it is solid.

**`--break no_ensemble` — "the chunk is enough, skip the averaging."** Same
trained weights, but at eval we execute only the first action of each fresh chunk
and throw the overlap away. Now be careful, because the honest result is more
interesting than a clean win. On **seed 0 the policy collapses to 0.0** — total
failure — while on **seed 1 it holds at 0.96**, as high as it ever gets there
(seed 1 *with* ensembling is 0.88, so dropping it cost nothing — if anything the
sign flipped, well inside the noise of 25 episodes). Temporal ensembling did not
reliably raise success; on one seed it was the difference between everything and
nothing, on another it did nothing at all. The mechanism is
the env's honest simplification biting back: the grasp is a *threshold* weld
(close past `CLOSE_FRAC` while within reach of the cube and it latches). The
gripper command sits right at that threshold during the pick, and the temporal
average of several confident chunks is sometimes exactly what tips it over to
latch. Strip the averaging and, on an unlucky seed, the grasp never catches. What
ensembling *reliably* gives you is not success but *smoothness* — scrub
`eval/*/action` in the two recordings and the ensembled trace is smooth where the
no_ensemble one chatters. Its designed payoff, robustness to real-robot
observation noise, is a thing this clean deterministic sim cannot show you at all.

**`--break open_loop` — "commit to the whole chunk, then look again."** Run all
eight predicted actions, *then* re-query. Here it lands at **0.88**, the same as
ensembling — on this task the jerk at chunk seams is visible in the traces but not
fatal to success. It is the cleanest illustration that "chunk without ensembling"
is a real design point, not obviously wrong, and that the smoothness argument for
ensembling is an argument about *trajectories*, not always about task success.

The transferable lesson, and it is a sharper one than "ensembling is good":
chunking and ensembling are two separate ideas bundled under one name. Chunking
buys you a *plan*, and here it robustly buys success. Ensembling buys you a
*smooth* plan; whether smoothness also buys success depends on the task, the seed,
and how marginal your contacts are — and measuring that honestly, instead of
assuming it, is the whole game (chapter 1.6 is about exactly this fragility).

## Read the real thing

You have now built every idea real ACT is made of except the two we cut, and the
original code is public — `tonyzhaozh/act`, pinned here at commit `742c753`. Read it
next to `act.py` and the shape of what we simplified becomes exact.

**Temporal ensembling.** Our `eval` region keeps an `all_time` buffer, marks which
past chunks reached step `t` with a `populated` mask, and averages them with
`weights = np.exp(-args.ensemble_m * np.arange(len(votes)))`. The real loop is in
`imitate_episodes.py`, inside `eval_bc` (around lines 219–260):
`all_time_actions[[t], t:t+num_queries] = all_actions`, then `actions_for_curr_step
= all_time_actions[:, t]`, a "populated" check via `torch.all(... != 0)`, and `k =
0.01; exp_weights = np.exp(-k * np.arange(...))`. This is the one place where our
code is nearly line-for-line the original. The only real differences: theirs runs on
GPU against camera images, gates the whole thing behind a `temporal_agg` flag that
defaults *off* (you opt in, with `query_frequency` set to 1), and uses `k=0.01`
where our default is `0.1`. If you read one file to confirm we did not fake the eval,
read this one.

**The CVAE we cut.** Our `ACTPolicy` is deterministic: obs in, one chunk out, no
latent anywhere. The real policy's loss, in `policy.py` (`ACTPolicy.__call__`,
lines 18–35), is `l1 + kl_weight * kl` — a second term we never compute. The
machinery lives in `detr/models/detr_vae.py`: a *style encoder* (`encoder_action_proj`,
`cls_embed`, ~lines 69–104) reads the expert's whole action sequence, `latent_proj`
splits it into `mu`/`logvar`, and `reparametrize` (lines 17–20) samples a 32-D latent
`z` the decoder conditions on. At eval, `z` is zeroed to the prior mean (lines
112–113). That entire path is how real ACT represents a demonstrator who sometimes
goes left and sometimes right — the multimodality our unimodal scripted expert
doesn't have, which is why we dropped it whole rather than approximated it.

**The DETR transformer.** Our `EncoderBlock` / `DecoderBlock` (self-attention, then
cross-attention to memory, pre-norm) mirror `detr/models/transformer.py` —
`TransformerEncoderLayer` and `TransformerDecoderLayer`, with the same `self_attn` +
`multihead_attn` cross-attention and the `forward_pre` / `forward_post` norm variants
we chose pre-norm from. Real ACT runs 4 encoder / 7 decoder layers at 8 heads (the
ACT branch of `imitate_episodes.py`: `enc_layers = 4`, `dec_layers = 7`, `nheads =
8`) where we run 2 / 2 at 4. And it feeds *image* tokens: `detr/models/backbone.py`
is a ResNet-18 whose feature map becomes the token set, with sinusoidal
`detr/models/position_encoding.py` — the whole vision front-end our four entity
tokens stand in for.

None of these three make our version wrong; they make it a legible minimum, with the
production hardening peeled off so the chunk-and-ensemble core is the only thing on
the page. **Read them in this order:** `imitate_episodes.py` `eval_bc` first (you
already know it), then `policy.py` (short — find the `l1 + kl` loss), then
`detr/models/detr_vae.py`'s `forward` (the CVAE encoder, the piece we never built),
and last `detr/models/transformer.py` (the blocks you hand-rolled, at full size).

## Exercises

Four, in `exercises/`. Two ask you to commit to a prediction before the run is
allowed to answer — the chunked policy against the untrained baseline, and the
chunked policy against its own single-step (K=1) ablation. One is a bug-hunt where
the padding mask marks invented actions as real and every metric still prints
clean. One has you implement the temporal-ensembling weighting from its
definition, since it is the one line the whole eval turns on.

## What's next

You now have a policy that predicts the future in chunks and executes it
smoothly. But it still learns by *imitation* — its ceiling is the demonstrator,
and it has never once been told whether an action was good, only whether it
matched the expert. It cannot discover a hand-off the expert never showed it, and
it cannot recover from a state no demonstration ever visited. The next phase stops
handing the policy answers and starts handing it a *reward*: the policy tries,
fails, and improves from its own experience. Everything you have built — the
chunk, the transformer, the honest env — carries forward; what changes is where
the learning signal comes from.
