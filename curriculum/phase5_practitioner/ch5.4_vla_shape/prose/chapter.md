# 5.4: The Production VLA Shape: Prefix, Suffix, and the Action Expert

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## One tower was a teaching simplification

In ch1.8 you built a VLA as a single tower. You laid the three inputs out as one sequence
(`[CLS, vision, state, tok_0..tok_15]`), ran a few self-attention blocks, read the fused
representation off the **CLS token**, and flowed a **single action** out of that one vector. It
worked as a lesson. It is not the shape of a production VLA.

Open `pi0`, `SmolVLA`, or `OpenVLA-OFT` and you find the same different picture every time: two
towers sharing one attention. A **prefix** (the vision-language model) and a **suffix** (an
**action expert** that reads the prefix and emits a whole *chunk* of actions). This chapter
graduates ch1.8's single tower into that shape, by hand, PushT-only, still no `transformers`.
Open `vla_shape.py`. Six regions: **setup**, **data**, **model**, **train**, **eval**, **report**.

## The claim, up front (it is a mechanism, not a score)

We keep ch1.8's honesty. The vision here is still ch1.7's **frozen random CNN**, and PushT is
solvable from state, so, as in ch1.8, this from-scratch policy will **not** drive PushT to
success in a closed-loop rollout. That is fine, because the thing this chapter *measures* is not
a task score. It is the **routing**:

> The action expert's only window onto the state, the pixels, and the words is the
> **suffix→prefix cross-attention**. Cut that one block of the attention mask and the trained
> expert goes blind: its held-out velocity fit collapses toward the unconditional prior.

That collapse is byte-reproducible and seed-robust (a flow-MSE gap of +0.66 to +1.42 across seeds
0/1/2). It is the gated headline. The rollout is the higher bar, and we report, honestly, that
it floors.

## The data: ch1.7's pile, PushT half, now with time

```
[include-by-region: vla_shape.py#data]
```

We re-use ch1.7's recipe and re-contain it (no imports): replay the scripted PushT expert, keep
every other 64×64 frame with its state and action, and (the one new thing) the **episode
index** each frame came from. That index is what lets us build **action chunks**: for frame `i`,
the next `H=8` expert actions within *its* episode (ch1.3), padded and masked at the episode's
tail. Dropping ALOHA is deliberate: one embodiment means one action dimensionality, so there is
no padding and no masked multi-task loss to carry. The LOC goes into the two-tower instead. The
frozen random CNN featurizes every frame once; we keep the *same* encoder instance for training
and for live eval frames, so there is no train/eval vision mismatch to guard.

## The mask *is* the architecture

```
[include-by-region: vla_shape.py#model]
```

Two things live in this region, and both are the lesson.

**First, the block-attention mask.** The sequence is `[prefix | suffix]`: `P = 2 + 12` prefix
positions (vision, state, twelve instruction tokens) then `H = 8` suffix positions (the action
expert). `block_mask` fills four blocks:

- **prefix ↔ prefix**: full. The VLM fuses vision, state, and language bidirectionally (ch1.8's
  fusion, minus the CLS).
- **suffix → prefix**: full. Each action-query token reads the *whole* prefix. **This is the
  cross-attention**, and it is the block we will cut.
- **suffix ↔ suffix**: full. The `H` chunk steps coordinate among themselves.
- **prefix → suffix**: **blocked**. The prefix never reads the actions. That asymmetry is not
  cosmetic: it makes the prefix *action-independent*, so a deployment can compute it once and
  **KV-cache** it while the action expert denoises. This is exactly `pi0`'s `make_attn_mask`.

**Second, the expert as separate weights: the `pi0` "mixture."** This is the hardest idea in the
phase, so read it slowly. Picture two people in one meeting. They hear the *same* conversation
(one shared attention), but each keeps their own notebook and speaks their own vocabulary (their
own weights). That is the VLM and the action expert.

Look at `ExpertBlock`. There is *one* attention: a single scaled-dot-product softmax over the
joint sequence, under the mask. But the prefix tokens and the suffix tokens each own their
`Q/K/V`, their output projection, their MLP, and their norms. `per_tower` is the whole trick in
one line: run the prefix module on the first `P` positions, run the expert module on the rest,
then re-join. Two parameter sets, one attention op. The action expert is a **tower with its own
weights**, not a head bolted onto a pooled vector.

The suffix tokens themselves carry only a **noised action chunk + the flow time** (`action_in` +
`action_query` + the time embedding). They have no other access to the world. Hold onto that: it
is why the cut works. Each expert token's output goes through the **ch1.5 flow head**
(`vel_head`), so the model predicts the velocity of an `H`-step action chunk in one pass.

## Training is ordinary; the shape did the work

```
[include-by-region: vla_shape.py#train]
```

Nothing surprising here: it is ch1.5's conditional flow matching over a chunk, always trained
with the *full* mask. Sample a time `t`, put the chunk on its straight noise→data line, ask the
two-tower for the velocity, MSE over the valid (unpadded) chunk steps. The interesting part
already happened in the model: because the expert can *only* reach the state through
suffix→prefix, the optimizer is **forced** to route state through that block to fit the
state-dependent action. Remember that when you read the next region.

## The measurement: sever one block, watch the fit collapse

```
[include-by-region: vla_shape.py#eval]
```

We freeze the trained weights and ask one question two ways. `held_out_flow_mse(False)` measures
the held-out velocity fit under the **full** mask. `held_out_flow_mse(True)` measures the **same
weights** with suffix→prefix severed: the `--break cut_cross` mask, applied at inference. A fixed
`(t, noise)` pair makes it a clean paired comparison, and because it runs on *cached* features (no
rendering) the numbers are stable to the last ulp. As everywhere in this course, we gate the
**direction** (`gap > 0`), not an exact value (ch1.6). The result, every seed:

```
held-out flow-MSE:  full 1.48   cut-cross 2.89   gap +1.42   (seed 0, default config)
```

The gap is the headline, and here is why it lands where ch1.8's cut did not. In ch1.8,
`--break blind` zeroed the vision and *nothing moved*: PushT is state-solvable, so cutting the
pixels cost the policy nothing. **This cut bites because it severs the state itself.** In a
two-tower the action expert has no private line to the world. The state, the pixels, and the
words are *all* reachable only through suffix→prefix. Deny the expert that one block and it loses
its only path to the state; the best it can do is predict the *marginal* velocity, and the
held-out MSE jumps toward that unconditional prior.

The rollout is the honest counterweight. We roll the trained policy out on PushT (Wilson 95% CI,
ch1.6) and it **floors near zero for both masks** (0 to 1 of 12 across seeds, the Wilson interval
still covering zero): a from-scratch tiny two-tower on a frozen random vision backbone cannot
drive PushT any more than ch1.8's could. We report it plainly and gate nothing on it.

## What would make the vision load-bearing

```
[include-by-region: vla_shape.py#report]
```

The report writes `metrics.json` (headline: `flow_mse_gap`) and the toy's `vizdata.json`: the two
masks as heatmaps and a recorded rollout under each, so you can toggle the suffix→prefix cell and
watch the expert go blind.

One honest thread ties the arc together. The reason the rollout floors, and the reason the cut's
payload is *state* rather than pixels, is the **frozen random** encoder in the vision slot. You
already built the fix: ch5.2's **aligned** encoder puts scene geometry *into* the features. Drop
that into this prefix and the vision token starts carrying signal a controller can use, the
cross-arc payoff. And the whole two-tower (with a pretrained SigLIP tower, a real subword
tokenizer, and this exact prefix/suffix flow-expert) is `pi0` / `SmolVLA`, the read-the-real-thing
and the Scale Lab. You have now built its skeleton from scratch and *measured* the one thing that
makes it a VLA and not a pile of tokens: the routing.

## Break it

`--break cut_cross` severs suffix→prefix at inference, the exact ablation above, now as the
policy you roll out. Predict what happens to the held-out flow-MSE before you run it
(`ex1`), predict what happens to the *rollout* (`ex2`, the honest floor), and write the mask
yourself with the classic "fully bidirectional" bug and see the check catch it (`ex3`).
