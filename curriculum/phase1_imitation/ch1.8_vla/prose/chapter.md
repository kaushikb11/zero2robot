# 1.8: The Tiny VLA, Part II — Train It

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## Two halves, one policy

You have already built both halves of a vision-language-action policy. In **chapter 1.7**
you built the *data*: a multi-task pile of `(instruction_tokens, image_features, state) ->
action` examples, with a from-scratch word-level tokenizer and a frozen, random-init tiny
CNN standing in for a vision backbone. In **chapter 1.5** you built the *action head*: flow
matching — learn the velocity of the straight noise→data line, then sample an action by
integrating an ODE. This chapter fuses them into one language-conditioned policy and trains
it. Nothing here comes from a model zoo: the "VLM" is a token embedding plus a few lines of
from-scratch attention, and the head is the ch1.5 velocity field, now conditioned on what the
attention produced.

That word — *fuses* — is the whole chapter. A VLA has to take three very different things
(words, pixels-as-features, numbers) and turn them into **one** vector an action head can
condition on. Open `vla.py`. It has seven regions: **setup**, **data** (consume ch1.7's
`.npz`), **vision+language** (rebuild ch1.7's frozen encoder + tokenizer for eval),
**model** (the tiny VLM + flow head), **train**, **eval** (with ch1.6 error bars), and
**report**.

## The fusion backbone, from scratch

A transformer block is not magic. It is three `nn.Linear` projections (query, key, value),
one scaled dot-product, a softmax, and an output projection — plus a per-token MLP, wrapped
in pre-norm residuals. Here it is, the entire attention mechanism:

Three linear projections (query, key, value), one scaled dot-product, a softmax
over the sequence, and a weighted sum of values — the row that lets every token
read every other — and here it is in code:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

```
[include-by-region: vla.py#model]
```

Read `TinyVLA.fuse`. We lay the inputs out as **one sequence**:
`[CLS, vision, state, tok_0 … tok_15]`. The instruction tokens are embedded; the frozen
image feature and the state are each projected to the model width and become **one token
apiece**; a learned `CLS` token leads the sequence. A learned positional embedding is added,
padding tokens are masked out of attention, and a couple of self-attention blocks let every
element read every other. The `CLS` row that comes out has seen words, pixels, and numbers
at once — that is the fused conditioning vector. `velocity` is then exactly ch1.5's head:
concatenate the noised action, the sinusoidal flow-time embedding, and the fused vector, and
predict the velocity.

## Training on the multi-task pile — and the balancing trap

Training is ch1.5's loop with the flow loss conditioned on `fuse(...)` instead of a bare
state. But mixing two embodiments hides a trap. A PushT frame has **2** real action dims; an
ALOHA frame has **6**. If you sum the masked squared error over the batch and divide by the
number of valid dims — the obvious thing — every ALOHA frame contributes three times the
gradient of a PushT frame, and the 6-DOF task quietly takes over. Measured: with that naive
loss, PushT collapsed to **0.0** while ALOHA learned. The fix is one line — average each
example's error over *its* valid dims first, then average over examples, so a 2-D frame and
a 6-D frame weigh the same:

```
[include-by-region: vla.py#train]
```

(That trap is exercise **ex3** — a self-contained bug-hunt on the masked loss.)

## The frozen encoder is part of the policy

There is an honest subtlety in the **eval**. To roll out, the policy needs an image feature
for every live frame — so it needs the encoder. Chapter 1.7 saved the *features*, not the
encoder weights and not the frames. So `vla.py` **rebuilds ch1.7's frozen encoder** — same
class, same seed, same first random draw — and it comes out bit-identical (the envs and
experts touch no torch RNG, so the encoder is the first thing to draw after the seed). The
lesson is real: a frozen encoder is not a preprocessing step you can throw away — it is part
of the policy's input contract, and you carry it all the way to deployment.

```
[include-by-region: vla.py#vision_language]
```

## What it can do, and what it can't

Run it — `--seed 0 --device cpu`, about **2.2 minutes** on a CPU laptop:

```
[include-by-region: vla.py#eval]
```

The measured result, seed 0, default config:

- **PushT: trained 0.58**, Wilson 95% CI [0.32, 0.81], versus an **untrained 0.0**
  (CI [0.00, 0.24]); mean return **−48** against **−104**. Across seeds 0/1/2 the trained
  rate is 0.58 / 0.58 / 0.42 — above the untrained 0.0 every time. **The from-scratch VLA
  learns PushT.**
- **ALOHA: 0.0** on every seed. The bimanual handoff needs coordinated multi-step planning
  (the reason ch1.3's ACT chunked its actions); a tiny shared-capacity policy sampling one
  action at a time, through a random vision encoder, cannot do it.

So far this looks like a modest win. Now the uncomfortable question a good VLA engineer
always asks: **is the policy actually using its eyes?**

## Break It: `--break blind`

`--break blind` zeros the image feature at **both** train and eval. The policy still gets
the instruction and the state — it just never sees anything. Predict the PushT success
before you read on (that is exercise **ex1**).

Measured: PushT is **unchanged — 0.58 blind versus 0.58 sighted**, the mean return if
anything slightly *better*. Zeroing the camera did nothing, because PushT's answer is already
in the state vector (chapter 1.1 solved PushT from state alone), and a **random-init** vision
feature had nothing to add. The `fusion/cls_attention` bar in the `.rrd` tells the same
story: the read-out token barely weights the vision slot.

This is not a failure of the code — the code is a correct, honest VLA. It is a failure of the
**ingredients**. A from-scratch random encoder is a fixed projection of the pixels, not
perception. It preserves coarse layout (ch1.7), but nothing about it is aligned to objects or
to language, so a policy trained on it learns to ignore it whenever the state suffices — and
to fail whenever the state does not (ALOHA).

## Why you'd reach for SmolVLA (the Scale Lab)

That gap is the entire argument for **adapt-pretrained** over **from-scratch**. A real VLA —
SmolVLA, OpenVLA — replaces two of our from-scratch parts with pretrained ones: a vision
backbone (SigLIP/DINOv2) whose features *are* aligned to objects and language, and a subword
tokenizer from a pretrained LM. The action head can stay a flow-matching expert — the very
mechanism you built in 1.5. The Scale Lab fine-tunes SmolVLA on a consumer GPU and measures
the other half of this tradeoff. From-scratch taught you every moving part; adapt-pretrained
is what makes the vision and the language actually load-bearing.

## A note on what fit

An honest from-scratch VLA — tokenizer reuse, a rebuilt frozen encoder, a multi-head
attention backbone, a conditioned flow head, masked multi-task training, and a two-task
error-bar eval — fits in **one 449-line file** (hard cap 450, target 400). It is tight. Two
things made it fit: there is no 2-D toy here (unlike ch1.5), and there is no ONNX export (a
full VLA does not fit the stateless demo contract anyway — the browser panel is blocked on the
same flow-sampler contract v2 as the ch1.5 policy). If a future teaching-pass needs more room,
the honest cut is the ALOHA *eval* (keep multi-task training, evaluate only PushT) — not a
file split.

## What we cut

- **Pretrained everything.** Real VLAs use a pretrained vision backbone and a pretrained-LM
  subword tokenizer. Ours are frozen-random and 46-words-fixed (ch1.7). That is the Scale
  Lab, and the reason this policy ignores its vision.
- **Action chunking.** We sample a single action per step (ch1.5). ACT (ch1.3) and real VLAs
  predict a *chunk*; that is likely the missing ingredient for ALOHA.
- **A bigger backbone, more data, more tasks.** Scale knobs are visible flags (`--model_dim`,
  `--layers`, `--heads`, `--epochs`, `--episodes_per_task`).

## Read the real thing

The Scale Lab is not hypothetical — it is a real policy you can read line for line.
**`huggingface/lerobot`**, pinned here at `v0.4.4` (commit `8fff0fd`), ships **SmolVLA**
under `src/lerobot/policies/smolvla/`. It is the exact same three parts you just built —
a fusion backbone, a flow head, a tokenizer — with our from-scratch pieces swapped for
pretrained ones. Read it in three passes, against your `vla.py`.

**The fusion backbone.** Your `TinyVLA.fuse` (the **model** region) lays out
`[CLS, vision, state, tok_0..tok_15]` and runs a couple of from-scratch attention `Block`s;
the vision token is `vision_proj` of a **frozen random CNN** (the **vision_language**
region's `FrozenVisionEncoder`). SmolVLA's `VLAFlowMatching.embed_prefix()` in
`src/lerobot/policies/smolvla/modeling_smolvla.py` does the identical lay-out — image
tokens, language tokens, a `state_proj` — but the image tokens come from a **pretrained
SmolVLM2 backbone** (`SmolVLMWithExpertModel` in
`src/lerobot/policies/smolvla/smolvlm_with_expert.py` loads
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` via `AutoModelForImageTextToText`; the vision
tower is its SigLIP `vision_model` + `connector`). That is the one change that makes vision
load-bearing: features already aligned to objects and language, where ours were a fixed
random projection (`--break blind`).

**The action expert.** Your `flow_loss` regresses the straight-line velocity
`target_v = x0 - noise`, and `sample_action` integrates it with forward Euler in
`flow_steps` steps — the ch1.5 mechanism. SmolVLA's `VLAFlowMatching` builds
`x_t = t*noise + (1-t)*actions`, regresses `u_t = noise - actions` with an MSE on the
predicted velocity, and its `sample_actions()` / `denoise_step()` integrate noise→action
over `num_steps` (default `10`, `dt = -1/num_steps`) — your loop, on a bigger field. What
they add: the velocity head is a **half-width transformer "expert"** that cross-attends to
the frozen VLM prefix (`forward_cross_attn_layer`, `attention_mode`) and predicts a
**chunk** of 50 actions, not the single action our sampler draws.

**The tokenizer.** Your `encode_instruction` (the **vision_language** region) maps words
through ch1.7's fixed **46-word vocab**. SmolVLA's `processor_smolvla.py` runs a
`TokenizerProcessorStep(tokenizer_name=config.vlm_model_name, ...)` — the **pretrained
subword tokenizer** shipped with SmolVLM2 (`tokenizer_max_length=48`), plus a
`SmolVLANewLineProcessor` quirk-fix. Any instruction tokenizes; no OOV cliff at 46 words.

**Read these, in order.** `modeling_smolvla.py`'s `embed_prefix` and `sample_actions`
first — your `fuse` and `sample_action`, grown up. Then `smolvlm_with_expert.py`, to see
the pretrained backbone our `FrozenVisionEncoder` stands in for. That backbone is the
whole reason `--break blind` would *not* be a no-op for SmolVLA — and the whole reason to
adapt-pretrained when performance matters.

## Exercises

- **ex1 (predict-then-run):** does `--break blind` hurt PushT? (Measure that vision is not
  load-bearing.)
- **ex2 (predict-then-run):** which task does the tiny VLA learn — PushT or ALOHA?
- **ex3 (bug-hunt):** the multi-task masked loss — fix the sum-weighting that lets the 6-DOF
  embodiment dominate.
- **ex4 (code-completion):** write the scaled dot-product attention at the heart of the
  fusion backbone.
