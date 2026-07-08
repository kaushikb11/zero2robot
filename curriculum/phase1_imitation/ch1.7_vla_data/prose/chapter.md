# 1.7: Tokens Meet Torques — The Tiny VLA, Part I (the data)

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## What a VLA eats, and why this chapter is only the plate

A vision-language-action policy takes three things at once — an *instruction*
("push the T onto the target"), an *image* (what the camera sees now), and a *state*
(the numbers the robot reports) — and predicts an *action*. Chapter 1.8 builds and
trains that policy. This chapter builds the **data** it trains on, and nothing more.

That boundary is deliberate. The machinery that turns raw demos into
`(instruction_tokens, image_features, state) -> action` examples is where most of the
real-world VLA effort — and most of the silent failures — actually live. Get the data
wrong and no policy can recover; get it right and the policy in 1.8 is almost boring.
So this file has no training loop anywhere. It runs top to bottom and writes a dataset.

Open `vla_data.py`. Five regions: **setup**, **tasks** (assemble two tasks into one
pile), **language** (templates plus a from-scratch tokenizer), **vision** (a frozen
tiny CNN), and **pipeline** (stitch it all into examples, then probe it for leakage).

## Multi-task: two embodiments, one pile

```
[include-by-region: vla_data.py#tasks]
```

We use both environments you already have: the PushT pusher (a 2-D velocity action)
and the ALOHA cube handoff (a 6-D bimanual action). Each brings its own scripted
expert — the *same* experts `gen_demos.py` uses to write the LeRobot datasets — and we
replay them in-process so we can render each frame without a video-decode dependency.

Two honest frictions show up the moment you mix tasks. First, the **action spaces
differ**: 2 dims versus 6. We zero-pad PushT's action up to a shared width of 6 and
carry an `action_mask` marking which dims are real — the same move a real
multi-embodiment dataset makes. Second, the **episode lengths differ**: PushT ends
when the block lands on the target; the handoff is a longer pick → carry → handoff →
place. So at 12 episodes each you do *not* get equal frame counts — **309** PushT
frames and **170** ALOHA, 479 total. Every example carries a `task_id`, so the two
tasks share a tensor but are never confused.

## Language: templates and a tokenizer you can read

```
[include-by-region: vla_data.py#language]
```

Each task gets a small set of paraphrase **templates** ("push the t block onto the
target", "slide the tee onto the goal", …). We pick one per episode, deterministically
from the seed, and reuse its wording across that episode's frames — real demos are
annotated once, not per timestep. The paraphrases earn their place: they teach a policy
that the *task* is the invariant, not one exact string, so it does not overfit to a
single phrasing.

Words have to become integers, so we build the whole **tokenizer from scratch** — no
`transformers`, no BPE. It is word-level: the vocab is the sorted set of every word
that can appear (template words plus a few extras), prefixed with four special ids
(`<pad>`, `<unk>`, `<bos>`, `<eos>`). `encode` wraps a sentence as
`[BOS] words… [EOS]`, maps any unknown word to `<unk>`, and pads or truncates to a
fixed 16 tokens. The resulting vocab is **46** ids — four specials and 42 words — and
every content token is in-vocab (`oov_rate` 0.0) because the vocab is closed by
construction.

Be clear on the scale: a real VLA carries a 30,000+ subword tokenizer inherited from a
pretrained language model. Ours is 46 ids. What is identical is the *mechanism* —
string in, fixed-length id array out, specials and OOV handled — and that mechanism is
all chapter 1.8 needs from this side.

## Vision: a frozen tiny encoder (and what it is *not*)

```
[include-by-region: vla_data.py#vision]
```

A policy cannot condition on a raw 96×96×3 image cheaply, so we hand it a compact
**feature vector** per frame. `FrozenVisionEncoder` is a from-scratch conv stack —
three `Conv2d`/`ReLU` stages that halve the resolution each time, a global average
pool, and a linear head to `feature_dim` (64 by default). It is **random-init** and
**frozen**: its weights are never trained, here or in 1.8.

This is the chapter's most important honesty. A frozen random CNN is *not* perception
in any meaningful sense — it is a fixed nonlinear projection of the pixels. But it is
not useless either: convolution and pooling preserve *coarse spatial layout*, so the
64 numbers still encode roughly *where* the block, cube, and arms are, which is already
a more policy-friendly signal than the raw pixel grid. That is the entire pedagogical
point — you can build a working VLA data pipeline whose "perception" is a random
projection, and *watch* it flow through to features.

What a frozen random encoder does **not** give you is what a **pretrained** backbone
(DINOv2, SigLIP — the backbones OpenVLA and SmolVLA actually use) does: features
aligned to *objects* and to *language*, transferable across scenes and tasks. Closing
that gap — swapping this stand-in for a real, trained backbone — is a large part of why
chapter 1.8 exists. When you read this chapter's features, read them as "the signal a
policy will condition on," never as "good vision."

## The pipeline, and the leak

```
[include-by-region: vla_data.py#pipeline]
```

The pipeline stacks both tasks, tokenizes every instruction, encodes every frame
through the frozen CNN, and writes one dataset: a documented `.npz` (a real stack would
stream LeRobot v3 from the Hub) plus a `manifest.json` recording the schema, the tasks,
and the full vocab. The result is **479** examples, `feature_dim` **64**, `vocab_size`
**46**.

Then it runs one diagnostic that is the heart of the chapter — a **leakage probe**. The
question: *how much of the action can a linear model read out of the instruction words
alone?* We build a bag-of-words vector from each instruction's tokens, least-squares
fit it onto the action per task, and report the R² of that fit. If words tell you
nothing about the moment-to-moment action, R² ≈ 0; if the words encode the action, R²
approaches 1.

With the clean task-level templates, **R² = 0.006** — the instruction names the *task*,
and within a task the words barely move, so language explains essentially none of the
per-frame action. That is what you want: the action has to come from the image and
state, exactly where 1.8's policy will be forced to look.

## Break it

**`--break leak` — "a more descriptive instruction can't hurt."** This appends the
current move direction to every frame's instruction ("… moving northeast"). It reads
like a *better* annotation. It is a trap. The probe R² jumps from **0.006** to
**0.71** — the action is now linearly decodable from the words alone. A policy trained
on this data would minimize its loss by reading the answer off the instruction and
*ignoring its camera entirely*; drop the image at test time and it would look fine on
this distribution and fail the instant the instruction stops narrating the move.

The gap is seed-robust — the leak probe reads 0.71 across seeds 0, 1, and 2 — and the
lesson is general: **anything an annotator can see that correlates with the action can
leak into the instruction and hollow out the vision pathway.** The loss curve in 1.8
would *never* warn you; the model would look like it is learning. The only place to
catch it is here, in the data.

## What we cut

This is a real VLA data pipeline in shape, but every heavy part is a stand-in:

- **The vision encoder is frozen and random**, not a pretrained DINOv2/SigLIP. No
  object or language alignment — that is 1.8's upgrade and the single biggest gap.
- **The tokenizer is a 46-id fixed vocab**, not a 30k+ subword tokenizer from a
  pretrained LM.
- **Two tasks, not dozens.** RT-X / OpenVLA mix many datasets across many robots; the
  multi-task *mechanism* (pad + mask + task_id) is already the real one.
- **We emit inputs, we don't condition a policy on them.** Wiring
  `(tokens, features, state)` into an action head is chapter 1.8.

None of these silently fakes a number — each is a whole capability deferred so the
data pipeline stays readable end to end. The "read the real thing" segment walks a
production VLA data stack so you can see exactly what these paragraphs left out.

## Run it

```
python curriculum/phase1_imitation/ch1.7_vla_data/vla_data.py --seed 0 --device cpu
```

<!-- wall-clock table renders from wallclock.csv (ch1.7-vla-data: cpu-laptop, t4, l40s all measured) -->

| | value |
|---|---|
| examples (pusht / aloha) | 479 (309 / 170) |
| tokenizer vocab | 46 ids |
| frozen feature dim | 64 |
| action_from_language_r2 (clean) | 0.006 |
| action_from_language_r2 (`--break leak`) | 0.71 |

```
rerun outputs/ch1.7-vla-data/vla_data.rrd
```

## Read the real thing

The paired reading is **`huggingface/lerobot`**, pinned here at tag **`v0.4.4`** — the
library whose data format our `.npz` is a stripped-down stand-in for, and whose
**SmolVLA** policy is a real version of the pipeline you just built. (At this tag the
package moved under `src/`, so every path below starts `src/lerobot/`.) Read it in four
passes, one per stand-in.

**The multi-task mix and the language annotation.** Our `tasks` region gathers two
`TASKS` in-process, and our `language` region picks one paraphrase per episode from a
hand-written `templates` list. Production does not template — the instruction is
*data*. In `src/lerobot/datasets/lerobot_dataset.py`, `MultiLeRobotDataset`
concatenates many `LeRobotDataset`s into one pile (our two-task `np.concatenate`,
generalized to dozens of datasets and embodiments), and each frame's instruction is a
real string an annotator wrote — stored per episode and looked up in `__getitem__`
(`item["task"] = self.meta.tasks.iloc[task_idx].name`). What they add: the language is
*collected*, not generated from a fixed vocabulary, so it carries the phrasing
diversity our four paraphrases only gesture at.

**The tokenizer.** Our `Tokenizer` is a 46-id closed word vocab. SmolVLA's
`TokenizerProcessorStep` in `src/lerobot/policies/smolvla/processor_smolvla.py` loads a
*pretrained* subword tokenizer by name (`tokenizer_name=config.vlm_model_name`) and
pads to `tokenizer_max_length` (48). What they add: a 30k+ subword vocabulary inherited
from the language model, so an unseen word decomposes into known subwords instead of
collapsing to our single `<unk>`.

**The vision backbone.** Our `FrozenVisionEncoder` is a random-init conv stack. SmolVLA
loads a whole pretrained VLM in `src/lerobot/policies/smolvla/smolvlm_with_expert.py` —
`AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")` —
and encodes frames through its `vision_model` (a SigLIP encoder) via `embed_image`. What
they add: features aligned to *objects* and to *language*, transferable across scenes —
exactly the "good vision" our frozen projection is honest about not being.

**What conditions the action head.** Here we only *emit* `(tokens, features, state)`;
nothing consumes them. In `src/lerobot/policies/smolvla/modeling_smolvla.py`,
`embed_prefix` fuses image, language, and state embeddings into a prefix; `embed_suffix`
embeds the noisy action and the flow-matching timestep; and the action expert
(`smolvlm_with_expert.py`, `forward_cross_attn_layer`) attends from action tokens to
that prefix. That cross-attention is the wire chapter 1.8 solders — the flow-matching
head from chapter 1.5, conditioned at last on all three inputs.

**Read next**, in order: `MultiLeRobotDataset` in `lerobot_dataset.py`, then
`processor_smolvla.py` for the tokenizer step, then `modeling_smolvla.py`'s
`embed_prefix` / `embed_suffix` — the three lines of your pipeline, grown up.

## Exercises

Four, in `exercises/`. Two ask you to commit to a prediction before the run answers —
the leakage probe (clean vs `--break leak`), and the multi-task mix (do both tasks
appear, and in what proportion). One is a bug-hunt in the tokenizer (the leading `<bos>`
is dropped, so every id is shifted a slot). One has you implement the bag-of-words
count the leakage probe is built on.

## What's next

You now have the three inputs a VLA conditions on — tokenized instructions, frozen
image features, and state — stacked into one multi-task dataset, plus a probe that
tells you whether your language is honestly *describing* the task or secretly *dictating*
the action. Chapter 1.8 wires these inputs into a tiny VLA: it swaps the frozen random
CNN for a real backbone, the 46-id vocab for a real tokenizer, and puts the
flow-matching action head from chapter 1.5 on top — conditioned, at last, on all three.
