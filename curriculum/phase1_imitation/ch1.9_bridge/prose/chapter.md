# 1.9: Graduation Bridge I: LeRobot for Real

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## Where you are

You have built ACT from scratch. In 1.3 you wrote the encoder that lets four
entity tokens attend to each other, the decoder whose K query tokens each emit one
action of the chunk, the temporal ensembler that averages overlapping predictions
into a smooth trajectory — ~380 lines you can read top to bottom. That was the
point of the whole exercise: you now *know* what ACT is, mechanically, with no
magic left in it.

This chapter cashes that in. The same policy already lives in `lerobot`, the
library the field actually ships and the one whose dataset format your `gen_demos`
and your 0.4 recorder have been writing all along. Instantiating ACT there is not
380 lines. It is about twenty:

```
[include-by-region: bridge.py#official]
```

That block is the entire policy: the encoder, the decoder, the chunking head, the
CVAE, and the temporal ensembler — every piece you built by hand — behind three
constructors. This is the first time in the course you are *encouraged* to import a
framework that hides the loop, because you are no longer learning the loop. You are
graduating past it.

## The reversal, and why you earned it

Everywhere else in Phase 1 the doctrine is strict: no `hydra`, no
`pytorch_lightning`, no wrappers that swallow the training step, because a hidden
loop you have never written is a loop you do not understand. That rule was for the
learner you were. The rule for the learner you are now is the opposite: reach for
the maintained, tested, community implementation — *and read its arguments like a
menu you understand*. `use_vae=True` turns the CVAE latent back on (1.3 cut it).
`temporal_ensemble_coeff=0.01` is the exact ensembling you hand-rolled. The ResNet
image backbone is in there too, dormant only because we feed state, not pixels. You
can see all of it because you have already written all of it.

## Loading your data, and the one honest wrinkle

The dataset loads unchanged — it is the same LeRobot-v3 format throughout Phase 1:

```
[include-by-region: bridge.py#data]
```

There is exactly one bridge to build, and it is a good bug to have met: ACT's
state-only schema requires the input key `observation.environment_state`, while our
recorder emits `observation.state`. The fix is not just a rename — it is a *re-type*.
lerobot classifies every feature by its `FeatureType`, and ACT accepts a camera-less
policy only when it finds a feature typed `ENV`. Rename the key but leave the type
`STATE` and the policy is rejected (or, in a looser setup, the environment-state
token is silently never built and you train on a truncated input). Exercise 3 is
exactly this bug, isolated.

## Training: the chunk you built by hand is now one declaration

In 1.3 you assembled the action chunk with an explicit loop over every episode,
gathering the next K expert actions per frame and a pad mask. The official stack
does it from a single declaration — `delta_timestamps` — and hands you the
`action_is_pad` mask for free:

```
[include-by-region: bridge.py#train]
```

Same bookkeeping, one line. The preprocessor normalizes with the dataset's own
stats (this is why the stats must travel with the data — pass the wrong stats and
the policy silently trains on mis-scaled inputs), and `policy.forward` returns the
L1 loss (it adds a KL term when `use_vae=True`). On CPU the loss falls from about
0.51 to about 0.08 over 150 epochs.

## Run it

```
python curriculum/phase1_imitation/ch1.9_bridge/bridge.py --seed 0
```

<!-- wall-clock table renders from wallclock.csv (cpu-laptop / t4 / l40s — all measured) -->

One command trains the official ACT, trains your from-scratch 1.3 ACT on the same
demos as a subprocess, evaluates both on the same held-out seeds, and dry-runs the
publish. It finishes in a few minutes on a CPU laptop — the free-tier floor, no GPU
required.

## The comparison, with error bars

Here is the question the chapter turns on: now that the official ACT is twenty
lines, is it *better* than your 380-line one? bridge.py answers it honestly — it
trains your 1.3 `act.py` as a subprocess on the *same* demos and evaluates both on
the *same* held-out seeds, and — because 1.6 taught you that a bare success rate is
a lie — it reports the official policy's rate with its Wilson interval:

```
[include-by-region: bridge.py#eval]
```

At the free-tier default (matched to 1.3: no CVAE, lr 1e-3, 150 epochs) the official
ACT lands at **0.55 [0.34, 0.74]** on 20 held-out seeds. That interval is wide *on
purpose* — 20 episodes is thin, exactly the 1.6 point. The from-scratch 1.3 ACT at
the matched budget lands at **0.95**, and its point estimate sits *above* the
official interval: this is a **real gap**, not sampling noise.

Why does the hand-rolled version win? Not because the framework is worse — it is the
*same algorithm*. It is because in 1.3 you split the ten observation numbers into four
*semantic* tokens (right arm, left arm, cube, target) and let them attend to each
other. That decomposition is a task-specific inductive bias. The general lerobot config
sees one undifferentiated state vector. Your domain knowledge is the difference — and
it is portable: you could inject the same tokenization into a custom lerobot input
processor. What the framework bought you, then, is not a higher number; it is a
maintained, tested implementation, a dataset-and-model Hub, a real-robot deploy path,
and a dozen other policies one import away. That trade — you *keep* your modeling
insight, you *gain* the ecosystem — is the honest shape of graduating to a real stack.
(Both rates also rise with the scale knobs; the honest way up is `--epochs` and
`--num_demos`, not a louder claim.)

## Break it: the 1.6 sin, one flag away

`--break train_dist` evaluates the official ACT on the *training* seeds instead of
the held-out ones — the starts it has already seen. In a brand-new framework this is
an easy accident: you grab whatever seed list is handy. The rate inflates to
**0.60 [0.39, 0.78]** against the held-out **0.55 [0.34, 0.74]**. Here the gap is
small — our held-out seeds are the *same distribution* as training (random resets,
different draws), so the policy has little to memorize — but it is in the wrong
direction to trust, and on a genuinely out-of-distribution held-out set (1.6's
LIBERO-style annulus, the block spawned farther than any demo) the same sin costs far
more. The invariant is not the magnitude; it is that you never report the train
number. Graduating to a real stack does not graduate you out of held-out evaluation.

## Publishing (when you have a token)

The last step of the real workflow is sharing: push the dataset and the trained
policy to the Hugging Face Hub so others can reproduce and build on them. That step
is **human-gated** — it needs a token and a network — so by default, and always in
CI, bridge.py *dry-runs* it: it prints the exact `push_to_hub` calls it would make
and serializes the policy locally instead.

```
[include-by-region: bridge.py#publish]
```

To publish for real: `huggingface-cli login`, unset `HF_HUB_OFFLINE`, and pass
`--publish`. Nothing in this repository ships a checkpoint or a dataset — those live
on the Hub; the repo carries only the code that regenerates them.

## What the framework hides (and what you can now see through)

`lerobot` is not magic, and after 1.3 it is not opaque to you either. The CVAE you
dropped is back; the ResNet path you never needed is waiting; the normalization is a
processor pipeline instead of two lines of your own. When one of those matters — when
you need to change the objective, add a camera, or debug why the loss will not fall —
you have the from-scratch mental model to open the file and read it. That is the real
deliverable of this bridge: not that you *can* call `ACTPolicy`, but that you can
call it and still know exactly what it is doing.

## Exercises

Four, in `exercises/`. Two are predict-then-run and turn on the comparison: before
you run, you predict whether the official ACT beats your from-scratch 1.3 ACT at the
matched budget (it does not — the entity-token bias wins, above the official's Wilson
interval), and whether evaluating on the training seeds inflates the rate over
held-out (it does — the 1.6 sin, on the official stack). Two are fast and
self-contained: a bug-hunt on the `observation.environment_state` re-type (the key is
not enough; the `FeatureType` must be `ENV`), and a code-completion that rebuilds
1.3's action chunk as a one-line `delta_timestamps` declaration.

## Read the real thing

The paired reading is `lerobot` itself, at the course-pinned `v0.4.4` — the code your
`# --- region: official ---` just called. Everything lives under `src/lerobot/` at
this tag; read each file against the piece of yours it replaces.

**The policy — `src/lerobot/policies/act/modeling_act.py`.** Your 1.3 `act.py` was the
~380 lines you can hold in your head. This is roughly 800, and every class in it is one
you already wrote: `ACTPolicy` wraps `ACT` (the CVAE-plus-transformer core),
`ACTTemporalEnsembler` is the ensembler behind the official region's
`temporal_ensemble_coeff=0.01`, and `ACTEncoder` / `ACTDecoder` are your encoder and
chunking decoder. The docstring credits Zhao et al.'s original ACT — same algorithm,
same paper. What the extra ~400 lines buy is not accuracy (your entity-token 1.3 ACT
beat this file's general config, above its Wilson interval); they buy the `torchvision`
ResNet backbone, gated on `config.image_features` and dormant only because you feed
state — the camera path you never needed, waiting.

**The wiring — `src/lerobot/policies/factory.py`.** `make_policy(cfg, ds_meta=...)` is
the dispatch you skipped by importing `ACTPolicy` directly, and `get_policy_class` is a
registry mapping `"act"`, `"diffusion"`, `"smolvla"`, `"pi0"`, and six more to their
classes — a dozen policies one string away. It calls `dataset_to_policy_features`
(defined in `src/lerobot/datasets/utils.py`, not the factory itself), the exact function
whose output you re-typed in `# --- region: data ---` to slot
`observation.environment_state` in.

**The chunk — `src/lerobot/datasets/lerobot_dataset.py`.** The `delta_timestamps` you
passed in `# --- region: train ---` is checked by `check_delta_timestamps` and turned
into `delta_indices`; then `_get_query_indices` builds the `{key}_is_pad` mask — your
hand-written episode-boundary padding loop from 1.3, now a few lines behind a
`tolerance_s`.

**The stats — `src/lerobot/processor/normalize_processor.py`.** `NormalizerProcessorStep`
(and its inverse `UnnormalizerProcessorStep`) is the pipeline step behind
`make_act_pre_post_processors`. Read its missing-stats branch: when a feature has no
stats, it *returns the input unchanged* — the silent mis-scaling this chapter warned you
about, sitting in production code. It supports `MEAN_STD`, `MIN_MAX`, `QUANTILES`, and
`QUANTILE10`.

Read next: open `modeling_act.py` beside your own `act.py`. You built each of these once;
the framework's gift is that you never have to again — and that you can still read every
line when it matters. The number stayed yours; the ecosystem is what you gained.

## What's next

That closes the imitation spine. Everything in Phase 1 learned from a fixed dataset
of expert demonstrations — behavior cloning, ACT, diffusion, flow, the evaluation
harness, the VLA — and this chapter delivered you onto the stack the field actually
uses to do it. Phase 2 changes the question. Instead of imitating demonstrations you
already have, you will *generate* the data by trial and error: chapter 2.1 opens with
PPO from a blank file, a policy that improves against a reward instead of a teacher.
The Hub publish path you dry-ran here returns for real at the 4.4 capstone, where you
ship a trained policy, its dataset, and a writeup — the graduation this bridge was
named for.
