# 3.8: Reading the Frontier

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## Where you are

This is the last chapter of Phase 3, and it asks you to do something you could not have
done when you started this course: open a frontier robot policy — the kind a well-funded
lab released last month — and *read* it. Not run it, not fine-tune it. Read it, the way
you read a paper whose every idea you have already implemented.

Because you have. The action head inside Physical Intelligence's pi0 is flow matching —
you built that in 1.5: learn the velocity of the straight noise→action line, then
integrate an ODE to sample. Its conditioning is a vision-language fusion — you built that
in 1.7 and 1.8: lay the instruction tokens, the image feature, and the state out as one
sequence and let attention fuse them into a single vector. NVIDIA's GR00T N1 splits its
policy into two systems running at two rates — a slow one that understands, a fast one
that acts — and the interface between them is exactly a fused representation conditioning
an action head. There is no piece of these systems you have not written by hand at small
scale.

So the skill of this chapter is not building. It is the four mechanical moves that
"reading a checkpoint" always comes down to, and they are the same whether the file on
disk is twenty kilobytes or fourteen gigabytes:

1. **Load** a saved `state_dict` into an architecture skeleton you hold.
2. **Inspect** the modules, their shapes, their parameter counts.
3. **Hook** a forward pass to capture a hidden layer's activations — you cannot edit a
   released model's `forward()`, so you attach a hook.
4. **Probe** those activations with a linear map, and ask: what has this layer actually
   learned to represent?

## The honest part: which checkpoint we probe, and why

We are going to run those four moves, end to end, on a real checkpoint file — but not
pi0's, and the reason is worth stating plainly. A real pi0 or GR00T checkpoint is several
gigabytes, and it is not something you can pull down and probe offline on the free tier;
the moment this chapter required a multi-gigabyte download it would stop being free-tier
and stop being reproducible on a plane. Nor do we probe the tiny VLA *you* trained in
1.8: that chapter saved its metrics but not its weights, so there is nothing on disk to
open. (An honest gap. If 1.8 later serializes a checkpoint, this probe reads it unchanged
— the code does not care whose checkpoint it is.)

So `probe.py` trains one tiny language-conditioned policy itself — once,
deterministically, in a few hundred steps — saves it, and then *reloads it from disk* as
if someone else had released it. It is a stand-in for a frontier VLA, and it is honest
about being one: we do not probe pi0, we probe a checkpoint built to behave like a
miniature of one. The task it learns is a miniature of the real thing too — an
instruction token selects *which* coordinate of the state should drive the action, a toy
"which skill" router. What transfers from this to pi0 is not the scale. It is the four
moves, which are size-invariant.

## Move 1 — Load: pour weights into a skeleton

```
[include-by-region: probe.py#checkpoint]
```

A checkpoint on disk is a `state_dict`: a flat dictionary of named tensors, no code. To
read it you supply the code — the `TinyPolicy` class, the architecture skeleton — and
pour the weights in with `load_state_dict(..., strict=True)`. `strict` earns its keep
here: it demands that every parameter name in the file match a slot in your skeleton,
which is exactly the check that tells you whether you have the architecture right. This is
the daily reality of reading a released model: find its model definition, instantiate it
empty, load. A missing or extra key means you have the architecture wrong — and the error
tells you which key.

## Move 2 — Inspect: read the table of contents

```
[include-by-region: probe.py#inspect]
```

Before you run a model you read its shape. `named_parameters()` is the table of contents
— every weight, where it lives, how big it is. On our stand-in it prints a handful of
modules totalling about 7,300 parameters. On a real VLA the very same enumeration is how
you find the vision tower, the language model, and the action expert, and how you see
where the parameters actually sit — almost always mostly in the pretrained backbone, a
sliver in the freshly-trained action head. You cannot read a paper's architecture figure
honestly until you have matched it against this table.

## Move 3 — Hook: capture what you cannot edit

```
[include-by-region: probe.py#forward]
```

Here is the constraint that shapes everything about reading someone else's model: you
cannot edit its `forward()`. You did not write it, and even with the source in front of
you the clean way to observe an internal activation is not to fork the code — it is to
*hook* it. `register_forward_hook` attaches a listener to a layer and captures its output
the next time the model runs, without touching a line of the model. We hook `policy.norm`,
the fused token every action is read from, and run one forward pass on a fresh batch the
checkpoint never trained on. That captured tensor is the object of the whole
investigation: whatever the policy "knows" about this input, it knows it here.

## Move 4 — Probe, and the trap in probing

```
[include-by-region: probe.py#probe]
```

A linear probe is the simplest honest question you can ask an activation: *is some known
factor linearly readable from it?* Fit a linear map from the frozen features to the factor
on one split of the data, score it on a held-out split. A high score means the layer
already represents that factor in a linearly accessible way; the model, in that sense,
"knows" it. We fit the map in closed form — ridge normal equations, no optimizer, no seed
to chase — so the read itself is deterministic and nothing about it can be blamed on
training luck.

We ask two questions of the fused token. First: which task did the instruction select?
The probe recovers it with about **1.0** accuracy. That looks like a triumph — until you
run the exact same probe on a **random-init** checkpoint of the same shape and it *also*
scores about **1.0**. Nothing was trained, and the probe still "found" the task. This is
the trap at the center of the chapter: the instruction token is a literal *input*, sitting
right there in the sequence, and a linear read of almost any projection of the input
recovers it. A probe that recovers an input has told you nothing about what the model
*learned*.

So we ask a second question, the one that means something: does the fused token encode the
*value* of the state coordinate the task routes to — a number the policy had to *compute*
by combining the instruction with the state? Here training finally shows up. The trained
checkpoint probes at **R² ≈ 0.90**; the random-init control at **≈ 0.16**. That gap is the
readout. It is the difference between a representation that merely carries an input and one
that carries a computed quantity — and it is the single most important habit to bring to
any paper that claims a layer of some real VLA "encodes" a concept: ask first whether the
probe merely recovered an input.

(The attention tells a small, interpretable story of its own: the fused token attends most
to the *task* token in the instruction — the fusion looks at the word that decides the
routing, which is exactly what you would hope to see.)

## Read the real thing

Now take the four moves to the real thing. This is a STUDY-tier segment — you read, you do
not run. The checkpoints are too large for the free tier, which is precisely why you built
the pieces yourself: the reading is the payoff, not the download. Three beats — our
stand-in, the production version, and what the labs add.

**What we built.** The `checkpoint` region of `probe.py` is the whole subject: one
`TinyPolicy` whose fused CLS — the output of `self.norm`, the single vector every action is
read from — is the boundary the `forward` region hooks. That fused-token → action-head seam
is where the probe sits, and it is the first place you would probe *anything*. Hold it in
mind; everything below is that seam, scaled.

**The production version — pi0.** Physical Intelligence's pi0 lives at
`src/openpi/models/pi0.py` (class `Pi0`, at the pinned commit `15a9616`). The same two
mechanisms, at scale. The conditioning is split across two methods: `embed_prefix()` lays
the image tokens and the language tokens out as a *prefix* — these are a **pretrained**
PaliGemma VLM's tokens, not our random-init embedding — and `embed_suffix()` lays the
state, the noisy action, and a sine-cosine timestep out as a *suffix*, the action expert's
inputs. Our fused CLS is that prefix→suffix attention boundary with a real VLM on one side.
The flow-matching head is two methods you can read against `flow.py` from 1.5 line for
line: `sample_actions()` integrates the inference ODE (`dt = -1/num_steps`, the update
`x_t + dt · v_t`, t=1 noise → t=0 action), and `compute_loss()` builds the training target
exactly as you did — `x_t = t·noise + (1−t)·action`, regress the velocity onto `noise −
action`.

**What they add, and why.** Two things we could not. First, a *real* pretrained VLM: pi0
runs two Gemma experts — a full PaliGemma for vision-language and a smaller action expert
(`action_expert_variant`) — so the conditioning tokens carry web-scale semantics, where
ours carried a four-word vocabulary. Second, in NVIDIA's GR00T N1.7
(`gr00t/model/gr00t_n1d7/gr00t_n1d7.py`, class `Gr00tN1d7`, tag `n1.7-release`) the
slow/fast split is made *structural*: `self.backbone` — System 2, a `Qwen3Backbone` VLM —
produces `backbone_outputs` that condition `self.action_head` — System 1, the flow head
running at control rate. The forward pass is literally `action_outputs =
self.action_head(backbone_outputs, action_inputs)`. That one line is the System-2 → System-1
interface, and it is our fused-token → head boundary with a name on it: if you had GR00T's
checkpoint, the first hook you would place is on `backbone_outputs`, for the same reason we
hooked `policy.norm`.

Honest: we probed a 20-KB stand-in, not pi0 — but the four moves and the one caveat (did
the probe merely recover an input?) are size-invariant, and the frontier is now something
you can *read*. **Read next**, in order: (1) `embed_prefix` / `embed_suffix` in
`src/openpi/models/pi0.py`, to see the conditioning sequence assembled; (2) `sample_actions`
and `compute_loss` in the same file, your 1.5 head at scale; (3) the
`self.action_head(backbone_outputs, ...)` line in `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`,
the dual-system seam you would probe first.

## Exercises

- **ex1 (predict-then-run):** of the two probes, which one separates the trained
  checkpoint from the random-init control? Run it and see why task-id does not.
- **ex2 (bug-hunt):** the leaking probe — score it on the held-out split instead of the
  rows it was fit on, and watch a signal-free layer's R² fall back to zero.
- **ex3 (code-completion):** write the closed-form ridge linear probe from scratch — bias
  column, normal equations on the train split, R² on the held-out split.
- **ex4 (reading):** in GR00T N1's dual system, what is the relationship between System 2
  and System 1 — and which tensor boundary would you hook to probe it?
