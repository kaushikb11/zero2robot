# 5.6: LoRA From Scratch — Adapt a Frozen Policy

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## The frozen checkpoint problem

Real robot policies ship as **frozen checkpoints**. pi0, GR00T, SmolVLA — gigabytes of weights
someone else pretrained on data you will never have. You want to teach one a new skill. You have
two honest options. **Full fine-tuning**: unfreeze every weight and train. It works, but you now
own a full-size copy of the model per skill, and — as we will measure — it quietly overwrites what
the model knew. Or you **freeze the whole thing and bolt on a tiny trainable adapter**. That second
option, in its most-used form, is **LoRA — Low-Rank Adaptation** — and it is about forty lines of
code. This chapter builds it from scratch and measures what it actually buys.

The subject is deliberately small and state-based — the ch3.8 routing task, no rendering, so the
whole run is bitwise-reproducible on a CPU in seconds. An instruction token selects a **skill**, and
each action dimension reads a distinct state coordinate the skill picks out. Open `lora.py`. Six
regions: **setup**, **model**, **lora**, **pretrain**, **sweep**, **report**.

## The base we will freeze

```
[include-by-region: lora.py#model]
```

`TinyPolicy` is a compact conditioned MLP: embed the skill token, concatenate it with the raw state,
and map that through a two-layer action head to an action. Nothing exotic — the point is what we do
to it, not what it is. We pretrain it on **three** skills and **hold one out**. The held-out skill's
token embedding never receives a gradient and the trunk never sees its coordinates, so the reloaded
base genuinely cannot do it. `head` — the action head's output projection — is the layer LoRA will
wrap.

> The base here is ~78K parameters, not ch3.8's ~7K. That is deliberate: LoRA's real-world hook is
> "you train ~1% of the weights," and that fraction is only meaningful if the base is wide enough
> that a rank-r adapter is genuinely ~1% of it. It is still a tiny, CPU-instant, bitwise MLP.

## LoRA, in forty lines

```
[include-by-region: lora.py#lora]
```

Here is the entire idea. Freeze a linear layer `y = W x`. Add a **thin trainable bypass** made of two
skinny matrices — `A` projects the input **down** to a tiny rank `r`, `B` projects it back **up**:

```
y = W x  +  (alpha / r) · B (A x)
```

`W` never trains. `A` is `(r, in)` and `B` is `(out, r)`, so together they hold `r·(in + out)` numbers
instead of `in·out` — for our head, a rank-4 adapter is `4·(256 + 6) = 1048` numbers against 77,958,
about **1.3%**. The scaling `alpha/r` is peft's convention, and `--break` will show you why the last
detail matters more than any of them: **`B` is initialized to zero.** So `B(A x) = 0` at step 0, and
the adapted layer is *bitwise identical* to the frozen one. You begin fine-tuning from the exact model
you paid to pretrain, and the adapter grows out of it rather than shoving it around.

## Pretrain, freeze, adapt three ways

```
[include-by-region: lora.py#pretrain]
```

We pretrain on skills {0, 1, 2}, `torch.save` the weights, and **reload them from disk** — the move
you make with any released checkpoint you did not train. Then, from that one frozen state, we build
three arms and adapt each to the held-out skill: **frozen** (nothing trains — the zero-shot baseline),
**full-FT** (every weight trains), and **LoRA** (W frozen, only the head's `A`/`B` train). Each arm
prints its trainable-vs-total parameter count, which is the whole economic argument in one line.

## The rank dial: the elbow

```
[include-by-region: lora.py#sweep]
```

Now turn the dial. We sweep the LoRA rank `r` and read the held-out fit at each setting. The measured
result, seed 0 (and the shape holds on every seed):

| rank            | % of weights trained | held-out fit (R²) |
| --------------- | -------------------- | ----------------- |
| 0 (frozen)      | 0.00%                | −0.44             |
| 1               | 0.34%                | −0.08             |
| 2               | 0.67%                | 0.25              |
| **4**           | **1.34%**            | **0.77**          |
| 8               | 2.69%                | 1.00              |
| 16              | 5.38%                | 1.00              |
| full fine-tune  | 100%                 | 1.00              |

Read the shape, not the decimals. The frozen base **cannot** do the held-out skill (a negative R² is
worse than guessing the mean). Then the fit **rises** with rank and **plateaus** onto the full-fine-tune
ceiling. A **rank-4 adapter, training ~1.3% of the weights, recovers ~84% of full fine-tuning's fit**;
by rank 8 the curve sits *on* the full-FT line. Past that knee, more trainable parameters buy almost
nothing. That knee is the **elbow**, and it kills a tempting intuition: *fewer trainable parameters
must mean a worse fit.* They don't. Most of what full fine-tuning learns here lives in a handful of
directions, and a low-rank bypass finds them.

## The honest twist: freezing W is not freezing behavior

There is a second intuition, just as tempting, and it is **wrong** — so we measure it rather than
assert it. Surely, the story goes, LoRA *can't forget* the skills the base already knew: its `W` is
frozen, so the old mapping is untouched. We watch an in-distribution skill (`task_A`, one the base was
pretrained on) while we adapt to the held-out one:

```
[include-by-region: lora.py#report]
```

`task_A`'s fit, seed 0: frozen base **+1.00** → LoRA **−1.88** → full-FT **−0.95**. It **collapses** —
under LoRA as badly as under full fine-tuning, here *worse*. Freezing `W` did not protect it, and the
reason is right there in the update rule. The adapter `(alpha/r)·B(A x)` is **added to every input**,
including `task_A`'s, and a single low-rank *linear* map cannot switch itself off for one skill and on
for another — that would take a multiplication it doesn't have. So the correction we trained for the
new skill bleeds straight onto the old one. LoRA's real win here is **parameter efficiency, not free
memory.** "Frozen weights" is not "frozen behavior." (Real systems *do* see LoRA forget less than full
fine-tuning — but that comes from scale and regularization, not from the frozen-W intuition, and in
this clean toy full fine-tuning, which can localize its update to the new skill's own weights, often
protects the old skill *better*.)

## Break it: why B starts at zero

You wrote the LoRA update in the exercise; the single most natural way to get it wrong is to
initialize `B` the way you'd init any other linear layer instead of zeroing it. Predict what that does,
then run `--break rand_init_B`.

The adapted policy is no longer identical to the frozen one at step 0 — the step-0 gap jumps from
**exactly 0.0** to **~0.17** (seed-robust). You have perturbed the model you paid to pretrain before a
single gradient step. Here adaptation still recovers the held-out fit, so the *outcome* doesn't
collapse — the lesson is subtler and more important: **zero-init `B` is what makes LoRA a no-op at
step 0**, so you begin adaptation *from the pretrained model itself* rather than from a randomly
jostled copy of it. That is the whole reason the real implementation zeroes it.

## Read the real thing

Everything here has a production form, and it is remarkably close. The read-the-real-thing segment
pairs this chapter with `huggingface/peft`, `src/peft/tuners/lora/layer.py`, `class Linear`. You will
recognize every piece: `update_layer` builds `lora_A = nn.Linear(in, r)` and `lora_B = nn.Linear(r, out)`
— our `A` and `B`; `reset_lora_parameters` kaiming-inits `lora_A` and **zeroes `lora_B`** — our exact
init and the thing `--break` violates; `scaling = lora_alpha / r` — our `alpha/r`; and the forward is
`result + lora_B(lora_A(x)) * scaling` — our `W x + (alpha/r)·B(A x)`. The same forty lines wrap the
attention and MLP projections of a billion-parameter VLA. The mechanism is size-invariant; that is the
point of building it small.

## What's next

You built the single most-used fine-tuning primitive in the modern stack from scratch, and measured it
honestly: a low-rank bypass recovers most of full fine-tuning's fit for ~1% of the trainable weights
(the elbow) — and does **not** hand you free memory (freezing `W` did not stop the old skill from being
forgotten). That is the real practitioner's calculus: LoRA is how you adapt a frozen policy cheaply,
and knowing exactly what it does and does not protect is how you use it without fooling yourself.
