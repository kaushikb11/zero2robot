# 3.1 — World Models I: Learning the Simulator

<!-- objectives: rendered from meta.yaml, do not duplicate here -->
<!-- wall-clock table renders from wallclock.csv -->

## The one function you never questioned

For thirty chapters there was a line you never looked twice at: `obs, reward, done, info = env.step(action)`. You hand MuJoCo an action, it hands you back the next state. `mj_step` is the oracle at the center of everything — every rollout, every training loop, every eval. You trusted it completely, because it is real physics and it is not yours to learn.

This chapter takes it away. We are going to *learn* that function from data — build a model that, given where things are and what you did, dreams where things go next. No `mj_step` inside. Just a neural network that watched the simulator run and learned to imitate its stepping. This is a **world model**, and it is the idea underneath Dreamer, underneath MuZero, underneath the whole "learn to imagine, then plan in imagination" line of work you'll build on in 3.2.

The payoff is not that the learned simulator is *better* than MuJoCo — it never will be. The payoff is that it is *differentiable*, *fast to roll out*, and *yours* — and in 3.2 you'll train a policy entirely inside it, never touching the real env during learning.

## Reconstruction is not prediction

Here is the distinction the entire field turns on, and the one this chapter is built to make you feel.

Give a model the current observation and ask it to spit the same observation back out. That is **reconstruction**. It is an autoencoder. It is easy — squeeze the state through a bottleneck and expand it again, and with enough capacity the error goes to zero. Reconstruction tells you the model can *represent* a state. It tells you nothing about whether it understands *dynamics*.

Now hide the future from the model. Give it a few frames to orient, then feed it only the *actions* and ask it to roll forward — to say what the state will be in one step, two steps, ten steps, without ever seeing those frames. That is **prediction**, and it is a completely different question. To predict, the model has to have learned how a state *flows* into the next one under an action. It has to have learned the simulator.

The honest test of prediction is a baseline so dumb it is almost insulting: **copy-last**. Just assume nothing moves — predict that every future state equals the last one you saw. On slow dynamics this is shockingly strong. At 10 Hz control the pusher moves maybe a centimetre per step; for one step, "nothing changed" is nearly right. A world model that cannot beat copy-last has not learned anything a constant couldn't tell you.

So that is the bar. Not "does reconstruction work" (it will). The bar is: **does k-step prediction beat copy-last?**

## The architecture: prior and posterior

```
[include-by-region: wm.py#model]
```

Four learned pieces, and one idea holding them together:

- an **encoder** turns an observation into a feature vector;
- a **GRU** carries a *deterministic* recurrent state `h` — the running summary of everything that has happened;
- a **decoder** turns `(h, latent)` back into an observation;
- and two heads that both produce the latent `z`, differing only in what they are allowed to look at.

That last part is the whole design. The **posterior** forms the latent from `(h, encoded observation)` — it *has seen* the current frame, so it is the accurate, corrected belief. The **prior** forms a latent from `h` *alone* — it must *predict* what the posterior will be, before the observation arrives. This is the [RSSM](rtrt) prior/posterior split, stripped to its skeleton (we keep the deterministic latent and drop the stochastic-KL machinery — see "What we cut").

Training ties them together. The decoder learns to reconstruct from the *posterior* latent; the prior learns to *match* the posterior it can't see. Minimize both and you get a model that reconstructs accurately when it has an observation and predicts accurately when it doesn't — because the prior has been trained to stand in for the posterior.

## Data: sequences, not steps

```
[include-by-region: wm.py#data]
```

A world model cannot learn dynamics from a pile of independent `(state, action)` pairs — it has to see how a state becomes the *next* state, so the unit of data is a rollout. We reuse the PushT env and its scripted expert (the same simulator we are learning to imitate), add exploration noise for coverage, and crop each episode to a fixed-length sequence of states and the actions between them. States are standardized so every coordinate carries comparable weight in the loss; the fixed target-pose dimensions get unit scale rather than a divide-by-zero.

## Training: one loss, two terms

```
[include-by-region: wm.py#train]
```

The objective is reconstruction loss plus a **dynamics** loss (the prior matching the detached posterior). The dynamics term is what turns an autoencoder into a simulator — delete it and the prior is never trained, so prediction is noise no matter how good reconstruction looks. On the default config this trains in about **20 seconds on a CPU laptop**. It is deliberately tiny — the lesson is the mechanism, not a competitive Dreamer.

## The headline, measured

```
[include-by-region: wm.py#eval]
```

Warm the state on a few observed frames, then roll the *prior* forward on actions alone and score each horizon `k` against copy-last. Measured across seeds 0-2 at the default config:

- **Reconstruction** (posterior, sees the frame): val MSE **~0.035** — the model recreates what it sees. Easy half, done.
- **Prediction vs copy-last**: at **k=1 copy-last wins** — almost nothing moved in a tenth of a second. The world model **overtakes it by k=2-3** (the *crossover*), and the gap **widens** with the horizon. Averaged over the 12-step horizon, the world model's error is **~2.3x lower** than copy-last (range 2.18-2.45x across seeds).

That crossover is the entire chapter. Copy-last decays because the world keeps moving and "nothing changed" gets more wrong every step. The world model holds low because it *knows what the action does* and integrates it forward. The moment its curve dips under copy-last's is the moment you can say, with a measurement behind it, that it learned to step the simulator.

But read the eval's per-dim line before you celebrate, because it keeps you honest. Split the observation into the two things that move — the **pusher** and the **object** — and the aggregate win comes apart. On the pusher dimensions the model beats copy-last by roughly **8x**, which sounds like triumph until you notice why: the pusher's next position is a near-trivial integral of the velocity you just commanded. On the **object** dimensions — the T-block's own pushed-around pose, the contact dynamics PushT is actually about — **copy-last still wins**, on every seed. The block barely moves per step, so "nothing changed" is nearly right, and the model's prediction only adds noise.

So state the headline honestly: this tiny model learned the *easy* half of the simulator — the kinematics of the thing you drive directly — and not yet the *hard* half, the dynamics of the thing you drive it into. The aggregate crossover is real and worth measuring, but it is not "it learned the physics." That gap between the halves is not a failure to hide; it is the honest reason chapter 3.2 and the pixel Scale Lab exist.

A note on the `--break peek` mode: it lets the prediction rollout illegally re-filter the *posterior* on each true frame. "Prediction" error collapses toward the reconstruction floor and the model looks fantastic — because it is secretly reconstructing, not predicting. It is the single most common way to fool yourself with a world model, and it is worth running once to see how good cheating looks.

## Why state, not pixels (the honest part)

The curriculum map says "PushT pixels" for this chapter, and we ran a feasibility spike to hold that promise honestly. The spike measured three things:

1. **Rendering pixels is cheap.** PushT renders 32x32 grayscale frames offscreen at ~0.7 ms each on CPU — a pixel dataset is entirely free-tier.
2. **Reconstruction on pixels is beautiful.** A conv encoder/deconv decoder reconstructs frames to ~5e-4 MSE. The frames look right.
3. **Prediction on pixels cannot beat copy-last at free-tier scale.** And this is the finding. A T and a pusher are a handful of pixels on a static background, so copy-last's pixel-MSE is tiny — it is *pixel-perfect* on the ~95% of the frame that never changes. A small model carries a reconstruction-blur floor across the *whole* frame, and that floor never dips below copy-last's motion error. We measured no crossover through a 22-step horizon. Beating copy-last on pixels needs a sharper (larger) decoder — more compute — which is exactly the point.

That is the honest reason world models "eat compute," and it is the map's own framing for this chapter: *genuinely instructive, deliberately small*. So the free-tier chapter teaches the **mechanism** on the low-dim state, where the same recipe wins cleanly and runs in seconds; the pixel world model is the **Scale Lab** and the substrate that 3.2 grows into. The physics of the lesson — encoder, prior/posterior, decoder, reconstruction-vs-prediction — is identical either way. Only the observation changed, and with it the honest reach of a free-tier model.

## Report and what carries forward

```
[include-by-region: wm.py#report]
```

The metrics record the reconstruction floor, the prediction-vs-copy-last means, the ratio, and the crossover horizon — the numbers the exercises reproduce. In **3.2** you'll stop *evaluating* the prior rollout and start *learning inside it*: a policy trained on imagined trajectories, the real env untouched during learning. The `imagine()` rollout and the prior/posterior split you built here are exactly what that needs.

## What we cut

- **The stochastic latent.** Real RSSMs make `z` a distribution (Gaussian or categorical) and train the prior/posterior gap as a KL divergence; we use a single deterministic vector and an MSE between prior and posterior — the same *shape* of idea, none of the sampling. This costs prediction sharpness and rules out proper uncertainty, and it is the first thing to add when you read the real thing.
- **Pixels.** The observation is the 10-D state, not an image (see above). The encoder/decoder are MLPs, not conv stacks.
- **KL balancing, free bits, reward/continue heads, symlog** — all the DreamerV3 stabilizers. We predict only the observation, not reward or episode-continuation, because 3.1 is about learning the *simulator*; acting comes in 3.2.

## Read the real thing

Everything we cut is sitting, fully assembled, in one file of Danijar Hafner's DreamerV3 — `dreamerv3/rssm.py`, pinned at commit `e3f02248`. It is the production version of the exact skeleton you just built. Reading it is not "here is the better code"; it is watching each simplification we made get its real machinery bolted back on.

Start with the prior/posterior split — our `prior()` and `posterior()` in `wm.py#model`. One honesty note first: this chapter's `meta.yaml` names them `obs_step`/`img_step`, the historical DreamerV2 names, but at this pinned commit those methods no longer exist under those names. The posterior path is now `RSSM.observe` (and its inner `_observe`); the prior path is `RSSM.imagine`, which calls `_prior(feat)`; the GRU lives in `_core`. Same two questions — correct the latent from an observation, or predict it from the recurrent state alone — different names. Map our `posterior()` onto `_observe`, our `prior()` onto `_prior`, and our `imagine()` onto `RSSM.imagine`, and the architectures line up beat for beat.

Now the latent itself. Our `z` is a single deterministic vector — one `nn.Linear` output, tied to the posterior by MSE. The real `RSSM` keeps *both* halves of the state: a deterministic recurrent part (`deter: int = 4096`, the `_core` GRU state, our `h`) *and* a stochastic part that is categorical — `stoch: int = 32`, `classes: int = 32`, so 32 discrete variables of 32 classes each, sampled with a `unimix: float = 0.01` uniform floor. That stochastic latent is precisely what lets the model represent a *distribution* over next states instead of one averaged guess — the same multi-modality problem that made BC blur in `ch1.1`. We kept only the deterministic half, which is exactly why our object dynamics collapsed into noise: a T-block's contact outcome is multi-modal, and a deterministic latent can only average the modes.

Finally the loss. Ours is `dyn_loss = F.mse_loss(zhat, z.detach())` in `WorldModel.observe` — a point-wise MSE with a stop-gradient. In `RSSM.loss` it becomes a KL between *distributions*, split two ways: `dyn = self._dist(sg(post)).kl(self._dist(prior))` and `rep = self._dist(post).kl(self._dist(sg(prior)))`, each floored by `free_nats: float = 1.0`. That is KL balancing (the `sg` stop-gradient on opposite sides sets how fast prior and posterior move toward each other) and free bits (the floor stops the KL collapsing to zero). Our `.detach()` on the posterior is the stripped cousin of `dyn`; we simply dropped `rep` and the entire distributional layer.

Read these next, in order: (1) `dreamerv3/rssm.py` `observe`/`_observe` and `imagine`/`_prior` — trace them against our `posterior`/`prior`; (2) `RSSM.loss` — the `dyn`/`rep`/`free_nats` block, the KL our `dyn_loss` approximates; (3) `_core`, `_prior`, `_dist` — the GRU and the categorical latent we cut. Then, when you reach 3.2, `dreamerv3/agent.py` — the actor-critic that learns *inside* this model by rolling `imagine` forward.
