# 3.2: World Models II: Acting in Imagination

## The promise, and the trap inside it

In 3.1 you did something that should still feel slightly illicit: you *learned a
simulator*. An encoder, a GRU, a prior, a decoder: a little network that, given a
state and an action, dreams the next state, no `mj_step` inside. And you measured,
honestly, that it learned the **easy** half (the pusher, whose next position is
basically the integral of the velocity you commanded) and **not** the hard half (the
T-block's own pushed-around dynamics, where copy-last still beat it).

Here is the move this chapter is named for. If you have a simulator you can roll
forward cheaply and differentiably, you don't need the real environment to learn a
policy. You can learn one **inside the dream**. Freeze the world model, imagine
trajectories, score their reward, and push a gradient back into an actor. The real
PushT sim is never touched during policy learning. This is Dreamer, and it is one of
the most seductive ideas in the field: *learn to imagine, then learn to act in
imagination, and pay for almost no real experience.*

The trap is right there in the setup, and this chapter is built to make you measure
it. The reward for PushT depends on the **block**: get the T to the target. But your
world model got the block wrong. So the actor is about to optimize a reward computed
inside a dream whose hard half is a hallucination. It may learn to look like a
champion in imagination and then **fail completely in reality**. That gap, imagined
success minus real success, is the entire lesson. We are going to measure it and
refuse to look away from it.

## Reward without a reward head

```
[include-by-region: dreamer.py#reward]
```

Dreamer learns a *reward head*, a network that predicts reward from the latent,
because in a pixel game it has no other access to it. We don't need one: the PushT
reward is a known function of the state (drive the block's pose to the origin), so we
compute it analytically from the **decoded** observation. That is not a shortcut past
the hard part. It *is* the hard part, in plain sight. The reward reads `tee_xy` and
`tee_yaw` straight out of the frame the world model imagined. When the imagined block
is wrong, the imagined reward is wrong, and the actor optimizes the wrong thing. A
learned reward head would inherit exactly the same poison from exactly the same
place; making the reward analytic just puts the poison where you can see it.

## The actor and critic live on the model's state

```
[include-by-region: dreamer.py#agent]
```

The actor and critic both read the world model's state `[h, z]`: the deterministic
recurrent state and the latent. That choice matters for a reason that only pays off
at deployment: `[h, z]` is available *in a dream* (roll the prior) **and** *on the
real robot* (filter the posterior on real frames). Same features, both worlds, so the
same policy runs in each. The actor is a tanh-squashed Gaussian, which keeps actions
in the env's `[-1, 1]` and stays reparameterized, so the imagined return can
differentiate straight through the (frozen) dynamics into the policy. That is
Dreamer's analytic actor gradient; at this scale we need no REINFORCE estimator.

## Step 1: learn the simulator (unchanged from 3.1)

```
[include-by-region: dreamer.py#train_wm]
```

Nothing new here: it is 3.1's world model, trained on scripted-expert sequences with
the same reconstruction-plus-dynamics loss. We spend the fewest words on it because
you already built it; it is the substrate, not the point. When it is trained, we
**freeze** it.

## Step 2: learn a policy inside the frozen dream

```
[include-by-region: dreamer.py#train_actor]
```

This is the imagination loop. Sample real start states, filter them to `(h, z)`, then
roll the **frozen** model forward for `imag_horizon` steps under the actor's *own*
actions, decode each dreamed state, score its reward, and stop. That sequence of
imagined rewards and critic values becomes a **lambda-return** (Dreamer's
bias/variance-controlled target); the actor maximizes it by analytic gradient, and the
critic regresses onto it. Every gradient the actor ever sees comes from a dream. The
one honest caveat in the code: the analytic gradient through a frozen GRU is
high-variance, so the learning rate is deliberately small (`--actor_lr 1e-4`). A
larger one diverges on some seeds. On the default config this whole chapter runs in
**~0.22 min on a CPU laptop** (measured; T4 not yet measured). It is deliberately
tiny: the lesson is the mechanism and the gap, not a competitive agent.

And it works, in the sense that matters for the setup: the imagined return **rises**.
Across seeds the actor learns to drive the imagined block from its ~0.17 m spawn down
to about **0.01 m** from the target. In the dream, it parks the block. The policy is
learning. Hold that thought.

## The headline, measured: the imagination gap

```
[include-by-region: dreamer.py#eval]
```

Now deploy the *same* deterministic policy in two worlds and compare. **Imagined:**
warm the state on the start frame, roll the prior forward under the actor, score the
decoded reward (a pure dream). **Real:** filter the state on true frames, act, and
step the real PushT sim, scoring the same reward function on the true state. Measured
across seeds 0-2 at the default config:

- **Imagined return/step: ~-0.19.** Final imagined block-to-target distance **~0.01 m**.
  The dream is convinced the policy solved the task.
- **Real return/step: ~-0.39.** Final real block-to-target distance **~0.16 m**. The
  block barely moved from where it spawned.
- **The gap is +0.17 to +0.23 return/step, and it is positive on every seed.** Real
  task success rate: **0.00.**

Read that again. The policy did not fail to train. It trained *beautifully*, in
imagination. It learned to move the pusher and to sweep the imagined block to the
target. But the world it trained in is wrong exactly where the reward looks, so the
skill it acquired is skill at pleasing a hallucination. In the real sim the pusher
moves and the block does not follow the way the dream promised, and the T never
parks. **Imagination is only as good as your world model**, and yours, from 3.1,
learned the pusher and not the block. This is not a bug to fix in 3.2; it is the
measured, honest sequel to the finding you already made.

## A longer dream is not a better plan

The obvious hope is that a longer imagination horizon (letting the actor dream
further before it scores its plan) would hand it a better plan and rescue the real
performance. It does neither, and the investigation exercise has you measure exactly
why. Sweep `imag_horizon` from 5 to 30 and two numbers refuse to move: the dream parks
the block at **every** horizon (imagined final block-distance ~0.01–0.02 m whether the
actor imagines 5 steps or 30), and reality stays exactly as stuck (real block-distance
~0.16 m, real success 0.00 at every horizon). The delusion is **horizon-invariant**.
Rolling a wrong model further does not buy real skill and does not deepen the
delusion either: the model is simply wrong about the block wherever you read it, at
step 5 and at step 30 alike. The ceiling is the world model's block dynamics, and no
length of rollout raises it. That is why the thing that would actually close this gap
is a **better world model**, which, per 3.1, costs compute: the Scale Lab, and the
map's "why world models eat compute" thesis applied now to *acting*.

## Report and what carries forward

```
[include-by-region: dreamer.py#report]
```

The metrics record the imagined and real return, the gap, the block-to-target
distance in each world, and the real success rate (the numbers the exercises
reproduce). You have now built the full Dreamer skeleton by hand: a world model
(3.1) and an actor-critic that learns inside it (3.2). And you have measured the
honest boundary of the idea: that a policy is only ever as good as the dream it was
raised in. That boundary is not a discouragement; it is the map. When you read the
real DreamerV3 (see "read the real thing"), the machinery that closes this gap (a
far better world model, a learned reward and continue head, a stochastic latent with
KL balancing) is exactly the machinery you can now see the *need* for, because you
measured what its absence costs.

## What we cut

- **The learned reward and continue heads.** Real Dreamer predicts reward and
  episode-continuation from the latent; we compute reward analytically and imagine a
  fixed horizon. This is honest here (the reward is a known function of state) and it
  puts the world model's error in plain view, but it means we skip the machinery that
  lets Dreamer act on pixels with no state access.
- **The stochastic latent and KL balancing** (cut in 3.1, still cut here), so the
  imagined rollout has no calibrated uncertainty, which is one thing a real Dreamer
  would use to avoid over-trusting a shaky dream.
- **A world model good enough to act in.** The deepest cut, and the point. We act
  inside the deliberately-tiny 3.1 model, measure the gap, and name the fix (a better
  model = more compute = the Scale Lab) rather than paying for it free-tier.

## Read the real thing

`meta.yaml` pins `danijar/dreamerv3` at `e3f0224`. Three files hold the loop you
just hand-built. Read them against it.

**`imagine()`: rolling the prior under the actor.** Our imagination loop is
`dreamer.py#train_actor`: warm `(h, z)` from real frames, then step the frozen model
forward under `agent.act` for `imag_horizon` ticks, collecting each decoded reward.
Production does exactly this in `dreamerv3/rssm.py`, method `imagine`: it `nj.scan`s
the prior transition (`self._prior(deter)`, no posterior, no observation) forward
under a `policy` callable, the same "roll the prior on the actor's own actions" move.
The wrapper is `dreamerv3/agent.py`'s `imag_loss`, which calls
`self.dyn.imagine(starts, policyfn, H, ...)` with `policyfn` sampling the actor. What
they add: the rollout is a jitted `scan` over batched latents starting from *every*
real step in a replay buffer, refilled as new experience arrives. Where ours dreams
off a handful of warmed starts, theirs dreams off thousands, continuously.

**Lambda-returns, and where the actor gradient comes from.** Our `lambda_return` in
`dreamer.py#train_actor` is the textbook recursion `G_t = r_t + γ((1-λ)V_{t+1} +
λG_{t+1})`, and our actor maximizes it by an *analytic* gradient: the reparameterized
tanh-Gaussian differentiates straight through the frozen GRU. Production keeps the
lambda-return (`dreamerv3/agent.py`, `lambda_return`, the same recursion with a
continue-masked discount) but **does not** use that analytic dynamics-gradient. Its
`imagine` stop-gradients the state into the policy (`policy(sg(carry))`), and
`imag_loss` trains the actor by REINFORCE: `policy_loss = -(logpi * sg(adv_normed) +
actent * ents)`, advantages normalized by a running scale. This is the one place our
chapter and DreamerV3 genuinely part ways: DreamerV1 backpropped through the dynamics
as we do; v3 switched to the score-function estimator because it survives discrete
actions and long unrolls where the pathwise gradient explodes, the very
high-variance we bought down with `actor_lr 1e-4`. Same target, different way of
pushing it into the policy.

**The learned reward head vs our analytic reward.** Our `dreamer.py#reward` reads
`tee_xy`/`tee_yaw` straight out of the decoded frame, no reward head, because PushT's
reward is a known function of state. Production can't do that: on pixels it has no
state, so `dreamerv3/agent.py` learns one, `self.rew = embodied.jax.MLPHead(...)`,
trained by `losses['rew'] = self.rew(inp, 2).loss(obs['reward'])`, alongside a
*continue* head `self.con` predicting episode termination (we imagine a fixed horizon
instead). But swapping analytic for learned changes nothing about the lesson you
measured: both read reward off the world model's *imagined* state, so a model wrong
about the block poisons either one. The imagination gap is a property of the dream,
not of where the reward comes from, which is exactly why rolling further (the
horizon investigation) never closed it.

**Read next:** open `dreamerv3/agent.py`, find `imag_loss`, and read down to the
`lambda_return` call right inside it. That method is the entire actor-critic-in-imagination
loop you built by hand. Read it knowing the return is computed the same way you
computed it, and only the actor's gradient path differs.
