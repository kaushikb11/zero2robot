# 4.1: Offline RL Primer: Beat the Data With Its Own Reward

## The setting: you can't collect more

Every learning method so far got to *act*. Behavior cloning (1.1) had a dataset,
but PPO and SAC (2.1, 2.2) learned by stepping the env millions of times, and
even the corrections chapter (4.2) closes the loop: you fly the policy, catch it
failing, and record the fix. Now take that away. **You have a fixed pile of
logged data (some good demonstrations, some clumsy or failed attempts) and you
cannot collect one more transition while you learn.** No rollouts, no
exploration, no on-policy correction. Just the log.

This is not a contrived constraint; it is the *normal* one. Robot data is
expensive and sometimes dangerous to collect, and the data you already have
(teleop sessions, logged corrections, a fleet's worth of mixed-quality trajectories)
is the data you get. The question this chapter answers by *measuring*: given a
fixed, mixed-quality log, what is the best policy you can squeeze out of it?

## Why behavior cloning leaves reward on the table

BC has an answer: clone the data. Regress a network from observation to the action
the log took, with plain MSE (1.1). But look at what MSE does to a **mixed**
dataset. In a given state it minimizes squared error to *every* logged action at
once, so it converges to their **average**. Average a clean expert action with a
random flail and you get a watered-down action that is neither. BC's ceiling is
therefore the dataset's *average* quality, and it has no way to do better,
because **it never looks at the reward.** It cannot tell a good action from a bad
one; it only knows they were both in the log.

We build the dataset to make this bite. It is the canonical **"expert + random"**
mix: a `--expert_frac` slice is the scripted IK reacher from 2.2 (near-optimal),
and the rest is a uniform-random policy (flails, rarely reaches). Measured
behavior returns: expert **~-2.3**, random **~-16.0**, genuinely mixed quality.
The random half is not only junk to average over; it also *covers* the
action space, which the offline learner will need.

```
[include-by-region: offline.py#data]
```

## What offline RL does instead: use the reward

Offline RL's move is to *not* treat all logged actions equally. It uses the
reward to prefer the actions that were **above average**, and it does this in two
steps you already know how to build.

**Step one: a critic.** Fit a Q-function on the fixed data, the reward-to-go of
taking action `a` in state `s`. This is exactly 2.2's twin-Q critic: two
Q-networks, clipped double-Q (bootstrap from the *min*), and slow-moving target
networks. The only change from SAC is that there is no replay buffer being filled
by a live policy: the "buffer" is the frozen dataset, sampled forever.

**Step two: advantage-weighted policy extraction.** Now extract a policy. Here is
the whole trick, and it is one line different from BC. BC regresses toward the
logged action with a uniform weight. **AWAC** regresses toward the same logged
action, but weights each sample by

    weight = exp( A(s, a) / beta ),   A(s, a) = Q(s, a) - Q(s, pi(s))

the **advantage** of the logged action over what the current policy would do. An
action the reward says beat the policy gets `weight > 1` and is pulled toward; a
bad action gets a weight near zero and is ignored. That is the entire difference
between "clone the data" and "extract the best policy the data supports." `beta`
is the temperature: small sharpens the weighting toward pure filtering, large
melts it back into BC.

```
[include-by-region: offline.py#model]
```

The two learners share the same network and the same regression, and differ only
in that weight, so read their losses side by side:

AWAC is BC's regression toward the logged action, but each sample scaled by the
exponentiated advantage of that action over what the policy would do, and here it
is in code:

$$
\mathcal{L}_{\mathrm{AWAC}}(\theta) = \mathbb{E}_{(s,a)\sim\mathcal{D}}\!\left[\, \exp\!\Big(\tfrac{1}{\beta}\,A(s,a)\Big)\; \big\lVert \pi_\theta(s) - a \big\rVert^2 \,\right],
\qquad A(s,a) = Q(s,a) - Q\big(s, \pi(s)\big)
$$

```
[include-by-region: offline.py#train]
```

## Why naive offline RL is not enough (and why the constraint exists)

A reasonable objection: if we have a Q-function, why the weighting ceremony: why
not just train the policy to **maximize Q**, the way DDPG or SAC do? Run `--naive`
and find out. Maximizing Q with no anchor to the data lets the policy walk to
**out-of-distribution actions**, actions the log never contains, where the
Q-function is pure extrapolation. Offline, there is no env step to check that
fantasy, so the critic **overestimates** those actions and the policy happily
chases the overestimate.

Here is the honest, measured subtlety, and it is the best part of the lesson.
**How badly this bites depends on how narrow your data is.** On the broad
expert+random mix, the random half *covers* the action space, the critic stays
calibrated, and even naive maximize-Q survives. But on **narrow** data
(expert-only, `--naive --expert_frac 1.0`, which is the shape of real demo and
correction logs) the naive critic inflates: measured mean `|Q|` **~7x** the
AWAC critic's while its eval collapses toward random. AWAC's advantage weighting
is the fix that **does not depend on coverage**: it only ever asks the critic
about actions the data actually contains, so there is nothing to extrapolate
into. That is why offline RL is its own algorithm and not just "Q-learning on a
log."

## The result: measured, honest

Train both on the same fixed dataset, then grade them with 1.6's error bars, a
Wilson interval on each success rate and a difference CI that decides whether the
gap is real. On the default free-tier config (mixed dataset, held out over 100
rollouts), across seeds 0, 1, 2:

| learner | success rate | mean final dist |
|---|---|---|
| behavior cloning | **0.03 / 0.07 / 0.05** | 0.154 / 0.147 / 0.144 m |
| offline RL (AWAC) | **0.27 / 0.20 / 0.21** | 0.109 / 0.100 / 0.089 m |

The difference CI (offline − BC) **excludes zero on every seed**: the win is
significant, not noise, and it holds across seeds. Offline RL beat BC on the same
data by using the reward BC ignored.

Be honest about the size of it. Offline RL reaches ~0.09–0.11 m (random leaves
~0.176 m), still well short of the scripted expert's ~0.0001 m. This is a *primer*
on a tiny free-tier budget, not a maxed-out policy. And BC does not improve much
even as you clean the data up (exercise 2 sweeps 15% → 60% expert): it stays
~0.03–0.05 success as long as the log still carries a meaningful junk fraction,
because averaging *any* junk into the regression target corrupts it. **The
mechanism is the lesson, not the number**: reward-aware extraction beats cloning,
and the advantage constraint is what makes it safe to do offline.

Now the boundary that locates the whole mechanism, and it is the most important
honesty in the chapter. Take the junk away entirely (`--expert_frac 1.0`, a
*clean* expert log and nothing else) and AWAC's edge over BC evaporates. Measured
across seeds 0, 1, 2: BC jumps to ~0.2 success (there is no bad action left to
average in, so cloning the expert *is* the right move) and actually holds a
slightly *lower* final distance than AWAC (~0.053 m vs ~0.076 m); the success-rate
difference CI is marginal and not seed-robust: a tie on seed 0, a hair above zero
on 1 and 2. AWAC lost nothing; it simply had nothing left to reweight. That is the
real headline, stated carefully: **offline RL beats cloning when the log carries
suboptimal actions worth down-weighting**, the mixed, correction-shaped data of
the real world. On a pristine expert log, cloning already does the reweighting for
free, and the honest reading is a wash. This is why the default dataset is a mix,
and why the win to trust is the seed-robust one *on that mix*, not a blanket claim
that "AWAC beats BC."

```
[include-by-region: offline.py#report]
```

## Why this is the prior 4.3 builds on

The post-training arc's endpoint (4.3, HIL-SERL) fine-tunes an RL policy starting
from your **correction data as a prior**, and sample efficiency is the whole
point, so it cannot afford to start from scratch. That prior is precisely what
this chapter builds: a policy *extracted* from a fixed log of mixed-quality
behavior, using the reward, without the extrapolation error that sinks naive
offline RL. Correction data is **narrow**, exactly the regime where the naive
approach fails and the advantage constraint earns its keep. When 4.3 says
"offline-primed," this is the priming.

## Running it

    python curriculum/phase4_capstone/ch4_offline_primer/offline.py --seed 0
    # Break it (narrow data): ... --seed 0 --naive --expert_frac 1.0
    # Investigate:            ... --seed 0 --expert_frac 0.6

<!-- wall-clock renders from wallclock.csv (ch4-offline-primer) -->

Determinism: `--seed` fixes the dataset, both trainings, and every eval reset, so
two CPU runs produce byte-identical `metrics.json`. CPU is the one configuration
this book promises is bitwise-reproducible; on a GPU or an Apple mps backend the
same seed reproduces the *result* statistically, not bitwise. The numbers above
are `--device cpu`, and the exercises force it.

## Read the real thing

The AWAC you just built is the honest core of a real algorithm, not a toy of one.
Two readings show you where the production implementations go from here.

The **AWAC paper** (Nair et al. 2020, *Accelerating Online Reinforcement Learning
with Offline Datasets*) is where the advantage weight comes from. The
`exp(A / beta)` you wrote is not a heuristic someone tuned; it falls out as the
closed-form solution to a KL-constrained policy-improvement step, which is why the
weight is an *exponential* of the advantage and why `beta` is a temperature. Read
it for the derivation the code can only assert.

**IQL** (Kostrikov's reference implementation, `ikostrikov/implicit_q_learning`)
is the "what the grown-up version does differently" reading. Find the one place
our critic is still asked about an action it never saw: the `Q(s, pi(s))` baseline
in the advantage. IQL removes even that, replacing it with an *expectile*
regression that estimates the state value without ever querying an out-of-sample
action, closing the last crack the advantage constraint left open. The expectile
loss is a single line; find it, and you have the whole idea.
