# 4.3: RL Post-Training: HIL-SERL in Sim

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## Everything you built, in one loop

This is the convergence chapter. Trace what you already own:

- **A base policy** by cloning demos (BC, 1.1), and the covariate-shift death you
  measured when it drifted into states no demo covered.
- **Corrections** (4.2): you rolled out that policy, caught it failing, and had an
  expert (the scripted stand-in for your own teleop hand) label the states it
  actually visits. On-policy corrections, exactly where the policy is weak.
- **An offline prior** (the primer): given a fixed log, don't just clone it:
  fit a Q-function and extract the *above-average* actions with advantage-weighted
  regression (AWAC). The reward BC threw away, put to work.
- **Off-policy RL** (2.2): SAC, with its replay buffer, twin-Q critics, target
  networks, and squashed-Gaussian policy, learning from every transition many
  times.

**HIL-SERL (Human-in-the-Loop Sample-Efficient RL, Luo et al.) is those four
things composed into one loop.** Take the corrections. Turn them into an offline
prior with AWAC. Then fine-tune that prior online with SAC, keeping the
corrections in the replay buffer the whole time, so the policy reaches a good
return in *far fewer online environment samples* than RL starting from scratch.
Sample-efficiency is the whole point, and this chapter measures it.

Why "sample-efficient" is the right obsession: online RL's currency is
environment interaction, and on a real robot every environment step is a real
second of real hardware wearing out. RL-from-scratch on a manipulation task can
cost millions of steps. If your correction data can buy back most of that, that is
the difference between "trains overnight on a robot in a lab" and "never."

## The setup, honestly scaled

We run the real algorithm on the public `pusher_reach` env (2.2's: dense reward,
a 2-link arm reaching a seeded target) so every line is readable and reproducible
on a laptop. This chapter teaches the mechanism at free-tier scale. Nothing here
claims a strong trained robot: it is the algorithm, shown honestly.

The corrections are scripted-expert transitions: the analytic IK reacher driving
the task to the goal, our offline stand-in for a human teleoperator taking over on
the policy's failures (4.2's mechanism, exactly). This is the only data the
offline prior ever sees.

```
[include-by-region: serl.py#data]
```

## Build the prior: corrections → AWAC

The prior is the primer's AWAC, run on the corrections. A twin-Q critic learns the
reward-to-go on the fixed correction transitions; then the policy is extracted by
advantage-weighted regression: behavior cloning's MSE onto the correction action,
but each sample scaled by `exp(A/beta)`, where `A = Q(s, a_corr) - Q(s, pi(s))` is
how much better the correction was than what the current policy would do. This
anchors the policy to actions the corrections actually contain (no
out-of-distribution extrapolation, the failure the primer's `--naive` showed) and
warm-starts both the actor and the critics that SAC will continue from.

```
[include-by-region: serl.py#prior]
```

Note one small but load-bearing detail in the actor: we initialize `log_std`
small, so a freshly-primed policy is **near-deterministic**. A wide-open
stochastic policy would explore off the prior's good manifold on the very first
online steps and unlearn it, the offline-to-online collapse. Starting quiet is
half the battle of not throwing the prior away.

## Fine-tune online: SAC with the corrections still in the replay

Now the online phase. It is ch2.2's SAC, unchanged (one env step, one gradient
step on a replay batch) with two differences that make it HIL-SERL:

1. The actor and critics start from the prior, not from random.
2. The replay buffer is **pre-loaded with the corrections**, and they never get
   evicted (RLPD-style). Every online gradient step samples a mix of fresh online
   transitions and the original corrections: the corrections keep pulling the
   critic toward known-good actions while online data refines it.

The same loop runs the from-scratch baseline: identical algorithm, but cold nets
and an empty replay with a random warmup. The only differences between the two
arms are the starting weights and the replay's initial contents. That is the
experiment.

```
[include-by-region: serl.py#online]
```

## What we measure: samples-to-threshold

The headline is a curve: eval reach-distance versus online environment samples,
for both arms. We read off **samples-to-threshold**: the first online step at
which the eval mean final distance drops below the bar (`--threshold`, 0.10 m;
random leaves ~0.176 m).

**The result, stated truthfully.** HIL-SERL clears the threshold at **zero**
online samples on every seed, because the offline prior, built entirely from the
corrections with no environment interaction, is already past the bar (eval
distance 0.061 / 0.074 / 0.069 m across seeds 0–2, well under the 0.10 m
threshold). From-scratch SAC has to discover the reach from nothing and needs
**~10,000–11,000** online samples to reach the same line. That horizontal gap on
the sample axis is the sample efficiency, and it is large and seed-robust. On the
pooled held-out eval, HIL-SERL's success rate is at least from-scratch's on all
three seeds, and the difference interval excludes zero on two of them.

And the honest part, which is itself the lesson: on this small dense-reward task,
online fine-tuning **holds** the prior rather than beating it. The prior already
sits near what SAC can reach on this task in a short horizon (2.2's own SAC needs
~18k steps just to solve it), so there is little headroom left for online RL to
add, and the offline-to-online transition can wobble the policy before it
recovers, which is exactly why we (a) keep the corrections in the replay to anchor
it and (b) return the **best** checkpoint over the run (4.2's return-the-best
idiom), prior included. On harder tasks (the real HIL-SERL paper's contact-rich
manipulation) the prior is far from optimal and the online phase is where the
policy actually gets good. Here, the corrections do the
work, and the honest headline is the sample axis.

## The ablation: what each piece buys

```
[include-by-region: serl.py#report]
```

Read the three arms together. **Prior alone** ≈ **HIL-SERL** ≫ **from-scratch at
the same small online budget** (measured pooled success 0.37 / 0.33 / 0.33 for the
prior and for HIL-SERL, versus 0.10 / 0.07 / 0.27 for scratch across seeds 0–2).
The corrections-as-prior are the sample-efficiency win; online fine-tuning
maintains it. If a piece did not help on the free tier, we report that, not paper
over it (the ch2.1 spike doctrine). The mechanism is what transfers: corrections
become a prior, the prior gives you a running start, and RL post-training refines
from there.

## Wall-clock (free tier)

<!-- wall-clock table renders from wallclock.csv -->

## Read the real thing

The single-file HIL-SERL you just built is the mechanism with the scaffolding
stripped away. Two papers are the natural graduation. The **HIL-SERL** paper (Luo
et al., "Precise and Dexterous Robotic Manipulation via Human-in-the-Loop RL") is
the grown-up version of this exact loop: a real human teleoperator in the loop
online instead of a scripted stand-in, a **learned reward classifier** instead of
a dense analytic reward, and contact-rich tasks where the prior is far from
optimal and the online phase genuinely earns its keep. **RLPD** (Ball et al.,
"Efficient Online RL with Offline Data") is where the corrections-in-replay trick
comes from, plus the stabilizers this file leaves out (explicit symmetric
sampling, a high update-to-data ratio, and LayerNorm critics) which are what let
online fine-tuning improve a prior on hard tasks rather than merely hold it. (The
exact upstream commits are pinned by the read-the-real-thing build step.)

## Exercises

Two, both predict-then-run and both graded on a seed-robust signal rather than one
lucky run.

- **Predict the sample-efficiency gap** (`ex1_predict_sample_efficiency.py`): does
  HIL-SERL clear the threshold in fewer online samples than from-scratch, on every
  one of seeds 0–2? Predict before you run.
- **Investigate the corrections** (`ex2_investigate_corrections.py`): starve the
  correction data and watch the prior (and HIL-SERL's head start) weaken. The
  corrections are the prior; the prior is the sample efficiency.
