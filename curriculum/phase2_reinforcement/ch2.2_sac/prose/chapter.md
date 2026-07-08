# 2.2: SAC and the Off-Policy Bargain

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## What PPO threw away

In chapter 2.1, PPO learned by acting. And then it deleted the evidence. Look
again at the loop: it collects a rollout, takes ten passes of minibatch SGD on
exactly that rollout, and throws every transition away. It has to. On-policy
means the update is only valid for data drawn from the *current* policy; the
moment the weights move, yesterday's rollout is off-distribution and the ratio
that keeps PPO honest stops meaning anything. PPO buys its famous stability by
being wasteful: each environment step is used once and discarded.

That waste has a name when it starts to hurt: **sample efficiency**. On a real
robot, where every environment step is a second of wall-clock and a little wear
on a motor, using each interaction exactly once is a luxury you cannot afford.
So here is the question this chapter answers by *measuring*, not asserting: can
we reuse experience (learn from transitions the current policy would never have
chosen) without the whole thing diverging?

## The bargain

Soft Actor-Critic takes the opposite deal. **Keep everything.** Every transition
the policy ever generated goes into a fixed-size **replay buffer**, and each
gradient step samples a fresh batch from the entire history: data from a hundred
different past policies, replayed dozens of times over. That is *off-policy*
learning, and it is where the sample efficiency comes from.

Nothing is free. Off-policy learning is unstable in exactly the way PPO's trust
region was built to prevent. You are training a Q-function to predict returns and
bootstrapping it from its own estimates on stale data; small overestimates
compound into a feedback loop that blows the value function to infinity. SAC pays
for the bargain with three braces, and you build all three from scratch:

1. **Twin Q critics with target networks** (clipped double-Q). Two independent
   Q-functions; always bootstrap from the *minimum* of the pair, read off
   *slow-moving copies*, the target networks. The min kills systematic
   overestimation; the slow targets stop the value function from chasing itself.
2. **A squashed-Gaussian policy**, trained by the reparameterization trick, so
   the actor can follow the Q-gradient straight through its own sampled actions.
3. **The maximum-entropy objective** with an **auto-tuned temperature `alpha`**,
   which pays the policy to stay uncertain (to keep exploring) instead of
   collapsing onto whatever the current, still-imperfect Q-function happens to
   like best.

## The task: dense reward is what off-policy exploits

Cartpole gave PPO a `+1`-per-step alive bonus, a signal with no gradient telling
you *which way* is better, which is why an on-policy method that reasons over
whole trajectories suited it. Pusher-reach (`common/envs/pusher_reach`) is the
opposite, and deliberately so: a planar two-link arm, a torque on each joint, and
a reward of **`-distance`** from the fingertip to a seeded-random target, *every
step*. That dense per-step signal is exactly what an off-policy learner can
bootstrap on from a buffer full of old transitions. The pairing is the point:
SAC's machinery earns its keep precisely when the reward is dense.

The baselines fix the scale. A random policy leaves the fingertip about
**0.176 m** from the target; the scripted IK reacher in the env gets it to
**~0.0001 m**. SAC has to climb from the first toward the second by acting,
storing, and replaying.

## The shape of the file

`sac.py` is one loop wrapped around the three braces. It reuses `common/`: the
pusher-reach env, the seeding helper, the device banner.

The squashed-Gaussian actor, a state-dependent mean and log-std, a
reparameterized sample, and the tanh log-prob correction that most from-scratch
implementations get subtly wrong:

```
[include-by-region: sac.py#model]
```

That correction inside `sample` is the one line worth slowing down for. We draw a
Gaussian and squash it through `tanh` so actions land in `[-1, 1]`, but `tanh`
compresses probability mass near the edges: a change of variables the log-prob
has to account for, or the entropy term SAC optimizes is simply wrong. The
`log(1 - tanh(x)^2)` subtraction is that accounting. Omit it and the temperature
tunes against a fiction; exploration quietly falls apart.

The replay buffer *is* the bargain, so it stays in plain sight: a circular NumPy
buffer, no framework. Note what it stores: `terminated`, not `done`. A time-limit
truncation must still bootstrap from the real next state (the ch2.1 lesson,
carried over unchanged); only a *true* terminal masks the future value.
Pusher-reach never terminates early, so `terminated` is always zero and every
target bootstraps:

```
[include-by-region: sac.py#replay]
```

The whole algorithm is one update function: critic step, actor step, temperature
step, then the soft target nudge.

The critic regresses both Q's toward the entropy-augmented Bellman target: the
clipped double-Q of the *next* action, minus the entropy bonus, bootstrapped one
step, and here it is in code:

$$
y = r + \gamma\,(1 - d)\Big(\min_{i=1,2} Q_{\bar{\theta}_i}(s', a') \;-\; \alpha \log \pi_\phi(a' \mid s')\Big),
\qquad a' \sim \pi_\phi(\,\cdot \mid s')
$$

```
[include-by-region: sac.py#update]
```

The training loop: act (random during warmup, then the policy), store the
transition, and once the buffer has warmed up, take exactly one gradient step per
environment step. Because there is a single env, `global_step` *is* the
environment-step count, so the eval curve reads directly as return-versus-env-steps,
the sample-efficiency signal the whole chapter turns on:

```
[include-by-region: sac.py#train]
```

## Run it

```
python curriculum/phase2_reinforcement/ch2.2_sac/sac.py --seed 0 --device cpu
```

<!-- wall-clock table renders from wallclock.csv -->

On a CPU laptop the default 30k-step config takes about **three minutes**, and
the held-out eval distance trends toward the scripted baseline (in fits and
starts, as RL does) while the success rate works its way up:

```
step   4000/30000  eval_dist 0.1483m  success 0.00
step  12000/30000  eval_dist 0.0626m  success 0.10
step  18000/30000  eval_dist 0.0358m  success 0.30
step  24000/30000  eval_dist 0.0475m  success 0.10
step  30000/30000  eval_dist 0.0434m  success 0.40
eval: mean final dist 0.0434m  success 0.40  return -5.8
      (random ~0.176m, scripted 0.0001m, solve<0.05m)
sample efficiency: solved (eval_dist<0.05m) at 18000 env steps
```

Nobody wrote the arm's controller gains this time; SAC found them by reaching,
storing every reach, and replaying the buffer dry. This is one seed, and RL is
noisy: the eval distance wobbles from step to step (watch it dip to 0.036 at
18k, drift back to 0.048 at 24k) because a held-out eval over ten episodes is a
small sample and the policy is still moving. The exercises read the signal
*across* seeds for exactly this reason.

## The bargain, measured against PPO

The headline claim (off-policy beats on-policy *here*) is measured
head-to-head, not asserted. `compare_ppo_sac.py` runs SAC and a compact
on-policy PPO reference (the ch2.1 family, retargeted to pusher-reach) on the
*same* env and counts the environment steps each needs to drive the eval
distance below the `0.05 m` solve bar:

```
=== sample efficiency: env steps to eval mean final dist < 0.05 m (seed 0) ===
  SAC (off-policy):  18,000 env steps       [budget 30,000]
  PPO (on-policy):   NOT solved in budget   [budget 200,000]
  -> SAC solved; PPO did not reach the bar within 200,000 steps.
```

The on-policy reference is not broken: it *learns*, dragging the fingertip from
the random baseline (~0.176 m) down to a plateau around **0.13 m**. It just
plateaus there, still short of the bar SAC cleared at 18k, after spending its
entire 200k-step budget: an order of magnitude more environment interaction.
That is the off-policy bargain paid off: replaying each transition dozens of
times converts the same dense signal into far more learning per environment step.

Two honesties keep that claim from becoming a slogan. First, **the PPO reference
is untuned.** A PPO with reward normalization, an entropy bonus, and a tuned
learning rate would close some of this gap. The point is not that PPO *cannot*
solve pusher-reach, it is that off-policy needs far less of that tuning to be
sample-efficient here. Second, **the win is the sample-efficiency gap in
environment steps, not a claim that off-policy always wins.** It wins in *this*
regime: dense reward, cheap env, heavy sample reuse. Change the regime and the
ledger changes with it, which is the next section.

Why the comparison lives in its own file and not inside `sac.py`: a faithful
second RL algorithm does not fit under the 450-line cap alongside SAC. The
teaching artifact stays SAC; the measurement rig is companion tooling, and it
imports no RL framework: plain torch and numpy, the ch2.1 PPO inlined.

## When off-policy does NOT win

The lesson is *when*, not *always*. Off-policy replay wins here because three
things line up: the reward is **dense** (every stored transition carries a usable
gradient), the env is **cheap** (so PPO's extra environment steps are the cost
that dominates, and SAC's per-step gradient work stays affordable), and the task
tolerates **heavy reuse** (old transitions stay informative). Flip any one of
them (a sparse reward where most stored transitions say nothing, or an expensive
rollout where SAC's many gradient steps come to dominate the wall-clock) and the
bargain gets worse. The replay-size exercise lets you feel one edge of this
directly.

## Break it (optional, not graded)

`--break` bootstraps the Q-target off the *online* critics instead of the target
networks. Run it and the failure is not subtle: the Q-loss blows up by five or
six orders of magnitude (a healthy run settles around `0.003`; this one posts
`q_loss` in the tens of thousands), the eval distance never leaves the random
baseline, and the auto-tuned `alpha` runs away chasing an entropy target the
diverging critics have made meaningless. That is the exact feedback loop the slow
target network exists to break. This is a teaching toggle, not a graded bug-hunt:
per the RL doctrine, a single-run bug can heal across seeds, so the graded
exercises are multi-seed instead.

## Exercises

Two, in `exercises/`, both predict-then-run and both graded on a seed-robust
signal rather than one lucky run. The first has you predict whether SAC's reach
survives across seeds 0–2 before you train them; the second is a hyperparameter
investigation: starve the replay buffer to 1/20th its size and predict what
happens to the bargain, then read the effect against the default across the same
three seeds.

## Read the real thing

The single-file SAC you just built mirrors CleanRL's `sac_continuous_action.py`,
pinned at `v1.0.0` (commit `c37a3ec`). It is the same algorithm at the same
altitude (one script, roughly 310 lines, much of it argparse and TensorBoard),
so reading it next is a graduation, not a leap. Three points, ours set beside
theirs.

**The squashed-Gaussian actor.** Ours is the `Actor` in the `model` region: a
state-dependent mean and log-std, an `rsample` reparameterized draw, and the
`log(1 - tanh(x)^2)` correction on one line. CleanRL's is `Actor.get_action`
(`sac_continuous_action.py:135–147` @ `c37a3ec`), the *same* three moves:
`rsample` at line 139, the tanh squash at 140, and the log-prob correction at
line 144. Two differences worth seeing. Its `forward` bounds the log-std by
squashing through `tanh` into `[-5, 2]` (lines 130–131) where we hard-`clamp`;
and it carries `action_scale`/`action_bias` buffers (lines 117–123) to remap the
`[-1, 1]` tanh output onto arbitrary action bounds. Pusher-reach's torques
already live in `[-1, 1]`, so we cut the rescale, one fewer thing between you
and the correction.

**The twin critics and the bargain.** Ours are two `Critic`s and their
`deepcopy` targets in `model`, bootstrapped from the min in `update`. CleanRL's
`SoftQNetwork` (lines 91–103) is instantiated four-up (`qf1, qf2, qf1_target,
qf2_target`) at lines 186–191, and the same clipped-double-Q target is
assembled at lines 248–253. Its replay, though, is not hand-rolled: it imports
`stable_baselines3`'s `ReplayBuffer` (lines 205–211) with
`handle_timeout_termination=True` (line 210): the library doing exactly the
`terminated`-not-`done` bookkeeping our `replay` region spells out by hand.

**What it adds, and why.** Three things we stripped. Vectorized envs:
`gym.vector.SyncVectorEnv` (line 180) behind a `make_env` thunk (lines 75–87),
so `global_step` is no longer one env step; we kept the single env so the
sample-efficiency curve reads straight off the loop. Delayed, TD3-style updates:
the actor and target networks fire every `policy_frequency` steps (line 265),
not every step like ours. And split learning rates, `q-lr 1e-3` against
`policy-lr 3e-4` (lines 56, 58), where we share one. None of these changes the
bargain. They tune it. The off-policy machinery is identical; production just
adds the knobs and the plumbing this chapter held back to keep the algorithm in
view.

Read next: open the file at `get_action` (lines 135–147), the part you already
own, then follow `global_step` down from line 216; CleanRL's write-up walks the
full derivation at `docs.cleanrl.dev/rl-algorithms/sac`.

## What's next

You now have two RL policies in hand and a working sense of the on-policy /
off-policy trade: PPO's stable-but-hungry rollouts against SAC's sample-efficient
replay. Chapter 2.3 attacks the other axis entirely: instead of squeezing more
learning out of each environment step, it runs *thousands of environments at
once* on the GPU with MJX, and the wall-clock cliff that opens up is its own
lesson. And the actor you just trained does not retire: `sac.py` saved it to
`outputs/ch2.2-sac/sac_actor.pt`, and chapter 2.6 drags it out of pusher-reach's
perfect world to see how a policy trained on clean observations holds up when the
sensors start to lie. The replay-and-entropy substrate you built here also
returns in Phase 4, where HIL-SERL builds human-in-the-loop corrections on top of
exactly this off-policy machinery.
