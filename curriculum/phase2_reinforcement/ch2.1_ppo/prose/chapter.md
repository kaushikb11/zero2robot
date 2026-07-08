# 2.1: PPO: The Policy That Acts and Sees the Consequences

## The death you already measured

In chapter 1.1 you watched behavior cloning die. Not on the training set (the
loss curve there looked fine, right to the end), but in the rollout, where the
policy chose the states it visited. The moment it drifted a little off the
demonstrator's path it landed in a state the demos never covered, made a
slightly worse guess, drifted further, and the rerun trace showed the block
sliding hopelessly past the goal. That is covariate shift, and it is not a bug
you fix by cloning harder. The policy never *acts* during training, so it never
*sees* the consequences of its own mistakes, so it never learns to correct
them. You measured the death. This chapter is the cure.

Reinforcement learning removes the demonstrator entirely. The policy acts. The
environment answers with a reward. And (this is the whole point) the states it
trains on are the states *it* causes. The distribution BC could never reach is
the only distribution RL ever sees. This chapter builds Proximal Policy
Optimization, the workhorse that makes that idea stable enough to actually run,
from scratch in one file, on the cartpole you met in `common/envs/cartpole`.

## The task, and the one thing it forces you to get right

Cartpole: push a cart left and right to keep a hinged pole upright. The reward
is `+1` for every step the pole stays up, so an episode's return is simply how
many steps you survived, capped at 500. A random policy scores about 34; the
scripted balancer built into the env scores a perfect 500. PPO has to climb from
the first number to the second by trial and error, with nobody telling it which
action was right.

Cartpole is the standard first PPO task for a reason, but it also hides a trap
most tutorials get wrong. An episode can end two ways, and they are not the same
event:

- **terminated**: the pole fell (or the cart ran off the rail). This is a real
  terminal state: there is no future, so the value ahead of it is exactly 0.
- **truncated**: the 500-step budget ran out with the pole *still up*. This is
  not a failure. The pole would have kept balancing; we just stopped watching.
  The value function must **bootstrap** from where the episode would have
  continued. The future is real, we simply didn't record it.

Conflate them (treat "time ran out" as "the pole fell") and you teach the
value function that surviving all the way to the horizon is worth nothing, which
is precisely backwards: reaching the horizon is the goal. The env hands you both
flags separately (`info["terminated"]`, `info["truncated"]`) for exactly this
reason, and getting that one distinction into the advantage estimate is the
subtlest line in the file.

## The shape of the file

`ppo.py` is one loop wrapped around a handful of ideas. It reuses `common/`: the
cartpole env, the seeding helper, the device banner. Everything specific to PPO
is on the page.

The Gaussian policy and value net come first: separate MLPs, a learned
log-std, and the orthogonal init that quietly keeps PPO stable. The policy
outputs the *mean* of a Gaussian over the cart's force; exploration is the
spread of that Gaussian, and the spread is a learned parameter rather than a
network output. That is the standard continuous-control choice, and the simplest
thing that works here:

```
[include-by-region: ppo.py#model]
```

Rollout collection is deliberately un-hidden: a plain list of envs stepped in a
Python loop, with no `gym` vector-env wrapper swallowing the reset logic. The
autoreset and the truncation bootstrap are the lesson, so they stay in front of
you. Watch what happens when an episode ends: before the env resets, we store
`V(the real next state)` in `bootstrap_buf`, because the very next slot in the
buffer will hold a *fresh* episode's first observation, and bootstrapping off
that would be nonsense:

```
[include-by-region: ppo.py#rollout]
```

The advantage estimate is where the termination/truncation distinction collapses
to a single line. Generalized Advantage Estimation walks the rollout backward;
the mask `(1 - terminated)` zeroes the bootstrap after a fall, while a truncated
step keeps its stored `bootstrap_value`. The `--break` flag swaps `terminated`
for `done` (treating a time-limit truncation like a fall), so you can measure
what conflating them costs instead of taking my word for it.

With advantages in hand, the update is PPO's defining move: the **clipped
surrogate**. Maximize the policy ratio times the advantage, but clip the ratio
to a narrow band so no single minibatch can shove the policy far from the data it
was collected under. That clip is the entire reason on-policy updates are stable
enough to repeat for ten epochs over the same rollout:

Maximize the policy ratio times the advantage, but take the *pessimistic* branch
against a clipped ratio so no minibatch can shove the policy far from the data it
was collected under, and here it is in code:

$$
L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\!\left[\,\min\!\big(r_t(\theta)\,\hat{A}_t,\;\; \mathrm{clip}\big(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\big)\,\hat{A}_t\big)\right],
\qquad r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
$$

```
[include-by-region: ppo.py#update]
```

## Run it

```
python curriculum/phase2_reinforcement/ch2.1_ppo/ppo.py --seed 0 --device cpu
```

The reference numbers here are from `--device cpu`, the one configuration this
book can promise is bitwise-deterministic under `--seed`. Run it twice and the
checkpoint matches byte-for-byte. On a GPU or Apple's mps backend the same seed
reproduces the *result* (the curve climbs, the eval solves) but not the exact
bytes, and this chapter never pretends otherwise.

<!-- wall-clock table renders from wallclock.csv (ch2.1-ppo) -->

On a CPU laptop the default 200k-step config finishes in about **0.32 min**, and
the mean episodic return climbs from the random baseline toward the ceiling:

```
iter  10/97  mean_return  154.5
iter  30/97  mean_return  267.7
iter  60/97  mean_return  387.1
iter  90/97  mean_return  496.0
eval: mean return 500.0 over 20 episodes (random ~34, scripted 500, cap 500)
```

PPO matches the scripted balancer (a perfect 500), but nobody wrote this
balancer's gains. The policy found them by acting, failing, and adjusting, on
states no demonstrator ever showed it. Open the recording to watch it happen:

```
rerun outputs/ch2.1-ppo/ppo.rrd
```

The healthy signature is `charts/episodic_return` sawing upward and
`losses/approx_kl` staying small: each update nudges the policy, never yanks it.
If the return curve spikes and collapses, the clip is doing its job too loosely;
that is a knob, and the next section is about turning the knobs.

## The tricks are flags: turn them off and watch

The gap between "PPO the paper" and "PPO that trains" is a handful of engineering
tricks. Rather than bake them in silently, each is a flag you can switch off to
measure what it buys: `--no-norm-adv` (advantage normalization),
`--no-clip-vloss` (value-loss clipping), `--gae_lambda 1.0` (plain Monte-Carlo
returns, no GAE), `--no-anneal-lr` (constant learning rate). Run the ablation and
read the reward curves side by side in rerun.

But there is a second, harder lesson underneath the first, and it is the reason
this chapter has no single-run "break it" bug the way the imitation chapters do:
**one RL training run is noise.** Turn off advantage normalization and, across
seeds 0–2 at the default config, the eval returns come out `[332, 500, 500]`
against the reference's `[500, 500, 500]`. Advantage normalization clearly helped,
on average. But on two of those three seeds the ablation still solved the task
outright. Had you run one seed and drawn a conclusion, that seed could have told
you anything. In RL you predict, then you run several seeds, then you read the
average; exercise 2 makes you feel that spread with your own hands, which is the
whole point of it.

## Read the real thing

Every chapter points you at the production code it was carved from, but this is
the one place where the from-scratch file and the real file are nearly the same
object. CleanRL's `ppo_continuous_action.py` is a single script with no framework
underneath it (the exact discipline `ppo.py` holds itself to), which is why the
RL community reaches for it as the reference PPO. At the pin this chapter reads
(`vwxyzjn/cleanrl` at tag `v1.0.0`) it is 319 lines to our 349; of every "real
thing" in the course, this is the closest match to what you just built, and the
diff is small enough to read line by line.

Start with what is identical, because most of it is. Our `#model` region
(separate critic and `actor_mean` MLPs, a learned `actor_logstd` parameter,
orthogonal `layer_init` with the 0.01 final-layer gain) is line-for-line
CleanRL's `Agent` (`ppo_continuous_action.py:106–135`) and `layer_init`
(`:100–103`), down to the `get_action_and_value` method that samples or scores
depending on whether you pass an action. Our clipped surrogate in `#update` is
CleanRL's `pg_loss1`/`pg_loss2`/`torch.max` (`:270–272`); our value-loss clipping
matches `:276–287`; the backward GAE walk that produces our advantages mirrors
CleanRL's "bootstrap value if not done" block (`:222–236`). You wrote this file
already. Seeing it in a second hand is how you learn which lines were PPO and
which were mine.

Now the one place they diverge, and it cuts in our favor. CleanRL v1.0.0 predates
gym's termination/truncation split: it steps the old API,
`next_obs, reward, done, info = envs.step(...)` (`:211`), and its GAE masks
purely on `done` (`:229`, `:232`), so a time-limit truncation is folded into a
fall exactly the way our `--break` flag does it *on purpose*. Our `bootstrap_buf`
and the `1 - terminated` mask exist precisely to not make that mistake. This is
the rare chapter where the teaching version is the more correct one, because we
wrote ours after gym split the flags apart.

What CleanRL adds is everything around the algorithm, and it earns the extra
lines honestly. `gym.vector.SyncVectorEnv` (`:168–170`) and the
observation/reward-normalizing wrapper stack in `make_env` (`:80–97`) are what
let the same loop train on MuJoCo continuous control, not just cartpole. This is
the `gym` machinery we deliberately unhid in our `#rollout` region. TensorBoard plus
optional Weights & Biases logging (`:153`, `:305–315`) sit where we log to rerun.
And the diagnostics you will want the moment a run misbehaves are all there:
`approx_kl` and `clipfracs` (`:260–263`), `explained_variance` (`:301–303`). None
of it is the algorithm. All of it is what turns the algorithm into something you
can run at scale and debug when it breaks.

Read next: `cleanrl/ppo_continuous_action.py` at `v1.0.0`. It is the file you
graduate to.

## Exercises

Two, in `exercises/`. The first (`ex1_completion_gae`) hands you GAE with the
bootstrap logic blanked out and a hand-built rollout containing exactly one fall
and one time-out. The only way to pass is to bootstrap on truncation but not on
termination, the one line this whole chapter turns on. It runs on fixed arrays,
not a training run, so it is deterministic and never flakes on RL variance. The
second (`ex2_predict_ablation`) is a predict-then-run: commit to whether
advantage normalization helps *before* you watch three seeds decide it, and
reconcile your prediction with the spread.

## What's next

PPO works, but look at what it cost: 200k environment steps to balance a pole a
scripted controller solves in four lines. PPO is *on-policy*: every update
throws away the rollout that produced it and collects fresh data, because the
clipped surrogate is only valid near the data it was measured on. That is a
brutal sample budget for anything slower than cartpole. The next chapter keeps
the acting-and-observing cure but stops throwing the data away: SAC learns
off-policy, reusing every transition from a replay buffer, and pays for that
sample efficiency with a different kind of instability. That trade (sample
efficiency against stability) is the bargain the rest of Phase 2 negotiates.
