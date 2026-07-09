# 2.4: Reward Design Is Programming

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## You never tell the robot to walk

In every chapter so far you handed the robot its behavior. In Phase 1 you handed
it a demonstrator's trajectories; in 2.1 you handed it a reward the environment
had already written for you. This chapter hands you the pen. You will not tell the
quadruped to walk. You will write down a *number* for every state it can be in, and
Proximal Policy Optimization (the same PPO you built in 2.1, unchanged) will find
whatever behavior makes that number as large as it can.

That is the uncomfortable truth of reinforcement learning: **the reward function is
the program you write for the robot.** And like any program, it does exactly what
you *wrote*, which is frequently nothing like what you *meant*. This chapter builds
four reward programs for one robot and measures what each actually produces. Three of
them teach a walk (one barely, one well, one best of all). One of them cheats.

## The setup: one variable, everything else held fixed

The `common/envs/quadruped` env is built for exactly this. Instead of returning a
single opaque reward, every `step()` hands back the reward already split into five
named, shapeable terms in `info["reward_terms"]` (`forward`, `upright`, `height`,
`alive`, `ctrl`), plus the raw signals behind them (`height`, `up_z`,
`forward_vel`). So a "reward design" in this chapter is just a Python function of
that info:

```
[include-by-region: rewards.py#rewards]
```

We *ignore* the env's own summed reward and score each step with one of these
programs instead. Everything else (the env, the PPO, the seeds) is identical
across all four designs. The reward is the only thing that changes, and that is the
entire architecture of the file: a single `train(reward_fn)` that every design
calls.

```
[include-by-region: rewards.py#train]
```

(The PPO is 2.1's, trimmed: rollout with the truncation-versus-termination
bootstrap you already learned, GAE, a clipped surrogate. It is deliberately *not*
the subject here. The reward is.)

## Lesson 1. Shaping: sparse rewards barely train

The most honest reward for "walk forward" is sparse: pay the robot only when it is
already moving forward fast, and nothing otherwise (`r_sparse`). It is correct (a
policy that maximized it *would* walk) and it is nearly useless. From a standing
crouch the robot gets zero signal in every direction; there is no gradient telling
it which way is "warmer". It flails, often falls, and drifts forward almost by
accident.

The env's **dense shaped** reward (`r_shaped`, the five terms summed) instead pays a
little for every good thing along the way: some forward velocity, staying upright,
holding ride height, not falling. That graded signal is a slope the policy can climb
out of the crouch and into a gait. Measured, seed 0, at the default config:

```
sparse   forward +0.79 m   (it stumbles forward and falls short of the horizon)
shaped   forward +4.61 m   (a sustained, full-horizon walk)
```

Same robot, same PPO, same seed. The only difference is that one reward described
the *destination* and the other described the *path*. Shaping is how you make a walk
learnable at free-tier scale. And, not by coincidence, it is exactly what makes the
next lesson possible.

## Lesson 2. Reward hacking: what you said vs what you meant

Now write a reward carelessly. Suppose you reason: "a walking robot holds itself up,
so I'll just reward height" (`r_hack`: raw torso height, no forward term at all). It
is a plausible-sounding proxy. It is also a trap.

PPO optimizes it beautifully. Over training the hacked reward climbs by roughly
**10×** (measured: its own return rose from about 105 to about 1048). The policy is
unmistakably *learning*. But look at what it learned:

```
hack   height_m   0.277    (the tallest of all four designs — it reared up)
       forward_m -0.181    (it walked NOWHERE — a hair backward)
```

The reward went up and to the right while the robot stood tall and went nowhere.
Nothing broke; PPO did its job perfectly. **The policy did exactly what you said (be tall), not what you meant (go forward).** That gap is *specification gaming*, and
it is not a corner case; it is the default failure mode of reward design. The
canonical example is a boat-race game in which an agent asked to win the race
instead learned to spin in a lagoon collecting respawning power-ups, scoring higher
than any finisher while never completing the course (see *Read the real thing*). Our
height-hack is the same bug in miniature, and (the point of this chapter) it is
*measured*, not asserted: the reward rises, the intended metric does not.

How do we know the intended metric did not move? Because eval measures behavior
directly, separately from whatever reward trained the policy. It rolls out the
policy mean on held-out seeds and records forward distance in meters (the thing you
actually wanted) alongside the design's own return:

```
[include-by-region: rewards.py#eval]
```

The fix is not "optimize harder"; that only games the proxy better. The fix is a
better program: say what you actually meant. Add the forward term back and the
robot walks. That is the chapter's code-completion exercise.

## Lesson 3. Curriculum: shape the reward in time

You can also stage the reward. `r_curriculum` pays only the standing terms (upright,
height, alive, ctrl, everything *except* forward) for the first half of training,
then switches the forward term on. Learn to stand solidly, *then* learn to move. At
the default config, seed 0, this staged reward produced the strongest walk of the
four:

```
curriculum   forward +7.84 m
```

Report that honestly, though: curriculum won *here*, at this seed and this
switch-point, but the win is not guaranteed across seeds and is sensitive to where
you place the switch (`CURRICULUM_SWITCH`). Staging the reward is a tool that often
helps, not a law that curriculum always beats flat shaping. The claims that hold
seed-to-seed, and that the exercises pin, are the shaping *ordering* (shaped walks
farther than sparse) and the hack *mismatch* (its reward climbs while its forward
distance stays near zero), not this particular magnitude.

## Run it

```
python curriculum/phase2_reinforcement/ch2.4_rewards/rewards.py --seed 0 --device cpu
```

This trains all four designs and prints the comparison. On a T4 the full run is
about 12 minutes; on a cpu-laptop it is 2.61 minutes, about 4.5× faster, because at
this tiny network size a T4's per-kernel launch overhead costs more than its
throughput buys back. The startup banner prints the live figure for your machine.

<!-- wall-clock renders from wallclock.csv (measured: T4 11.73 min, L40S 8.13 min, cpu-laptop 2.61 min) --> The reward curves and the per-term
contributions go to the rerun recording (`--rerun` is on by default): open it and
watch the hack's height term climb while its forward term flatlines.

## The honest limits

This is a cartoon quadruped (two-DOF legs in the sagittal plane, see the env README)
and free-tier PPO gives wobbly, short walks, not a transferable gait. That is
fine, because the lesson is not the gait. The lesson is the *relationship between the
number you wrote and the behavior you got*, and that relationship (including the
hack) reproduces cleanly at this scale.

## Read the real thing

Pair this chapter with the specification-gaming literature: OpenAI's *Faulty Reward
Functions in the Wild* (the CoastRunners boat-race hack that Lesson 2 borrows) and
DeepMind's running list of specification-gaming examples. Dozens of agents, across
domains, all optimizing exactly what was written and nothing that was meant. Then
read one production legged-locomotion reward module to see "the reward is a program"
at full scale: the same five ideas you shaped here (forward drive, an upright term,
a height target, an alive bonus, a control penalty) become forty weighted terms an
engineer tunes for weeks. The upstream commit is pinned by the read-the-real-thing
segment.
