# 2.5 — Locomotion: The Quadruped Walks

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## Nobody writes the gait

Back in chapter 2.4 you had a walking quadruped, and you wrote its gait by hand.
`trot_action` is a sinusoid — hips sweeping fore-and-aft, knees flexing on the
swing — with amplitudes and a frequency you tuned until the legs happened to carry
the body forward. It works, and it is a lie of sorts: *you* are the controller.
Change the robot, change the floor, change the speed you want, and you are back at
the whiteboard re-deriving sinusoids.

This chapter takes you out of the loop. You hand SAC — the off-policy learner you
built from scratch in chapter 2.2 — the *same* quadruped and the *same* five-term
reward (go forward, stay upright, hold your ride height, don't fall, don't thrash),
and let it act. Out of thousands of episodes of trial and error, a repeating stride
**emerges**. Nobody specified a gait; the reward and the physics did. That
emergence is the whole chapter.

## The env is unchanged — that is the point

We do not touch `common/envs/quadruped`. Same `obs[23]`, same `action[8]`, same
reward weights chapter 2.4 shipped. If a gait is going to appear, it has to appear
from the reward *as given*, on the body *as built*. The only thing that changes
between 2.4 and 2.5 is who chooses the joint targets each step: a hand-written
sinusoid, or a neural network trained to maximize return.

## What the policy sees, and why (observation design)

A locomotion policy is only as good as its senses. The env's 23-number observation
is a deliberate menu, and reading it is half the lesson:

- **Joint angles + joint velocities (0..15)** — *proprioception*. Where each of the
  eight leg joints is, and how fast it is moving. This is what a real robot reads
  off its motor encoders; it is the policy's sense of its own body.
- **Torso height (16), up-vector (17..19)** — *posture*. How tall the body rides and
  which way "up" points. The reward's height and upright terms are computed from
  exactly these, so the policy can see the very quantities it is graded on.
- **Torso linear velocity (20..22)** — *how fast, which way*. Entry 20 is the forward
  speed `vx` the reward pays for. Without it the policy is walking with its eyes shut
  to its own momentum.

Two design choices are worth pausing on. First, there is **no clock and no
foot-contact flag** in the observation. A gait is periodic, but the policy is not
handed the phase of the cycle — it has to *infer* where it is in its stride from the
joint and body velocities. Second, orientation is an **up-vector, not a quaternion**:
a Gaussian policy trips over the sign/wrap seam of a quaternion, while the
up-vector's z-component is directly the number the reward reads. Good observation
design is choosing coordinates the network can actually learn from.

The `--blind_velocity` flag zeroes obs 20..22 so you can *measure* what that
forward-speed sense is worth — that is exercise 2.

## What the policy commands, and why (action design)

The eight actions are **residual position targets**, not torques. The env commands
`DEFAULT_POSE + ACTION_SCALE * action` to PD position servos, so:

- **action 0 = stand.** The servos already hold the crouch, so the do-nothing action
  doesn't fall over. The policy starts from a stable anchor and only has to learn the
  small *offsets* that turn standing into striding.
- **the search is easier.** Learning residuals around a good default is a far gentler
  optimization than learning raw joint torques from scratch, where do-nothing means
  collapse. This is the standard legged-RL setup for a reason, and it is why a
  from-scratch SAC can find a gait on a laptop at all.

## The algorithm is 2.2's SAC, unchanged

There is almost nothing new in the learner — and that reuse is deliberate. The
squashed-Gaussian actor, twin Q critics with target networks, the auto-tuned entropy
temperature, one gradient step per environment step: all of it is chapter 2.2,
pointed at a new env. Locomotion changes the *data*, not the algorithm.

```
[include-by-region: walk.py#model]
```

The replay buffer stores `terminated`, not `done` — the same truncation lesson from
2.1/2.2, and it *matters more* here. A fall is a true terminal (the future really is
zero), but running out the 500-step clock while still walking is a time limit you
must bootstrap through. Conflate them and the policy learns that walking all the way
to the horizon is worthless.

```
[include-by-region: walk.py#replay]
```

The update is 2.2 verbatim:

```
[include-by-region: walk.py#update]
```

## A first taste of domain randomization

There is one genuinely new line, and it lives at reset. Each training episode, we
scale the torso mass by a small random factor (default ±15%):

```
[include-by-region: walk.py#env]
```

Now the policy never meets the exact same body twice, so it cannot memorize one
body's precise dynamics — it has to find a gait that works across a little variation,
which is a more robust gait. This is a **preview**, one randomized number, of chapter
2.7, where randomizing friction, mass, latency, and external shoves becomes the
entire story of crossing the reality gap. The `--no-domain-rand` flag turns it
off so you can compare. Note that evaluation always uses the *nominal* body, so the
gait-emergence curve is measured on one fixed robot and stays comparable across the
run.

## Run it — and watch the gait emerge

```
python curriculum/phase2_reinforcement/ch2.5_walk/walk.py --seed 0 --device cpu
```

<!-- wall-clock table renders from wallclock.csv (measured: cpu-laptop 7.81 min, T4 13.58 min, L40S 8.77 min) -->

On a CPU laptop the default 60k-step config takes about **8 minutes**. Watch the
forward distance climb off the standing baseline as a stride settles in:

```
step   5000/60000  eval_return 179.1  fwd_dist +1.596m  fwd_vel +0.327m/s  len 255
step  15000/60000  eval_return 147.1  fwd_dist +1.247m  fwd_vel +0.358m/s  len 219
step  20000/60000  eval_return 307.6  fwd_dist +3.370m  fwd_vel +0.475m/s  len 357
step  30000/60000  eval_return 330.9  fwd_dist +3.732m  fwd_vel +0.517m/s  len 369
step  45000/60000  eval_return 374.6  fwd_dist +4.467m  fwd_vel +0.584m/s  len 386
step  60000/60000  eval_return 247.2  fwd_dist +3.025m  fwd_vel +0.581m/s  len 249
eval: forward +3.025 m  vel +0.581 m/s  return 247.2  len 249
      bar: scripted trot +2.154 m / return 306.6  (random ~-0.30 m, stand ~-0.01 m)
gait emergence: walks (fwd>0.5m) at 5000 env steps
```

The honest read is richer than "did it work". **A gait emerges, and it is fast.** By
20k steps the policy is walking, and it walks *further* than the hand-scripted trot —
**+3.0 m of forward travel versus the trot's +2.15 m**, at nearly triple the forward
speed. Nobody wrote that gait; SAC found it by maximizing the same five-term reward
the trot was tuned against.

But read the *length* column. The trot rides out the full 500-step horizon; the
emergent gait covers more ground in about **249 steps and then falls**. So on total
**return** the trot still wins, 307 to 247: the learned gait is faster but **less
stable** — it has learned to sprint, not yet to sprint *and* stay up for the full ten
seconds. That asterisk is the lesson. Emergent does not mean robust. A reward that
pays for forward velocity gets forward velocity; keeping the body up for the whole
episode is a harder, slower-to-learn skill on top. (Notice too that the return peaks
around 45k and dips by 60k — RL curves are non-monotonic, and the "best" policy is
not always the last one.)

The numbers above are the measured seed-0 run. RL is noisy, so the exercises read the
signal across several seeds and never off a single run.

## Emergent is not robust

Here is the trap the emergence sets. A gait appeared, on its own, and it covers more
ground than the gait you hand-tuned in 2.4 — **+3.0 m against the scripted trot's
+2.15 m**, at nearly triple the forward speed. The tempting read is that SAC beat you.
Before you accept it, predict one number: over the full ten-second episode, which
gait earns more *return* — the learned sprint or the hand-written trot?

The measurement says the trot, **307 to 247**, and the reason is one column over.
Read the `len` field in the run above: the trot rides out all 500 steps; the emergent
gait covers its distance in about **249 steps and then falls**. It learned to sprint,
not to sprint *and* stay up. A reward that pays for forward velocity gets forward
velocity; standing for the whole horizon is a separate, harder skill it has not
bought yet. Emergent does not mean robust — the gait is real, and it is fragile, and
no single training curve told you which; only rolling it out to the horizon did.
(That fragility is exactly what the Scale Lab's extra samples, and chapter 2.7's
randomization, exist to buy back.)

## The Scale Lab

The free-tier run already shows the gait *emerge* and outrun the scripted trot on
distance. What it has not bought is stability — the fall at ~249 steps. The Scale Lab
runs the identical file with a bigger network and far more environment steps on a
GPU (`--hidden_dim 512 --total_steps 500000`), where the open question is whether
those extra samples turn the fast-but-fragile sprint into a gait that also rides the
full 500-step horizon — so that *return*, not just distance, clears the trot. It is
optional, and its numbers are a Scale-Lab measurement on paid hardware, never a
free-tier promise. The lesson of this chapter — that a gait emerges at all, from the
reward alone, on a laptop — stands on its own.

## Read the real thing

Our gait is real, and it is fragile. `walk.py`'s `train` loop hands SAC the
five-term reward and one quadruped body, takes one `update` step per env step, and
out of it a stride *emerges* — then that same policy outruns the scripted trot on
distance and falls at ~249 steps. Nobody scripted the gait; nobody made it robust
either. Closing that gap — emergent but fragile → production-robust — is the entire
job of a real locomotion loop, and you can read exactly how it is done.

Clone `leggedrobotics/legged_gym` at the pinned commit `8fa29ac` and open
`legged_gym/envs/base/legged_robot.py`. Scroll to the reward block (roughly lines
846–933). Where we shipped five terms, the base robot defines about nineteen
`_reward_*` methods. `_reward_tracking_lin_vel` and `_reward_tracking_ang_vel`
(~lines 903–908) reward following a *commanded* velocity — ours just pays for raw
forward speed. `_reward_feet_air_time` (~line 913) pays for long swing phases: it
literally *shapes the stride* rather than hoping one appears. And a wall of
penalties — `_reward_action_rate`, `_reward_torques`, `_reward_dof_acc`,
`_reward_orientation`, `_reward_collision` (~lines 854–879) — taxes jerky, expensive,
or unsafe motion, which is most of what makes a gait smooth enough to survive real
hardware. The weights that fuse nineteen terms into one scalar live in
`legged_gym/envs/base/legged_robot_config.py`, in `LeggedRobotCfg.rewards.scales`
(~line 165).

Two more machines in the same two files are what make the gait *transfer*. Domain
randomization: `_process_rigid_shape_props` (~line 546) jitters ground friction,
`_process_rigid_body_props` (~line 564) jitters base mass, and `_push_robots`
(~line 698) shoves the torso mid-episode — all switched on in the config's
`LeggedRobotCfg.domain_rand` block (~line 156). That is the full version of the one
jittered torso-mass line you previewed at reset. And curriculum:
`_update_terrain_curriculum` (~line 722) ramps terrain difficulty as the robot
proves it can cover ground, while `update_command_curriculum` (~line 741) widens the
commanded-velocity range only once tracking is good enough. The policy *earns* harder
problems instead of being handed them cold — the reason a production gait learns to
stay up, not just to sprint.

Two things you already built transfer *unchanged*, and spotting them is the payoff.
The action is a residual position target around a default stance — in
`LeggedRobotCfg.control` (~lines 120–131), `control_type = 'P'` and
`action_scale = 0.5`, so action zero stands, exactly our anchor. And the
episode-boundary bootstrap that separates a real fall from a time-limit truncation —
our `terminated`-not-`done` lesson — is what the sibling `rsl_rl` PPO loop consumes
on the training side.

**Read next:** `legged_gym/envs/base/legged_robot.py`, the `_reward_*` block (lines
~846–933), then the `scales`, `domain_rand`, and curriculum flags in
`legged_robot_config.py`. Pin the commit above; do not read against the moving
default branch. That is the difference between a gait that *emerges* and a gait that
*ships*.
