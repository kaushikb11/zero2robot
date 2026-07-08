# 3.9 — Plan Through Your Engine: Sampling-Based MPC (CEM / MPPI)

## Every controller you built until now learned. This one doesn't.

Behavior cloning copied a demonstrator. PPO and SAC learned a policy the hard
way — thousands of episodes of acting, failing, and nudging weights until a
network mapped states to good actions. Different as they are, they share one
assumption so deep you probably never questioned it: that *controlling* a robot
means first *learning* a policy. Pour experience in, read actions out.

This chapter throws the network away.

You spent chapters 3.3 through 3.6 prying open `mj_step` — integrators,
joints as constraints you solve for, contact as a complementarity you project —
until the simulator stopped being a black box and became an *engine* you
understand. Step back and notice what an engine actually is: a function that
takes a state and an action and hands you the next state. That is the one thing
you need to *plan*. If you can ask "what happens if I do this?" and get an
honest answer, you don't have to learn what to do — you can *search* for it,
live, at every control step. That is Model Predictive Control, and it is how a
large fraction of real robots are actually driven.

## The task: swing a pole up with no demonstrator and no reward to learn from

Cartpole again, but harder, and pointed the other way. In chapter 2.1 the pole
started near upright and PPO learned to *balance* it. Here the pole starts
hanging straight **down**, and the only actuator pushes the cart sideways. You
cannot torque the pole up directly — the system is underactuated. The only way
up is to pump: rock the cart to build the pole's energy, let it swing higher
each time, and catch it at the top. It is the canonical demonstration that
planning beats reacting, because acting greedily — always reducing your cost
*right now* — never lets the pole fall the "wrong" way long enough to gain the
momentum that brings it up.

There is no training run in this chapter. No demonstrator, no learned reward, no
replay buffer. Just a model and a search. The pole swings up anyway.

## The idea, in four steps

At every control step, from the state you are actually in:

1. **Sample** `N` candidate action sequences, each `H` steps long — a spray of
   possible short-term plans.
2. **Roll** each one forward through the model and **score** it with a cost you
   simply *write down* (upright pole, centered cart, settled motion). No network
   estimates the value; the engine tells you exactly what each plan leads to.
3. **Refine** the distribution you're sampling from toward the plans that scored
   well, and repeat step 1–2 a couple of times.
4. **Execute only the first action** of the best plan, let the real world advance
   one tick, and **re-plan from where you actually landed.**

Step 4 is the "predictive" and the "receding horizon" in Model Predictive
Control: you always re-decide from the true current state, throwing away the
tail of last step's plan, because no model is perfect and the world drifts. You
plan far, commit little, and plan again.

## The model is a second copy of the engine

MPC's whole premise is that you *have* a model — something you can fast-forward
and, crucially, *rewind*, so you can try one plan, snap back to the start, and
try the next. We keep the real world and the model as two separate objects on
purpose. The planner may only touch `sim`; the true state only ever advances
through `env`. Saving and restoring the model's state is just reading and
writing MuJoCo's `qpos`/`qvel` — the full state the engine needs to continue:

```
[include-by-region: mpc.py#model]
```

The cost function *is* the task. There is no learned objective anywhere: you
state what "good" means — pole upright, cart centered, motion settled — and the
search does the rest. Read `state_cost` and notice that this single, hand-written
function scores both the imagined rollouts and the realized step. Writing the
objective down instead of learning it is the whole trade this chapter makes.

## Same loop, one idea differs: CEM vs MPPI

Here is the entire planner. Sample, roll, score — then update the sampling mean
toward the good plans. The two update rules in the course's "same file, one idea
differs" tradition are about fifteen lines apart:

```
[include-by-region: mpc.py#planner]
```

- **CEM** (the cross-entropy method) draws a hard line: sort by cost, keep the
  **elite** fraction — the few lowest-cost plans — and refit the Gaussian to
  them. The mean marches toward the elites; the std shrinks as they agree, so
  the search focuses in.
- **MPPI** (model-predictive path integral) refuses to throw anything away.
  Every sample gets a weight `exp(-cost / temperature)` — a softmax over negative
  cost — and the new mean is the weighted average of *all* of them. Good plans
  dominate smoothly; the temperature sets how sharply.

They are two answers to one question — *how much should a plan's score move the
mean?* — and at the limit they meet: CEM with a single elite keeps the one best
plan, and MPPI as the temperature goes to zero collapses to that same plan.
Exercise 3 makes you fill both updates in and prove they agree there.

## Run it

```
python curriculum/phase3_advanced/ch3.9_mpc/mpc.py --seed 0
python .../mpc.py --seed 0 --method mppi
```

The reference numbers are from `--device cpu`, the configuration this book can
promise is bitwise-deterministic under `--seed`: numpy's sampling and CPU
`mj_step` are both deterministic, so the same seed run twice matches byte for
byte. Every run also plays a no-plan baseline — uniform-random actions from the
same start — so the payoff is right there in the output:

```
                        mean cost  upright frac
MPPI                        1.210          1.00
random (no plan)            1.988          0.00
```

<!-- wall-clock table renders from wallclock.csv (ch3.9-mpc) -->

`upright_frac` is the fraction of the settle window (the last quarter of the
episode) the pole spends within about 25° of straight up. MPC swings up and
*holds* — `1.00` — while random flails at `0.00`, and the smooth cost sits far
below the baseline. You solved an underactuated control problem with **zero
learning**: no policy, no training, just a model and a search. And it reproduces
— seeds 0, 1, 2 all swing up, for both CEM and MPPI. Open the recording to watch
the upright-cos curve climb from −1 to +1:

```
rerun outputs/ch3.9-mpc/mpc.rrd
```

## Break it: a plan that can't see far enough

The claim is a *mechanism* claim, not a magic one, so measure its edge rather
than trust it. `--break horizon` drops the look-ahead from 25 steps to 3 —
same model, same sampler, same cost, just a myopic plan:

```
python .../mpc.py --seed 0 --break horizon
```

The pole never comes up. `upright_frac` collapses to `0.00`, seed after seed. A
three-step plan cannot see that letting the pole fall further *now* is what buys
the swing *later*, so it does the locally sensible thing forever and gets
nowhere — exactly the greedy failure swing-up is designed to expose. `--break
samples` (3 samples instead of 64) fails the same way for the other reason: too
few tries to stumble on an energy-pumping sequence at all. Look-ahead and search
width are not free knobs you can shrink; they are what make planning work.

## The honest ceiling — and the trade against learning

MPC looks like a free lunch here, and it is not. It cost you two things.

**It needs a model.** In this chapter the model *is* the world — a perfect copy
of the simulator — which is the best case that exists, and the reason the swing
comes up so cleanly. Chapter 3.6 already showed you the other side: run a policy
in an engine that only *approximates* the true dynamics and it degrades, because
it is optimizing an imagined trajectory the real world will not follow. MPC has
exactly that exposure. A plan is only as good as the model it is dreamed
through; a wrong model plans confidently toward the wrong place.

**It needs compute at every step.** MPC does no work up front and then re-solves
an optimization *every single tick* — `N × H` model rollouts, times a few
refinement iterations, before it can even move. That is the mirror image of a
learned policy, which pays a huge one-time training cost and then acts in
microseconds with no model at all. This is the real axis this chapter adds to
the course: **learn once and react fast, or carry a model and plan every step.**
Neither dominates. The frontier systems you'll meet later blend them — a learned
model to plan through, a learned policy to warm-start the search — but you can't
understand those hybrids until you've felt both pure ends, and this is the end
the physics-engine arc was quietly building toward all along.

## Read the real thing

`google-deepmind/mujoco_mpc` (MJPC) is this file grown up. It is an interactive
controller that plans through `mj_step` — the same engine you plan through here —
fast enough to drive humanoids and manipulators in real time. Open its
`mjpc/planners/`: the **cross_entropy** planner is our CEM, the **sampling**
(predictive sampling) and MPPI-style planners are our weighted-average update,
and each task's cost is defined exactly the way `state_cost` is — a residual you
write down, not a reward you learn. The one thing MJPC has that we don't is
*throughput*: it rolls its candidate plans through the model across many CPU
threads at once, which is the honest answer to this chapter's ceiling — MPC's
price is compute-per-step, and you pay it with cores. Read our whole planner
first; then read theirs and see the same idea running a hundred times faster.

## Exercises

Three, in `exercises/`. The first (`ex1_predict_planning`) is the misconception,
head on: predict whether a search through a model can control the cartpole with
no learning at all, then run it and watch it swing up. The second
(`ex2_predict_break`) hands you the `--break horizon` failure to generate
yourself — predict what a myopic plan does, then measure the pole refusing to
rise. The third (`ex3_completion_update`) blanks out both the CEM elite-refit and
the MPPI weighted-average and asks you to fill them in and prove they agree at the
limit — the fifteen-line diff that is the entire difference between the two
methods.

## What's next

You now hold both pure ends of control: a policy that learns once and reacts, and
a planner that carries a model and searches every step. The physics-engine arc
closes here — you opened `mj_step`, then used the thing you understood to plan
through it. What's missing is a model you didn't get for free: real robots don't
ship with a perfect simulator of themselves. The world-model chapters (3.1–3.2)
learned dynamics from data and then acted *in imagination*; put that together
with what you just built and you get the modern recipe — learn a model, then plan
through it — which is where the frontier of this field actually lives.
