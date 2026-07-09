# 3.6: Full Circle: Run Your ch1.1 Policy in the Engine You Built

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## See it work

Two PushT sims, side by side, driven by the very same policy. On the left, MuJoCo:
the circular pusher noses up to the T-block, shoves it across the table, nudges it
square to the target, and the episode latches success: the behavior-cloning policy
you trained back in chapter 1.1, doing what it learned to do. On the right, the
*same* policy, the *same* starting position, reading the *same* ten numbers of
observation. But the sim underneath it is not MuJoCo. It is the physics engine you
built by hand over the last three chapters: a few hundred lines of numpy, semi-
implicit Euler, one distance constraint, a contact solve. And the policy still pushes.
It finds the block, it drives it toward the goal, it *mostly* works.

Mostly. Watch the two blocks, which start on top of each other, slide apart as the
episodes run: a little in position, a lot in *angle*. Watch the success counter on
the right land at roughly a third of the left's. The policy you have trusted since
chapter 0.1 is running inside an engine you can read every line of, and the gap
between the two sims is not a bug. It is the measurement, and it is the whole point of
this chapter, the last one in the "Build a Physics Engine" arc.

## The problem

Since chapter 0.1 you have called `mj_step` and believed what it returned. Chapters
3.3, 3.4, and 3.5 took that black box apart: **dynamics** (a state `(q, v, m)`
stepped by semi-implicit Euler), **joints** (a constraint `g(q) = 0` you solve for as
a Lagrange multiplier and hand back as a force), **contact** (the one-sided
complementarity you project at the velocity level). You built each piece and measured
exactly what it got wrong: energy drift, constraint drift, penetration.

This chapter spends it all at once. We take the three pieces and assemble **PushT**
(the environment from Phase 0, the one the ch1.1 policy learned in) inside your
engine, and then we run that policy. Not retrained. Reloaded: the exact TorchScript
checkpoint `bc.py` saved. If your engine is a faithful enough PushT, the policy should
still reach the goal; wherever your engine is a *simplified* PushT, the policy will
degrade in a way we can put a number on.

That number is a **sim-to-sim gap**, and it is worth being precise about why it
matters. In chapter 2.6 you measured a **sim-to-real** gap: a policy trained in one
world, dropped into another whose dynamics it never saw, degrading measurably. This is
the same phenomenon with the stakes turned down (both worlds are simulators, both are
deterministic, both are on your laptop), which makes it the ideal place to *study* the
thing. The mechanism is identical: unmodeled dynamics, and a policy that does not know
they are missing.

## Build

### Setup

```
[include-by-region: compare.py#setup]
```

Same house conventions as the rest of the arc: free-tier CPU, a `--smoke` mode with a
fixed short horizon so CI can diff two runs byte-for-byte, no GPU tier. The one new
input is the star of the show: a **trained ch1.1 policy** (`--policy`, a TorchScript
`bc_policy.ts.pt`). MuJoCo appears in this file for the first time in the arc, but
*only as the reference sim*: the ground truth we compare against. Your engine still
contains no MuJoCo. The knobs that carry the lesson are `--block_damp` and
`--pusher_mass`: two modeling choices that widen or close the gap.

### The policy, reloaded

```
[include-by-region: compare.py#policy]
```

Nothing about the policy changes. That is the entire idea. `bc.py` saved a TorchScript
copy that carries its own weights *and* its own normalization buffers, so we load it
with `torch.jit.load` and never need `bc.py`'s class definition on the path. Its
contract is exactly what it was in chapter 1.1: raw `obs[10]` in, raw `action[2]`
(pusher velocity) out. The world it acts in is the only variable. (For CI, which has no
trained checkpoint, a fresh seeded *untrained* policy stands in, enough to prove the
pipeline runs and is deterministic, honestly useless for transfer. The real numbers
need a real policy, and the file says so loudly if you hand it a stale one.)

### PushT, re-created in your engine

```
[include-by-region: compare.py#engine]
```

Here is the assembly, and here is where the honesty lives. PushT needs a pusher, a
T-block, and a target. The **target** is a fixed pose at the origin, a constant in the
observation. The **pusher** is a disk (a point mass with a radius) driven by an
idealized velocity servo toward the commanded action. The **T-block** is the
interesting one: it is **two point masses** (the bar center and the stem center,
0.06 m apart in the body frame) held rigid by *one distance constraint from chapter
3.4*. It has no yaw variable; its orientation is **emergent**, read straight off the
line between the two masses (`block_yaw`: one `arctan2` and the `+ pi/2` that inverts
the body frame). Its rotation is emergent too: push the bar mass off the block's
center of mass and the whole rigid dumbbell *turns*. That is the only reason a
frictionless normal contact (chapter 3.5's `solve_contacts`, reused wholesale on point
masses) can rotate a T at all.

Read the simplifications in the region comment and take them seriously, because they
are the sources of everything we are about to measure. Two point masses are not the
real welded two-box body: the inertia is approximate, and the policy gets **less
rotational leverage at the bar tips** than MuJoCo hands it. There is **no friction**:
PushT's draggy planar surface is faked as a viscous block drag. The pusher is an
*idealized* servo, and there is no continuous collision. None of these is a mistake.
Each is a choice, and the sum of the choices is the gap. A version of this engine that
matched MuJoCo exactly would not be a triumph; it would be a bug, because it is
demonstrably the cruder model.

The observation assembly is what lets the policy plug in at all: the same ten numbers,
in the same order, that `pusht_env` produces and ch1.1 trained on. Get that contract
right and a policy trained in one engine runs, unmodified, in another.

### Running it in both sims

```
[include-by-region: compare.py#rollouts]
```

Three rollouts per episode, all seeded *identically* so both sims start from the same
block and pusher, which is why we mirror `pusht_env`'s exact reset sampling, draw for
draw. **MuJoCo closed-loop** gives the ground-truth success and records the exact
action sequence the policy chose. **Your-engine closed-loop** gives the transfer
number: the same policy, picking its own states in your world. And **your-engine
open-loop** replays MuJoCo's *exact* actions from the shared start, with no policy
feedback in the loop to quietly correct the drift, so we can watch the block poses
diverge from pure dynamics alone. The first comparison answers "does it still
succeed?"; the second isolates "how different are the two engines, really?"

## Run it

```
python curriculum/phase3_advanced/ch3.6_compare/compare.py \
    --policy outputs/ch1.1-bc/bc_policy.ts.pt --seed 0
```

<!-- wall-clock table renders from wallclock.csv (measured: cpu-laptop 0.12 min) -->

Note what you are *not* waiting for: the reload runs two small sims over 50 episodes in
about seven seconds on a laptop. The ~4-minute training cost was ch1.1's, paid once,
chapters ago. If you have no trained policy on disk yet, the file's header prints the
two commands that make one.

Fifty episodes, the trained ch1.1 policy, seed 0:

| metric                                      | MuJoCo (ground truth) | your engine  |
| ------------------------------------------- | --------------------- | ------------ |
| BC success rate                             | **0.62**              | **0.20**     |
| mean block-position divergence (open-loop)  | n/a                   | 0.082 m      |
| mean block-**angle** divergence (open-loop) | n/a                   | **0.94 rad** |

Read it honestly. The policy **transfers, and it degrades**: it keeps about a third of
its MuJoCo success in the engine you built. The ground-truth side is real, and here is
the cross-check that proves it: this file loads ch1.1's *own* checkpoint and draws the
same held-out seeds, so its MuJoCo rollout reproduces ch1.1's reference eval, 31/50 =
0.62, *exactly*. It is the same policy; only the physics changed. The trajectories
start identical and pull apart, and the **angle** gap, 0.94 rad, dwarfs the position
gap. That is diagnostic, not incidental: a two-point-mass dumbbell rotates under a push
with far less authority than MuJoCo's shaped, welded body, so the policy aligns the
block much worse in your world than in the one it trained in. It can still shove the
block roughly toward the target; it struggles to square it up.

One caveat on the numbers, because this book does not oversell a decimal. The engine
success rate (0.20) is a coarse 0/1 count over 50 episodes, and it is noisy. Do not
over-read the exact value. The **divergence** is the smooth, seed-robust quantity. The
result to carry away is its shape: *lower success, and trajectories that diverge, most
of all in angle*, not a single digit.

### The gap is a knob

The best way to believe a modeling gap is real is to move it. `--block_damp` is your
engine's stand-in for the block's surface friction; MuJoCo's tee is heavily damped and
barely coasts. Turn your drag up toward that, and the two sims agree better,
monotonically:

| `--block_damp` | mean position divergence | mean angle divergence |
| -------------- | ------------------------ | --------------------- |
| 2              | 0.123 m                  | 1.204 rad             |
| 60             | **0.076 m**              | **0.877 rad**         |

More than a **1.6× reduction** in the position gap, from nothing but making your block as
sluggish as MuJoCo's. Here is one of the modeling choices with a slider on it.
(Exercise 3 has you move that slider and read the divergence off it yourself.)

## Matching MuJoCo would be a bug

It is tempting to read the gap as a defect: 0.62 in MuJoCo, 0.20 in your engine, so
your engine is broken and the job is to drive the distance to zero. Predict what
that would take. You would have to give the two point masses a shaped rigid body's
inertia, add a real Coulomb friction cone, and resolve contact continuously, which
is to say, rebuild the very subsystems this arc scoped out on purpose. An engine that
matched MuJoCo exactly would not be a triumph; it would be a *bug*, a sign you had
smuggled the cruder model back into agreement with the finer one it is demonstrably a
simplification of.

So the gap is not the error term: it *is* the measurement, and it points at where
the model is thin. The **angle divergence (0.94 rad) dwarfing the position gap** is
not noise to tune away; it is the two-mass dumbbell telling you, in radians, exactly
how much rotational authority it lacks against MuJoCo's shaped, welded body. You can
*move* the gap (`--block_damp` narrows it monotonically toward MuJoCo's sluggish
tee), but the honest run never tunes the knobs until the two success rates match.
That would not be closing the gap; it would be hiding it. The number worth keeping is
the one that says *it works, and here is precisely how much it does not.*

## Why this is the sim-to-real story in miniature

Sit with what just happened, because it is the lesson the whole arc was building
toward. You trained a policy in one simulator. You built a *second*, cruder simulator
from scratch. You ran the policy in it, and it worked worse: in a way you could
measure, partly explain (the missing rotational authority), and partly dial back (the
friction knob). Now replace "your from-scratch engine" with "a real robot" and you have
chapter 2.6, and you have the central problem of deploying learned policies: the world
you train in is always a model, the world you run in always has dynamics the model left
out, and the transfer gap is not something you eliminate: it is something you
**measure, respect, and shrink on purpose**. You have now felt that from both ends: as
the person whose policy degrades, and as the person who built the flawed physics that
degraded it.

## The honest limits

Say plainly what this engine and this comparison do *not* claim:

- Your PushT engine is **deliberately simplified**: two point masses instead of a shaped
  rigid body, frictionless normal contact with friction faked as drag, an idealized
  pusher, no continuous collision. It is a *worse* PushT than MuJoCo on purpose. The
  point is the gap, not a competitive physics engine.
- The transfer is **partial, and the success number is noisy**. 0.20 is one seed's coarse
  count; the divergence is the number to trust. We do not sell a clean transfer.
- The comparison uses MuJoCo as **ground truth**, which it is not (MuJoCo has its own
  sim-to-real gap). Here it is only the more-faithful reference. The arc's "read the
  real thing" segment points at what MuJoCo itself leaves out: the Coulomb friction, the
  friction cone, and the shaped rigid body your two-mass block is a stripped-down
  approximation of.

## Read the real thing

The honest limits above name the gaps in words. Here they are in MuJoCo's own C.
`meta.yaml` pins `google-deepmind/mujoco` at tag `3.10.0` (the exact version the rest
of the course calls), so read these three files against the `step_engine` you just wrote.

**Your `mj_step`, in the `engine` region.** Everything the last four chapters built lands
in one function: `step_engine` in `compare.py`. Per physics tick it does three things in
order: assemble forces (ch3.4's `link_force` holding the two masses rigid, plus the faked
block drag), take the ch3.3 semi-implicit velocity-then-position step, and resolve contact
with ch3.5's `solve_contacts` (`detect_pusher_contacts` → project the normal impulses).
Forces, integrate, constraints, `q += DT * v`. That is a real `mj_step`, small enough to
read in one sitting.

**MuJoCo's `mj_step`, in `src/engine/engine_forward.c`.** The real one has the same three
moves and the same shape. `mj_step` runs `mj_forward` and then an integrator (`mj_Euler` by
default; `mj_RungeKutta` / `mj_implicit` optional). `mj_forward` fans out through
`mj_forwardSkip` into an ordered pipeline you will recognize: `mj_fwdPosition` →
`mj_fwdVelocity` → `mj_fwdActuation` → `mj_fwdAcceleration` → `mj_fwdConstraint`, then
integrate. Your one line of force assembly is their whole *position* stage: inside
`mj_fwdPosition` (same file) the order is `mj_fwdKinematics`, build and factor the mass
matrix (`mj_makeM` / `mj_factorM`), `mj_collision`, `mj_makeConstraint`,
`mj_projectConstraint`. Same skeleton (kinematics, contact, constraints, integrate), but
every rib of it is a real subsystem.

**What they add, and why the transfer drops.** Two ribs your engine stubs are exactly where
the 0.62 → 0.20 gap lives. **Contact:** `mj_collision` in
`src/engine/engine_collision_driver.c` runs broadphase then narrowphase over the *shaped*
geoms of the actual T, producing contacts anywhere on the body; your `detect_pusher_contacts`
tests two disks. **Friction:** `mj_makeConstraint` / `mj_projectConstraint` in
`src/engine/engine_core_constraint.c` build and solve a real Coulomb friction cone (the
`mjCNSTR_CONTACT_PYRAMIDAL` and `mjCNSTR_CONTACT_ELLIPTIC` contact types, tangential forces
clamped by `mj_assignFriction`) while your `solve_contacts` is frictionless, normal-only.
Those two omissions are not incidental to the result; they *are* it. A shaped body caught
along a real contact patch, with friction resisting slip, rotates the T with an authority a
two-disk frictionless push cannot, which is precisely why the angle divergence (0.94 rad)
dwarfs the position gap. The gap you measured is the price of the pipeline you left out,
printed in radians.

None of this makes your engine wrong. It makes it a faithful *minimum* (the smallest thing
that steps PushT and runs a real policy), and it makes the missing subsystems legible. This
is the reason production simulators exist: not because `mj_step` is doing something magic,
but because it is doing the collision and the friction cone you can now name.

**Read next:** open `src/engine/engine_forward.c` and find `mj_step`. Follow its call into
`mj_forward` → `mj_fwdPosition`, and there (in `mj_collision` and `mj_makeConstraint`) sit
the shaped-body contact and the friction cone your two-mass block approximates. Read those
two, and you have read the gap.

## What you built, and where the arc closes

You closed the circle. Since chapter 0.1 you trusted `mj_step`; across 3.3–3.5 you
rebuilt its insides (dynamics, constraints, contact) and measured what each cost; and
here you re-created a whole task, PushT, out of those pieces and ran a real policy in
it. The behavior-cloning agent from the very first chapter of Phase 1 executed inside
the physics engine you wrote by hand, and the gap between your engine and MuJoCo turned
out to be the same kind of gap as the one between MuJoCo and reality, now with a number
on it, and a knob.

If you wanted to close that gap for real, you already know the three moves the honest
limits named: give the block a shaped rigid body instead of two point masses, add real
planar friction instead of faking it as drag, and resolve contact continuously. Each
would narrow the divergence, and each is a chapter's worth of work, which is exactly
why MuJoCo is a black box worth trusting, and exactly what it took you an arc to earn
the right to say.

The print table below is the deliverable: two success rates and two divergences that
say, together, *it works, and here is exactly how much it does not.*

```
[include-by-region: compare.py#report]
```
