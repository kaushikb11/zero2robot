# 3.4: Build a Physics Engine II: Joints & Constraints

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## See it work

A double pendulum, released from horizontal, allowed to fall. Two of them,
actually, drawn on top of each other in the same window. They start from the
same place and obey the same gravity. One swings the way a real double pendulum
swings, wild, beautiful, never repeating, but its two rods stay exactly the
length they started. The other swings almost the same way for a few seconds, and
then its rods begin, visibly, to *stretch*. By the end of the run its lower arm
is a third longer than it should be. It is coming apart.

Nothing about the two pendulums differs except one term in the force. The
stretching one is computing the constraint (the "these rods do not change
length" rule) the naive way. The other adds a single stabilizing correction.
The stretch is not a bug in the physics. It is a constraint being *enforced
slightly wrong*, the same slightly-wrong way every step, until the error is a
third of the arm.

That stretching pendulum is the subject of this chapter.

## The problem

In chapter 3.3 you built the inside of `mj_step` for a body that could go
anywhere the force sent it: state, a force law, three integrators, and energy
drift as the number that told you how wrong each integrator was. Bodies fell
freely.

A robot is the opposite of free. Its links are bolted together at joints. A
pendulum bob may only move on a circle a fixed distance from its pivot; the
second bob of a double pendulum may only move on a circle around the first. That
"may only" is a **constraint**, a rule the state must satisfy, written
`g(q) = 0`. For a distance joint, `g` measures how far the link's length is from
its rest length; `g = 0` means the rod is exactly right.

Here is the question the chapter turns on: gravity pulls the bob straight down,
but the rod will not let it go straight down, so what force does the rod apply
instead, and how do you compute it? You cannot write it down in advance the way
you wrote gravity, because it depends on gravity, on the current velocity, on
every other constraint in the chain, all at once. The constraint force is not a
law you know. It is a value you **solve for**. Learning to solve for it, and
then watching it drift and fighting the drift, is the whole chapter, in about
350 lines of numpy on top of last chapter's engine.

## Build

### Setup

```
[include-by-region: constraints.py#setup]
```

Same house conventions as ch3.3: free-tier defaults first, a `--smoke` mode with
a fixed step count so CI can diff two runs byte-for-byte, no GPU tier, no
training.

<!-- wall-clock table renders from wallclock.csv (measured: cpu-laptop 0.01 min) -->

The whole run is about 0.01 minutes: a chain of point masses and one small
linear solve per step is microseconds of arithmetic; there is nothing to wait
for. One flag does more work than its size suggests: `--seed` nudges exactly one
knob, the launch angle, by at most 0.05 radians, the same way ch3.3's seed
nudged the orbit speed. Because the double pendulum is chaotic, that hair's-width
nudge diverges fast, yet each seed's run is still bitwise reproducible. Hold onto
that; it is the whole of exercise 3. The other knobs carry the lesson directly:
`--stabilization` (the comparison), `--baumgarte` (the gain you will learn to
distrust), and `--dt` (shrink it and watch the whole thing converge).

### The systems

Last chapter a "system" was one body. Now it is a **chain**: N point masses,
each tied to its neighbor by a distance lock. We keep ch3.3's `(q, v, m)` shape
and simply stack it: `q` and `v` become `(N, 3)`, `m` becomes `(N, 1)` so
`force / m` still broadcasts per particle. The genuinely new fields are the
constraint *topology*: `pairs`, which particle is roped to which, and `lengths`,
each link's rest length.

```
[include-by-region: constraints.py#systems]
```

Three systems, all the same code with a different link count: a **pendulum**
(one link), a **double** pendulum (two, the chaotic classic), and a **triple**.
They start stretched out along the x-axis, horizontal, which is the position of
maximum gravitational torque, the dramatic drop. Gravity points along `-y`, so
the whole thing swings in the x-y plane and reads as a flat `(x, y)` curve in
rerun, exactly like ch3.3's orbit. `external_force` is just gravity, `m * g`,
returned in the same `(N, 3)` shape ch3.3's `force` returned, because to the
integrator, that is all a force is.

### The constraint force

This is the new idea, and it is worth reading slowly. A distance constraint is

```
g_k(q) = ½(|d_k|² − L_k²) = 0
```

where `d_k` is the vector along link `k`. We want the constraint force that keeps
this true. The trick is to differentiate `g = 0` with respect to time, twice.
Once gives the velocity condition `ġ = J v = 0` (the link's length is not
changing), where `J = ∂g/∂q` is the **constraint Jacobian**, and for a distance
constraint `J` is just `d` written into the moving particle's slots. Twice gives
the acceleration condition:

```
g̈ = J a + J̇ v = 0
```

Now substitute the equation of motion `a = M⁻¹(f_ext + Jᵀλ)`: the constraint
force is `Jᵀλ`, where `λ` is an unknown vector of **Lagrange multipliers**, one
per constraint. The accelerations drop out and you are left with a small linear
system in `λ` alone:

```
(J M⁻¹ Jᵀ) λ = −(J M⁻¹ f_ext + J̇ v)
```

Solve it (the matrix is `C × C` for `C` constraints, symmetric positive
definite, two or three rows for our chains) and you have the multipliers. Apply
`Jᵀλ` as an added force, and the integrator does the rest. That last sentence is
the entire seam with chapter 3.3: **a constraint is a force you add**, and the
integrators from last chapter never learn that `force` now hides a linear solve.

```
[include-by-region: constraints.py#constraints]
```

The `J̇ v` term is the one piece of calculus you cannot skip: for a distance
constraint it works out to `|ḋ|²`, the squared relative velocity across the link.
Drop it and you are solving the wrong equation. (Exercise 2 makes you supply it.)

### The integrators, unchanged

```
[include-by-region: constraints.py#integrators]
```

These are ch3.3's three integrators, copied in verbatim: the repetition is the
point. They take `(q, v)`, a mass, and a `force`, and step forward. They do not
know and do not care that `force` now solves a linear system on every call (RK4
solves it four times per step, once at each stage, which is exactly correct).
Semi-implicit Euler is the sane default for constraints, so it is the default
here.

### Simulate and measure

```
[include-by-region: constraints.py#simulate]
```

The loop is ch3.3's, and it records two honesty numbers at every step. The first
is the **constraint violation**: the worst link-length error over the chain,
`max_k | |d_k| − L_k |`. Zero means every rod is exactly its rest length; a
growing value is the pendulum literally stretching. This is the chapter's headline
metric, and it plays exactly the role energy drift played last chapter. The
second is the total **energy**, which a correct constraint force leaves alone:
constraint forces act along the rod, perpendicular to the motion they permit, so
they do no work. Energy is a second, independent check that the solve is right.

## The measurement

The report region is the driver. It runs the chosen stabilization mode (or, by
default, both), records the honesty curves to rerun, and prints the comparison
that is this chapter's deliverable.

```
[include-by-region: constraints.py#report]
```

Run the default double pendulum for twenty seconds of sim time, once with no
stabilization and once with it, and read the worst link-length error:

```
python constraints.py --seed 0
```

| stabilization | worst link-length error | final link error | energy drift |
| ------------- | ----------------------- | ---------------- | ------------ |
| none          | **0.384**               | 0.384            | 0.43         |
| Baumgarte     | 0.023                   | 0.0014           | 0.32         |

The naive solve enforced `g̈ = 0` (it froze the *acceleration* of the length
error) but it never enforced `g = 0` itself. So the length error, tiny at first,
has nowhere to be pushed back to. Each step's small discretization error in `g`
survives into the next, and they pile up: **0.384**, meaning the worst rod is 38%
off its length. The worst error equals the final error, which means it never came
back: it is still climbing when the run ends. This is the stretching pendulum
from the opening.

### Fixing the drift: Baumgarte stabilization

The fix is due to Baumgarte (1972) and is one line. Instead of aiming for
`g̈ = 0`, aim for a `g` that decays back to zero like a critically damped spring:

```
g̈ + 2ω ġ + ω² g = 0
```

The `ω² g` term pushes an existing position error back toward zero; the `2ω ġ`
term damps it so it does not overshoot. Fold those two terms into the
right-hand side of the same linear solve and the worst error drops from 0.384 to
**0.023** (sixteen times smaller) and the *final* error settles to 0.0014 and
stays there. The rods hold. That factor is seed-robust: across seeds 0/1/2 on the
pendulum, double, and triple, the naive error runs 5× to 17× the stabilized one.
The ordering is the rock: trust it before you trust any single exact number.

### Is the math even right?

A wrong constraint solver can look plausible and still be nonsense, so verify it.
Run the single pendulum under RK4 with no stabilization at all:

```
python constraints.py --system pendulum --integrator rk4 --stabilization none
```

The worst link error is 8.8e-6 and the energy drift is 1.2e-5: the
naive solve, given an accurate enough integrator, keeps the rod to five decimal
places and conserves energy to five. Both errors shrink as you shrink `dt`
(semi-implicit roughly first-order, RK4 far steeper). That convergence is the
proof: the drift is discretization error going to zero the way it should, not a
bug. The Jacobian and the multipliers are correct. What drifts in the default run
is not the *physics*; it is cheap first-order integration of a stiff system, and
that is precisely the regime a real-time engine lives in.

## The honest cost, and why MuJoCo looks the way it does

Baumgarte is not free, and the honesty of this chapter is in saying so. Look at
the energy column: stabilization holds the rod but its feedback force does a
little work, so energy conservation is only *slightly* better, not perfect. And
`ω` is a knob you had to pick: 20 worked; too large and `dt·ω` approaches 1 and
the correction itself goes unstable; too small and the drift creeps back. You
have traded a drift problem for a *tuning* problem, and hidden a small energy leak
inside the fix.

This is the exact pain that shaped MuJoCo. Rather than enforce hard equality
constraints and paper over the drift with a hand-tuned Baumgarte gain, MuJoCo
uses **soft** constraints: it builds the same `J M⁻¹ Jᵀ` matrix you just built,
but *regularizes* it and solves a convex optimization whose stiffness and damping
are principled, physical parameters instead of a magic `ω`. Every strange-looking
choice in MuJoCo's constraint documentation (softness, the reference
acceleration, the impedance) is an answer to a problem you just felt in your own
hands. That is the "read the real thing" segment for this chapter.

One honest boundary on what you built, since a robotics reader will ask. We solved
the constraint at the **acceleration** level (enforce `g̈`, hand the force to the
integrator) because it keeps the Lagrange-multiplier spine bare and reuses ch3.3's
integrators untouched. Real-time engines more often solve at the **velocity** level,
trading multipliers for impulses, and articulated mechanisms are usually handled by
Featherstone's articulated-body algorithm, which scales linearly in the number of
links instead of paying for a dense `J M⁻¹ Jᵀ` solve. Same physics, different
bookkeeping; the acceleration-level view is the one that makes the constraint force
visible as a force, which is the thing this chapter is for.

## Break It

Two seeds, 0 and 1, differing by at most 0.05 radians at launch. Run the double
pendulum at each and look at where the tip finishes after twenty seconds: the two
tips end up **1.1 units apart**, an order-of-one difference from a hair's-width
start. That is chaos, and this little engine reproduces it faithfully.

Now run seed 0 twice. The two runs are **bitwise identical**, down to the last
bit of the last float. Both things are true at once: the double pendulum is
*deterministic* (same seed, same bytes, always) and *unpredictable* (a
negligible change in the start becomes a total change in the finish). Those are
not the same word, and conflating them is one of the most common confusions in
all of simulation. Determinism is a property of the arithmetic; predictability is
a property of the system. Exercise 3 makes you watch both at once.

## Read the real thing

The chapter kept promising this, so here it is: the exact `J M⁻¹ Jᵀ` you built by
hand, sitting inside a production solver, doing everything you did and four things
you skipped.

Start with your own file. The whole idea lives in `constraints.py`'s
`#constraints` region: `constraint_force` assembles `J M⁻¹ Jᵀ`, forms the
right-hand side `−(J M⁻¹ f_ext + J̇v)`, folds in the Baumgarte feedback
`2ω ġ + ω² g`, and hands the result to one `np.linalg.solve`. One dense symmetric
solve, one hard equality per rod, one hand-tuned `ω`. That is the teaching skeleton.

Now the real one. The course pins `google-deepmind/mujoco` at tag `3.10.0`, and
the constraint system is built in `src/engine/engine_core_constraint.c` (verified
at that tag, ~2869 lines). `mj_makeConstraint` is the top-level driver;
`mj_projectConstraint` forms the same `J M⁻¹ Jᵀ` you did, storing it as `efc_AR`.
But look at what wraps it. `mj_makeImpedance` fills `efc_R` (a diagonal
*regularizer* added to the diagonal, so the system actually solved is
`J M⁻¹ Jᵀ + R`, not the bare matrix) and `mj_referenceConstraint` computes
`efc_aref = −B·vel − K·(pos − margin)`. Read that formula against your Baumgarte
term. It is the same shape (a stiffness on position error, a damping on velocity
error) except `K` and `B` are not a magic `ω` you guessed; they come from the
physical `solref`/`solimp` parameters (`mj_assignRef`, `mj_assignImp`), so the
Baumgarte gain becomes a time constant and damping ratio you can actually reason about.

One honesty note on the map. `meta.yaml`'s focus reads "engine_core_constraint /
the PGS-or-CG solver," but at 3.10.0 those are two files: the constraint system is
*built* in `engine_core_constraint.c`, while the solve *iterations* live next door
in `src/engine/engine_solver.c`: `mj_solPGS`, `mj_solCG`, and `mj_solNewton`
(Newton is MuJoCo's default). Same pipeline, two stops.

What they add, and why. Your one dense `np.linalg.solve` becomes three iterative
solvers that scale to thousands of constraints without ever forming a dense
matrix. Your one constraint type (a rod) becomes the full model: equality welds,
joint limits, dry friction, and the one-sided contacts of chapter 3.5, all packed
into the same `efc_*` arrays. And the `+ R` regularization is this chapter's last
section made real: because the matrix is regularized, the constraint is *soft*,
the solve is a convex optimization that always has an answer, and drift is bounded
by physical stiffness you set rather than a gain you tune until it stops exploding.
You built the spine; they armored it.

Read next, in order: `src/engine/engine_core_constraint.c`, to find
`mj_referenceConstraint` and read `efc_aref` against your Baumgarte line; then
`mj_makeImpedance` for where `efc_R` and the soft `K`, `B` come from; then
`src/engine/engine_solver.c`'s `mj_solNewton` to watch the regularized
`J M⁻¹ Jᵀ + R` finally get solved.

## What you built, and what comes next

You built joints. Given a rule the state must obey, you can now solve for the
force that enforces it, apply it as one more term in ch3.3's force law, and
measure (honestly) how well it holds. You met the drift that every hard
constraint suffers and the stabilization that tames it, along with the bill that
stabilization quietly runs up.

Chapter 3.5 takes the last and hardest step: **contact**. A joint is a constraint
that is always active: the rod is always exactly length L. A contact is a
constraint that switches on and off: the foot pushes on the floor only when it
is *touching* the floor, and never pulls. That one-sidedness (an inequality, not
an equality) is where physics engines get genuinely hard, and where the sim
artifacts you have watched since chapter 0.1 finally get explained.
