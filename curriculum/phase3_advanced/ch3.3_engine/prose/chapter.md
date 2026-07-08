# 3.3: Build a Physics Engine I: Unconstrained Dynamics

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## See it work

Three orbits, same planet, same starting push. Watch them for twenty seconds of
simulated time. One traces a clean ellipse and keeps tracing it. One traces the
same ellipse but with a hairline wobble you have to squint to see. And one
spirals visibly outward, orbit after orbit, climbing away from the planet it is
supposed to be bound to.

Nothing about the planet changed between the three. The force law is identical.
The only difference is the arithmetic each one uses to turn "force right now"
into "position a hundredth of a second from now": the integrator. The spiral
is not a bug in the physics. It is the physics being computed slightly wrong,
the same slightly-wrong way every step, until the error is the whole picture.

That third orbit is the subject of this chapter.

## The problem

Since chapter 0.1 you have called `mj_step` and trusted it. Write `data.ctrl`,
call the function, read the new `qpos`. Thirty-three chapters of robots stand on
top of that one call, and you have never once looked inside it.

Here is the question that separates using a simulator from understanding one:
when `mj_step` advances a floating body by one timestep, what arithmetic does it
actually do, and what does that arithmetic get *wrong*? Because it does get
something wrong. Every numerical integrator does. A simulator is a machine for
being wrong in a controlled, measurable way, and the entire craft of physics
simulation is choosing *which* way to be wrong.

So we build the inside of `mj_step` (the unconstrained part, no contacts yet)
from scratch in numpy. No MuJoCo in the dynamics; that would be circular. Just
state, a force law, and the integrator, in about 270 lines. Then we measure the
wrongness with the one number that cannot be argued with: energy.

## Build

### Setup

```
[include-by-region: engine.py#setup]
```

The flags follow the house convention: free-tier defaults first, a `--smoke`
mode that runs a fixed number of steps so CI can compare two runs byte-for-byte.
This chapter has no GPU tier and no training: it is pure numpy integration, and
the full run takes 0.01 minutes on a laptop; there is nothing to wait for. The
seed here controls initial conditions only (the orbit's eccentricity, the
spring's stretch, the launch angle) exactly as chapter 0.1's seed controlled
the shove; the physics is otherwise deterministic and the whole file is bitwise
reproducible. The knobs that matter pedagogically are `--dt` and `--steps`: they
are not a scaling ramp, they are the experiment.

### The systems

A physics engine needs three things from you: where everything starts, how hard
things push on each other, and (if you want to grade yourself) what quantity
ought to stay constant. We bundle those as a *system*: an initial position and
velocity, a `force(q, v)` law, and the true total `energy(q, v)`.

```
[include-by-region: engine.py#systems]
```

We give you three. An **orbit**: a point mass pulled toward a fixed attractor by
an inverse-square law: Newtonian gravity, the two-body problem. A **spring**: a
mass pulled back toward the origin in proportion to its stretch, Hooke's law, the
harmonic oscillator that every branch of physics reduces to eventually. And a
**freefall**: a body under constant gravity, which we include precisely because
it will *break* the neat story the other two tell. Hold that thought.

Read the shape of the state, because chapter 3.4 inherits it exactly. Position
`q`, velocity `v`, mass `m`, and a force that is a pure function of the two. When
3.4 adds constraints (a joint, a rod, a floor the body cannot pass through) it
adds a *constraint force* to this same `force`, and nothing else in the file has
to move. Unconstrained dynamics is the base case; constraints are a term you add.

### The integrators

Here is the whole chapter in three functions. Each one takes the current
`(q, v)` and returns the next `(q, v)`, one timestep `dt` later. They differ by
almost nothing, and the almost-nothing is everything.

```
[include-by-region: engine.py#integrators]
```

**Explicit Euler** is the one everyone writes first, because it is the
definition of a derivative with the limit removed: new position is old position
plus velocity times `dt`; new velocity is old velocity plus acceleration times
`dt`. Everything on the right-hand side is the *old* state. It is correct in the
limit `dt → 0` and wrong for every `dt` you can actually afford.

**Semi-implicit Euler** changes one line. Update the velocity first, then step
the position using the *new* velocity, not the old one. That is the entire
difference. It looks like a rounding-error-sized rearrangement. It is the
difference between an orbit that holds for a million steps and an orbit that
flies apart, and we are about to measure exactly that. This reordering is what
makes the method *symplectic*, and it is a close cousin of what MuJoCo does by
default. (Readers coming from a numerical-methods course will recognize it as
the first-order member of the symplectic family whose better-known relative is
leapfrog / Störmer–Verlet.)

**RK4** takes the accuracy question seriously. Instead of one estimate of the
derivative it takes four (at the start, twice at the midpoint, once at the end)
and blends them 1-2-2-1. Per step it is dramatically more accurate than either
Euler. It is also, and this is the twist the chapter turns on, *not symplectic*.

### Simulate and measure

```
[include-by-region: engine.py#simulate]
```

The loop is the ch0.1 rhythm with the black box removed: compute the force,
step, record. What we record alongside the trajectory is the total energy at
every step, because for all three of our conservative systems energy is exactly
the thing that must not change. A perfect integrator would hold it flat. The
amount it fails to is the *energy drift*, and we report it two ways: the signed
drift at the end of the run, and the largest excursion anywhere along it.

Running all three and printing the comparison is the deliverable: three drift
numbers that say, in order, "runs away", "holds", "holds best":

```
[include-by-region: engine.py#report]
```

## The measurement

Run the orbit under all three integrators for twenty seconds of sim time:

```
python engine.py --seed 0
```

and read the drift (seed 0, CPU):

| integrator      | final energy drift | worst excursion |
| --------------- | ------------------ | --------------- |
| explicit Euler  | **+22.1%**         | +22.1%          |
| semi-implicit   | +0.0098%           | 0.032%          |
| RK4             | −4.5e-9 %          | 5.1e-9 %        |

Read that table slowly, because three different things are true in it.

Explicit Euler **gains 22% of its energy** over three orbits, and the worst
excursion equals the final one, which means it never came back: the error is
monotone, one-directional, unbounded. More steps, more energy, wider spiral,
forever. This is the third orbit from the opening.

Semi-implicit Euler's drift is four orders of magnitude smaller, but the number
that matters is not its size, it is its *shape*: the worst excursion (0.032%) is
larger than the final drift (0.0098%), which means the energy went up, came back,
went down, came back. It **oscillates around the true value and stays inside a
band**, a band that does not grow with the length of the run. That bounded-ness
is the symplectic property, and it is worth more to a physics engine than raw
accuracy, because it means a simulation can run indefinitely without cooking off
or grinding to a halt.

RK4's drift is eleven orders of magnitude down: for this run it is *exact* to
the precision we can measure. So why does anyone ever pick semi-implicit over
it? Because RK4's tiny error, unlike semi-implicit's, is not bounded: it creeps
one direction. On this twenty-second run you cannot see it. On a run a thousand
times longer, the symplectic method's bounded wobble beats RK4's slow, honest
creep. Accuracy per step and stability over time are different virtues, and an
engine that must never fall over chooses the second. (Exercise 3 makes you watch
RK4's error shrink by a factor of sixteen when you halve the timestep, fourth
order, while Euler's only halves.)

## Break It

Run the same comparison on freefall:

```
python engine.py --system freefall
```

and the neat story falls apart. Explicit Euler gains energy (+9.3%),
semi-implicit *loses* the same amount (−9.3%), and neither is bounded. The
symplectic advantage vanished.

It vanished because freefall never comes back. A body thrown under constant
gravity recedes forever; there is no orbit to close, no oscillation to stay
inside of. The bounded-energy property of symplectic integrators is a statement
about *periodic and bounded* motion (orbits, springs, pendulums, walking gaits),
not about a rock falling into infinity. This is the honest caveat, and it is why
"semi-implicit conserves energy" is a claim you should never make without the
word *bounded* and the word *oscillatory* attached. The integrator did not get
better or worse; the system stopped being the kind of system the guarantee is
about.

## Read the real thing

The engine you have called since chapter 0.1 is C, and it is readable. The
unconstrained update you just built by hand is the core of MuJoCo's `mj_step`,
so read the original against your own three functions. Pinned to
`google-deepmind/mujoco` at tag `3.10.0`, the version this course pins.

**The integrator choice.** Your `report` region hard-codes the three-way
comparison; MuJoCo makes it a field. `mjModel.opt.integrator` is the
`int integrator;` line of the option struct in `include/mujoco/mjmodel.h`, and
its four legal values live in the `mjtIntegrator` enum in
`include/mujoco/mjtype.h`: `mjINT_EULER = 0`, `mjINT_RK4`, `mjINT_IMPLICIT`,
`mjINT_IMPLICITFAST`. (Heads-up: that enum *moved* in 3.x. Older code and blog
posts point at `mjmodel.h`; at 3.10.0 it is `mjtype.h`.) The default is value
`0`, and the source comments it in three words: `// semi-implicit Euler`. That
is this chapter's headline, confirmed in the enum: of every integrator MuJoCo
ships, the one it reaches for by default is your `semi_implicit_step`, not the
"more accurate" RK4. `mj_step` in `src/engine/engine_forward.c` is the `switch`
that dispatches on that field: `mjINT_EULER → mj_Euler`, `mjINT_RK4 →
mj_RungeKutta(m, d, 4)`.

**mj_Euler.** Your `semi_implicit_step` updates velocity, then steps position
with the *new* velocity. `mj_Euler` (`src/engine/engine_forward.c`, forwarding to
`mj_EulerSkip`) is that same method, and its comment says so word for word:
"Euler integrator, semi-implicit in velocity." Its no-damping branch is your two
lines. What it *adds* is the case your toy skips: when a DOF has damping it does
not integrate velocity explicitly: it factorizes the mass matrix plus
`h·diag(B)` and solves, integrating that damping *implicitly* for stability, plus
actuator dynamics and sleep filtering. Same skeleton, hardened for a real robot.

**mj_RungeKutta.** Your `rk4_step` writes the 1-2-2-1 blend by hand.
`mj_RungeKutta` in the same file drives it from a Butcher tableau (`RK4_A`,
`RK4_B`) so one loop serves any order, re-runs the *whole* forward dynamics per
stage through `mj_forwardSkip` (each of the four samples respects contacts and
constraints, which your flat force law has none of), and advances position with
`mj_integratePos`, which is quaternion-aware where yours is plain `q + dt·v` in
Euclidean space.

What the pure-numpy version omits, then, is two things. First, the
`mjINT_IMPLICIT` / `mjINT_IMPLICITFAST` integrators (`mj_implicit` /
`mj_implicitSkip`, same file): implicit-in-velocity methods that use the RNE
force derivative to stay stable under stiff damping and gyroscopic terms, the
precise stability-over-accuracy trade this chapter measured, taken one rung
further than semi-implicit Euler. Second, the constraint solver whose
`qfrc_constraint` feeds `qacc` before any integrator runs. That is chapter 3.4.

Read next, in order: `include/mujoco/mjtype.h` for the `mjtIntegrator` enum (your
three functions, named); then `mj_Euler` / `mj_EulerSkip` in
`src/engine/engine_forward.c` (your semi-implicit step, plus damping); then
`mj_RungeKutta` in the same file (your RK4, generalized); and last the `mj_step`
`switch` that ties the field to the function, the black box you have called
since chapter 0.1, finally open.

## What you built, and what comes next

You built the inside of `mj_step` for the case with no contacts: state, a force
law, three integrators, and a measured, defensible reason to prefer one. Every
integrator option in MuJoCo's documentation (`Euler`, `implicit`,
`implicitfast`, `RK4`) is a point on the trade you just measured by hand.

Chapter 3.4 keeps this exact `(q, v, m, force)` interface and asks the next
question: what if the body *cannot* go where the force sends it, because it is
bolted to a joint, or resting on a floor? That is a constraint, and a constraint
is a force you solve for rather than write down. The base case is done.

## Exercises

Three, in `exercises/`. One predict-then-run where you commit (before the code
is allowed to answer) to which of the three integrators sends the orbit's
energy climbing without bound. One code-completion: explicit Euler is handed to
you whole, and you turn it into semi-implicit Euler and RK4 by changing exactly
what this chapter says changes. And one more predict-then-run that halves the
timestep and makes you match each integrator's error-shrink factor to its order:
Euler's error roughly halving, RK4's dropping by about sixteen.
