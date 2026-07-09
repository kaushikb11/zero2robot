# 3.5: Build a Physics Engine III: Contact (The Hard Part)

<!-- objectives: rendered from meta.yaml, do not duplicate here -->
<!-- numbers below are seed 0, CPU, numpy 2.4.6; reproduce with contact.py --seed 0 -->
<!-- wall-clock renders from wallclock.csv (ch3.5-contact, cpu-laptop) -->

## See it work

A ball, dropped a meter onto a table. Two of them, actually, drawn in the same
window. Same gravity, same drop. One falls, meets the table, and rests on it:
sits exactly on the surface and stays there, still. The other falls, and where
the table should stop it, it keeps going: it drives a quarter of its own radius
*into* the table before something shoves it back, then bounces, rings, and finally
settles, but settles a hair's-width *below* the surface, resting inside the solid
table like a coin pressed into dough.

Nothing about the two balls differs except how each one answers a single question:
what force does the table apply? The sinking one models the table as a very stiff
spring: squeeze it and it pushes back. The other one solves, at every step, for
the exact push that stops the ball without letting it through. The sinking, the
ringing, the ball that ends up buried a millimeter deep: none of it is a bug in
the physics. It is contact, computed the easy way, and the easy way is not good
enough. That is the whole chapter.

And here is the part that should make you sit up: that jitter, that sinking, that
occasional catastrophic explosion where a body is flung off to infinity, you have
been watching those artifacts since chapter 0.1, every time a MuJoCo sim did
something faintly wrong. This is where they come from, and this is where they get
explained.

## The problem

Chapter 3.3 built the inside of `mj_step` for a free body. Chapter 3.4 added
**joints**, constraints `g(q) = 0` you cannot write as a force but must *solve*
for, as a Lagrange multiplier. A joint is a constraint that is **always on**: the
pendulum rod is always exactly length L, this step and every step.

Contact is the constraint that switches **on and off**. The table pushes up on the
ball only while the ball is touching it, and it *never pulls it down*. Write the
gap between ball and table as `phi`. Then contact is three conditions at once:

```
phi >= 0            the ball may not penetrate the table
lambda >= 0         the table may only push (never pull)
phi * lambda = 0    no push unless they are touching (phi = 0)
```

That last line is the whole difficulty. It says the force and the gap are
**complementary**: at least one of them is zero at all times. Either the ball is
off the table (`phi > 0`) and there is no force (`lambda = 0`), or the ball is on
the table (`phi = 0`) and there can be a force. This is not an equation you solve
with `np.linalg.solve`: it is an **inequality**, a *complementarity problem*, and
it is a genuinely different and harder kind of math than anything in 3.3 or 3.4.

The chapter builds the two honest ways to cope, from scratch, in about 400 lines of
numpy on top of the last two chapters' engine, and measures exactly what each one
gets wrong.

## Build

### Setup

```
[include-by-region: contact.py#setup]
```

Same house conventions as 3.3 and 3.4: free-tier defaults first, a `--smoke` mode
with a fixed step count so CI can diff two runs byte-for-byte, no GPU tier, no
training. The full run takes about 0.02 minutes on a laptop CPU: a point mass, a
scalar contact solve per step, and a small stability sweep; there is nothing to
wait for. The knobs that carry the lesson are `--contact` (the comparison),
`--stiffness` (the penalty spring you will learn to distrust), and `--dt` (raise it
past a critical value and watch penalty detonate). Every headline below is an
*ordering* (penalty sinks and can explode, LCP holds) and the `--seed` nudges only
the drop height by a couple of centimeters, so that ordering is robust to it.

We drop to **two dimensions** and model bodies as **point masses with a radius**:
pucks, effectively. No rotation, which means no friction cone: this chapter is
about the *normal* direction, the push perpendicular to the surface. Friction and
sliding stay out of scope for the whole from-scratch arc: chapter 3.6 fakes PushT's
surface friction as a viscous drag rather than model it, and the real friction cone
shows up only when you read MuJoCo's own contact solver. Being honest about that scope is part of the
lesson; contact is hard enough one axis at a time.

### The scenes

```
[include-by-region: contact.py#scenes]
```

Three scenes, the same `(q, v, m)` shape from 3.3 stacked to N bodies: a **drop**
(one ball, settles on the table), a **bounce** (one ball with restitution), and a
**stack** (three balls in a tower, the case with *several* contacts at once, and
the one chapter 3.6 will reuse when a pusher leans on a T-block). The material's
`restitution` is a number the LCP solver uses directly; the penalty contact has no
such knob, which is a tell we will come back to.

### The geometry: finding contacts

```
[include-by-region: contact.py#contacts]
```

Both solvers start here, and so will chapter 3.6. A **contact** is a place two
surfaces overlap: who is involved (a body, and either another body or the floor),
the unit **normal** they may only push apart along, and the **penetration depth**.
We look a hair *before* the surfaces actually touch (a small margin) so that a
resting contact (depth essentially zero) is still on the list and the solver can
hold it there. And we build the list in a fixed order, floor contacts then pairs,
so the whole run stays bitwise deterministic even though contact is famously the
part of a physics engine where determinism goes to die.

### Way one: penalty (a contact is a spring)

```
[include-by-region: contact.py#penalty]
```

The penalty method is almost too simple to believe. Pretend the surfaces are a
very stiff spring: the deeper they overlap, the harder they push apart,
`force = k * depth`, minus a damper so the bounce dies down. Clamp it to be
push-only, `max(0, ...)`, because a spring would otherwise *pull* a separating
body back, and the table must never pull. That single `max` is the one-sidedness,
enforced by hand.

And that is *all*. A penalty contact is just an added force, so chapter 3.3's
integrator (copied in here verbatim, the repetition is the point) steps it with
zero changes. The integrator never learns that `force` now runs a collision check.

You pay for the simplicity everywhere else, and the metrics name the bill:

- The ball **sinks**. To hold a ball of weight `mg` up, the spring must compress
  until `k * depth = mg`, so it rests at `depth = mg/k`, *inside* the table. With
  our stiff `k = 1e4` that is 0.0098 of the radius: small, but never zero, and it
  grows the moment you soften the spring.
- The ball **rings**. A spring plus a mass is an oscillator; on impact it bounces
  and jitters before the damper wins. In the bounce scene it visibly never stops.
- Worst of all, the ball can **explode**. An explicit integrator can only hold a
  spring of stiffness `k` stable while `dt < dt_crit ~ 2*sqrt(m/k)`. For our spring
  that is 0.02 s. Step at `dt = 0.03` and the spring pumps in energy faster than the
  damper removes it: the ball is flung away, its energy blowing up by a factor of
  **1409**. That is the timestep-instability from chapter 0.1, in the flesh.

You can fix any one of these by tuning `k`, but stiffening the spring to sink less
*shrinks* the stable timestep, and softening it to allow a bigger timestep makes it
sink more. There is no setting that wins. That trap is the reason we need way two.

### Way two: LCP-flavored (solve the complementarity)

```
[include-by-region: contact.py#lcp]
```

Instead of faking contact with a spring, solve the complementarity conditions
directly, but at the **velocity** level, which is what real-time engines do. Take a
gravity step to get a *predicted* velocity, then find the contact **impulse**
`lambda >= 0` that leaves the body no longer moving into the table. For a single
contact that is one clamp; for the stack's several contacts we sweep them one at a
time, a fixed number of passes: **projected Gauss-Seidel**, the humble workhorse
inside a great many physics engines. Each sweep, each contact asks for the impulse
that would hit its target normal velocity, and then we *project*: the accumulated
impulse may never go negative, because the table may only push. That `max(0, ...)`
is the complementarity condition, exactly as the clamp in the penalty force was:
the same idea, in the two different languages of the two solver families.

The crucial structural difference: this is **not a force added to the integrator**.
It is a velocity projection that *replaces* the plain step. Penalty bends to fit
chapter 3.3's `force(q, v)`; the complementarity solve does not. That is why contact
is where the tidy "a constraint is just an added force" story of 3.4 finally breaks.

A small `baumgarte` term (yes, the same Baumgarte from 3.4) feeds any leftover
penetration back as a gentle separating velocity, so the body is nudged back out to
the surface rather than left buried. It is a hand-tuned gain, and admitting that is
part of the honesty.

### Simulate and measure

```
[include-by-region: contact.py#simulate]
```

The loop is 3.3's, recording the contact-quality numbers at every step: the worst
**penetration** (a body should not sink through a table), the total **energy** (a
correct inelastic contact only ever *removes* energy, it never invents any), and,
from those, the jitter and the phantom energy. These play exactly the role energy
drift played in 3.3 and constraint drift played in 3.4.

## The measurement

Drop a ball for four seconds of sim time, penalty versus LCP-flavored, and read
the contact quality (seed 0; penetration as a fraction of the ball's radius):

| contact        | max penetration | rest penetration | stable up to dt |
| -------------- | --------------- | ---------------- | --------------- |
| penalty        | **0.251**       | 0.0098           | 0.008 s         |
| LCP-flavored   | 0.057           | **0.0**          | 0.064 s         |

The penalty ball drives 4.4× deeper on impact and comes to rest *inside* the table;
the LCP ball penetrates a quarter as much on impact (about one step's worth of
approach velocity, which no velocity-level solver can avoid without continuous
collision detection) and then holds exactly on the surface. And penalty is stable
only up to a timestep eight times smaller than the one LCP shrugs off.

The **bounce** scene adds the phantom-energy artifact. There, the penalty spring
gains energy out of nothing: `energy_excess = +0.006` at the default timestep,
climbing to **+0.148** (a 15% energy gain, the ball bouncing *higher* than it was
dropped) at `dt = 0.005`, while the LCP contact, which is told the restitution
directly, never exceeds the energy it started with. The **stack** scene adds the
several-contact case: penalty lets the tower sink and settle into itself (worst
overlap 0.059 of a radius), while the LCP sweep holds all three balls to within
five-*millionths* of a radius, half a micron.

## Break It

Penalty sank and rang, but at the default step it never exploded, so push the one
knob it cannot survive. Raise the timestep past the spring's stability limit:

```
python contact.py --contact penalty --dt 0.03
```

Predict first: the spring constant did not change, only how big a step you ask the
explicit integrator to take. An explicit integrator holds a spring of stiffness `k`
stable only while `dt < dt_crit ~ 2*sqrt(m/k)`; step past it and the spring pumps
energy in faster than the damper removes it. The measurement is the **×1409
energy blow-up** from the build section, in the flesh: the ball flung toward
infinity, the timestep-instability you first met in chapter 0.1.

Now try to escape it the way you'd reach for first: stiffen the spring with
`--stiffness` so the ball sinks less. It makes the cliff *worse*: a bigger `k` shrinks `dt_crit`, so the
step that was borderline now detonates. Soften `k` to widen the stable step and the
ball sinks deeper instead, at `depth = mg/k`. There is no single `k` that sinks
little and survives a big step: the artifacts trade against each other, and that
wall, not any one number you failed to find, is why way two abandons the spring
entirely. Run the same `--dt 0.03` with `--contact lcp` and it holds.

## Why MuJoCo looks the way it does

Sit with the trade you just fought. Penalty is trivial to write and folds into any
integrator, and it sinks, rings, needs a punishingly small timestep, and can blow
up. Hard complementarity holds the body honestly, and it is a solve, it has no
clean notion of a slightly-soft contact, and a true LCP is expensive and brittle
when contacts are many. Neither is what you want.

MuJoCo's answer is to sit deliberately *between* them: a **soft, regularized,
convex** contact model. It builds the very same per-contact structure you built (normals,
effective masses, a projected sweep), but instead of a hard `phi >= 0` it
allows a *little* give, governed by physical stiffness and damping parameters
(`solref`, `solimp`) rather than a raw spring constant, and solves a convex problem
that always has a unique, stable answer. Every strange-looking knob in MuJoCo's
contact documentation is an answer to a pain you just felt in your own hands: the
softness is penalty's give made principled; the reference parameters are the
Baumgarte gain made physical; the convex solve is the LCP made cheap and robust.
That is the reading for this chapter (see meta.yaml `rtrt`).

## The honest limits

Say plainly what this little engine does not do, because the lesson is the *trade*,
not a finished solver:

- **No friction.** Bodies are frictionless pucks; there is no rotation and no
  friction cone. The whole chapter is the *normal* direction. Sliding, stiction,
  and the friction pyramid stay out of scope for the whole engine arc: chapter 3.6
  fakes surface friction as drag, and you meet the real cone only when you read
  MuJoCo's own solver.
- The LCP-flavored solve is a fixed-iteration **projected Gauss-Seidel**, not a true
  LCP pivot. It is enough to hold a small stack; it is not a research contact solver.
- Even the LCP contact **penetrates on fast impact** (about one step of approach
  velocity) and leans on a hand-tuned Baumgarte gain. It is *better*, not perfect.
  Do not let the green curve fool you into thinking contact is solved. It is
  managed.

## Read the real thing

The `rtrt` block in `meta.yaml` pins `google-deepmind/mujoco` at tag `3.10.0`,
the same MuJoCo the whole course runs on. Two files hold the production answer to
the trade you just measured by hand. Read them against `contact.py`.

Start with what you built. Our `penalty` region is one function, `penalty_force`:
`fn = max(0, k*depth - c*vn)`, a stiff spring clamped push-only, and that raw `k` is
exactly what forces the `dt_crit ~ 2*sqrt(m/k)` cliff you watched detonate. Our `lcp`
region is `solve_contacts` (a fixed-iteration projected Gauss-Seidel sweep, one
`max(0, ...)` clamp per contact) wrapped by `lcp_step`, which predicts the velocity
and then projects it. Hold those two shapes in your head.

Now `src/engine/engine_core_constraint.c`. MuJoCo does not pick spring-*or*-solve; it
builds a *soft* reference and hands it to a solver. `mj_makeImpedance` (with its helper
`getimpedance`) turns each constraint's `solref` and `solimp` into a stiffness `K`, a
damping `B`, and an impedance `d` in `[0, 1]`: the standard-format comment spells the
map, `K = 1/(d_width^2 * timeconst^2 * dampratio^2)` and `B = 2/(d_width * timeconst)`.
Then `mj_referenceConstraint` computes a reference acceleration `aref = -B*vel -
K*I*(pos-margin)`. Read that slowly. It is our penalty spring (`k*depth`) and our
Baumgarte push (feeding `pos` back as a separating velocity) fused into one
physically-parameterized target, with `solimp`'s impedance softening the constraint
smoothly from free to rigid instead of our hard `max`. `solref` is `k` and `damping`
made principled; `solimp` is the softness we never had.

Then `src/engine/engine_solver.c`. `mj_solPGS` (core `solPGS`) is your `solve_contacts`
grown up (the same per-constraint projected sweep), but MuJoCo's default is the
*convex* primal solver, `mj_solNewton` (or `mj_solCG`) via `mj_solPrimal`, minimizing a
regularized cost (the `efc_R` diagonal that `mj_makeImpedance` also builds) with a
linesearch, so the answer is unique and stable where a hard LCP turns brittle.

What they add is everything we scoped out: real friction cones (our pucks have none), a
convex regularizer our bare PGS lacks, and the softness that keeps one `dt` safe across
a whole scene. Ours is honest, not finished: frictionless, and a fixed-sweep
Gauss-Seidel, not a true LCP pivot.

**Read next:** open `src/engine/engine_core_constraint.c` and find
`mj_referenceConstraint`'s one line, `aref = -B*vel - K*I*(pos-margin)`. That is your
penalty spring and your Baumgarte push, written once and physically: the whole chapter
in a single formula.

## What you built, and what comes next

You built contact: the one-sided, on-off, complementarity constraint that is the
hard heart of every physics engine. You built it twice, the easy way and the honest
way, and you *measured* the difference: penetration, phantom energy, jitter, and the
timestep cliff that has been haunting your sims since chapter 0.1. You can now look
at any contact artifact and name its cause.

Chapter 3.6 cashes this in. **PushT** (a circular pusher shoving a T-shaped block
across a table) is nothing but bodies in contact: the pusher contacting the block
is exactly the body-body path the `stack` scene exercised, `detect_contacts` and
`solve_contacts` reused wholesale. With a contact model in hand, you re-create the
whole PushT environment inside your engine and *reload* the behavior-cloning policy
you trained back in chapter 1.1 (no retraining), then run it in both your engine and
MuJoCo to measure exactly how far the two sims drift apart. That closes the loop from
raw dynamics back to the policies of Phase 1.

The print table this report region produces *is* the deliverable: the two rows
that say, in order, "sinks and can explode" and "holds":

```
[include-by-region: contact.py#report]
```

One closing distinction worth carrying into 3.6. The contact solve here works at the
**velocity** level (it computes an impulse that corrects the velocity in a single
step) where chapter 3.4's joint constraint worked at the **acceleration** level.
That is not an accident: an impact is a sudden jump in velocity, and an
acceleration-level solve cannot represent a jump. Impulses can. And real friction
would be a *second* complementarity problem stacked on this one, the friction cone,
solved in the same projected sweep. This little engine never builds it, and neither
does 3.6, which fakes PushT's surface friction as a viscous drag; you meet the real
cone only when you read MuJoCo's own contact solver. The velocity-level impulse spine
you built here is exactly what 3.6 reuses, wholesale, to push a T-block around.
