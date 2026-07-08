# 2.7: Sim-to-Real Intuition Lab II: Randomize to Generalize

## You can't fix the world, so stop training in only one

Chapter 2.6 handed you a measuring instrument and a verdict. You took a policy
raised in a flawless simulator, injected the perturbations that dominate the
reality gap, and watched it degrade. The lesson landed as a question: a policy
trained in *one* clean world has no defense it was never asked to learn. So what
do you do about it?

You cannot fix the real world to match your simulator. The robot's motors will
always be a little weaker than the model, its floor a little more or less grippy,
its payload never exactly the nominal mass. So do the opposite. Stop making the
simulator *one* fixed world. **Randomize its dynamics every episode** (sample a
different mass, a different friction, a different gravity each time the policy
resets) and the policy is forced to learn a behavior that works across the whole
*range* instead of memorizing the one point it was handed. Meet the reality gap in
sim, where falling over is free.

That is **domain randomization**, and it is the workhorse behind the famous
sim-to-real results: Tobin's randomized rendering for object detection, OpenAI's
in-hand cube that trained entirely in a randomized sim and transferred to a real
Shadow Hand. This chapter builds it from scratch on the quadruped, trains two
policies that differ in exactly one thing, and then does the part that separates
engineering from wishful thinking: it **measures whether it worked.**

## Two policies, one difference

We train two PPO policies on the quadruped, the same from-scratch PPO you wrote
in chapter 2.1, unchanged. The *only* thing that differs between them is the world
they train in:

- **NARROW** trains on the nominal dynamics, every episode. The mass, friction,
  and gravity are always the book values.
- **RANDOMIZED** resamples the torso/leg masses, the foot friction, and gravity
  from a band around nominal at the start of every episode.

Domain randomization, in code, is nothing more than scaling the simulator's
physical parameters in place, on top of the contact solver the quadruped env
pins. We touch mass, foot friction, and gravity; we never touch the solver
iterations or the contact-cone settings the env README fixes for determinism, so
the whole thing stays bitwise-reproducible per seed.

```
[include-by-region: dr.py#randomize]
```

A single knob, `--dr_width`, scales the band. `--dr_width 0` collapses it to a
point: that *is* the narrow policy, so both policies run through the identical
code path and differ only in one number. Re-seeding both trainings from the same
`--seed` means they start from the identical network initialization and see the
same reset noise; the only variable in the entire experiment is whether the
dynamics move. That is a controlled experiment, not a demo.

```
[include-by-region: dr.py#ppo]
```

## Evaluate across the gap, with error bars

Training return is not the point; *generalization* is. So we sweep a test-dynamics
axis (by default the robot's mass) from light to crushingly heavy, and evaluate
BOTH policies at every point on held-out seeds. For each point we report the mean
return, its spread, and the **survival rate**: the fraction of episodes the robot
stayed up for the full ten seconds. Survival is the honest binomial the reality
gap moves first (chapter 1.6's discipline: a rate is a band, not a number).

```
[include-by-region: dr.py#eval]
```

The sweep deliberately runs *past* the randomization band, into the gap where
neither policy trained. That is the "break-the-policy" playground the map
promises: crank the test dynamics until something falls, and see which policy
falls first.

## Run it

```
python curriculum/phase2_reinforcement/ch2.7_dr/dr.py --seed 0
```

<!-- wall-clock table renders from wallclock.csv (measured: T4 5.63 min, L40S 2.86 min) -->

This trains both policies from scratch and runs the full sweep across the mass
gap. Here is the generalization curve on the default `--seed 0` (mean return ± std
| survival rate, 16 held-out episodes):

```
  mass_scale        0.8           1.0           1.2           1.4           1.6
  narrow       206±1|1.00   206±0|1.00   112±82|0.44    32±1|0.00    27±1|0.00
  randomized   203±0|1.00   203±0|1.00   100±79|0.38    29±1|0.00    24±0|0.00
```

On this seed the two curves lie almost on top of each other: DR bought nothing.
But hold that thought: seeds 1 and 2 tell a different story, and the difference
between them is the point. Now read it honestly, because the honest reading is the
whole chapter.

## The result the map didn't promise

Here is what the measurement actually says, and it is subtler than "randomization
wins."

**Near nominal, both policies stand, and randomization buys nothing the narrow
policy didn't already have.** Standing near the book dynamics is a stable
equilibrium: the PD servos hold a crouch, and a small change in mass is a small
push the same feedback absorbs. The narrow policy generalizes across the easy part
of the range *for free*; there is nothing there for domain randomization to fix,
and both policies survive every episode at 0.8× and 1.0× mass.

**Deeper in the gap, domain randomization sometimes pays off spectacularly, and
sometimes not at all.** On seed 1, the randomized policy holds a full-survival,
~200-return stance all the way out to 1.4× mass, a load under which the narrow
policy has already collapsed (0.38 survival). That is the textbook result: the
narrow policy overfit the one stance that works at nominal, and the randomized
policy, forced to cope with heavier and lighter bodies in training, learned a
*sturdier* one that carries into the gap. It even shows the "1.2× ceiling" was
never physical: it was the *narrow policy's* ceiling, and DR pushed past it. But
run seed 0, or seed 2, and the edge evaporates: the randomized policy converges to
an ordinary nominal stand and falls right alongside the narrow one. Across three
seeds the survival edge across the gap is −0.02, +0.22, −0.09.

**That spread IS the finding.** The average benefit lives *inside the seed band*,
exactly the trap chapter 1.6 taught you to catch. If you had trained one seed, seen
+0.22, and written "domain randomization extends the robust range by 40%," you
would have published noise. The honest statement is: at this compute budget, DR
*can* produce a dramatically more robust policy, but it does so *unreliably*,
because randomizing the dynamics makes the training problem strictly harder, and
400k steps is not always enough to solve the harder problem. Randomization is not
free insurance; it is a harder problem that needs more compute to pay off
dependably. (Cut the step budget below this and it is worse still: the randomized
policy sometimes fails to learn even the nominal stand.)

## So when *does* domain randomization help, reliably?

This is the judgment the chapter exists to build. Domain randomization pays off,
and pays off *dependably*, when two conditions both hold:

- the narrow policy is **genuinely brittle** to the axis you randomize: the
  optimal behavior is **dynamics-specific**. A walking gait tuned to one friction
  slips on another; an in-hand manipulation tuned to one mass distribution drops a
  different one. There, the narrow policy *overfits sim-specific dynamics* and DR
  is the fix, not a coin flip. A passively-stable *stand* is the weakest case:
  most of its range needs no adaptation at all.
- you spend **enough compute** to actually solve the harder randomized problem.
  The famous transfers (OpenAI's in-hand cube, sim-to-real locomotion) randomize
  hard *and* train long. Under-train the randomized policy and you get the seed 0
  / seed 2 story: the extra difficulty, none of the payoff.

Notice how cleanly the negative half agrees with chapter 2.6: there, the worst
perturbation for these simple policies was **latency**, not model mismatch. They
already shrugged off mass and gravity. Domain randomization is the cure for model
mismatch. Where model mismatch was never the disease, the cure does little.
**Measure the gap, and measure it with error bars, before you trust the method.**

## The randomization band is a hyperparameter (exercise 2)

If a wider band always meant more robustness, you would just crank `--dr_width` to
the sky. The second exercise makes you test that folklore: sweep the band from 0
(no DR) to double-wide and watch what it buys and where it stops. A wider band does
buy more robustness through the *middle* of the gap (on average, though noisily)
and it does so without denting nominal return. But the *deepest* gap point stays
pinned on the floor at every width: by 1.6× mass the ±12 Nm servos simply cannot
hold, and no amount of sampling near an impossible load makes it possible.
Randomization widens the range you can reach; it never lifts the ceiling of what
the motors can do. "Randomize harder" is a real dial with a real limit, not a free
path to transfer.

## Determinism and honesty

Everything is seeded (torch, every env reset, and the domain-randomization draw
stream) so a fixed `--seed` reproduces byte-for-byte on CPU (CI checks the smoke
twice over), and randomizing on top of the pinned contact solver preserves that
guarantee. But the *conclusion* wobbles seed to seed, exactly as chapter 1.6
warned: on one seed the randomized policy looks a touch more reliable, on another
a touch worse. That is the tell. When the narrow-vs-randomized difference lives
inside the seed band, the honest report is that you have **not** demonstrated a
benefit, and this chapter reports that, rather than cherry-picking the seed where
DR happened to win.

## Read the real thing

Three papers built the intuition this chapter distills, and reading them against
what you just measured is the point. **Tobin et al. (2017)** introduced domain
randomization by randomizing a simulator's *rendering* (textures, lighting,
camera pose) so an object detector trained purely in sim transferred to real
images. **Peng et al. (2018)** moved the idea from pixels to *physics*, randomizing
the *dynamics* (masses, friction, latency) exactly the axes `dr.py` scales, and
transferred a manipulation policy to a real arm. **OpenAI's Dactyl (2018/2019)**
is the headline case: an in-hand cube-reorientation policy trained entirely in a
randomized sim and transferred to a physical Shadow Hand.

Read them for the contrast with what happened here. Every one of those results
randomizes a behavior that is *genuinely dynamics-brittle* (contact-rich
manipulation, a dexterous regrasp) where the optimal action really does change
with the physics, so the narrow policy has no choice but to overfit and DR is the
fix. And every one *trains long*, spending the compute to actually solve the
harder randomized problem. Those are the two conditions a passively stable
quadruped stand at a free-tier budget does not meet, which is precisely why your
measurement came back inside the seed band. The method is real; the substrate here
is honest about its limits.
