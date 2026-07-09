# 2.6: Sim-to-Real Intuition Lab I: Latency & Noise

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## Your policy has only ever lived in a perfect world

Every policy you have trained grew up somewhere flawless. The PPO balancer from
chapter 2.1 read the pole's exact angle, the instant it asked, in a world whose
gravity and masses were the true ones, because in a simulator they are the
*only* ones. The SAC reacher from 2.2 had the same charmed childhood. Both score
near-perfectly on the task they were trained on, and both are, in a specific and
measurable way, spoiled.

A real robot lives in none of that. Its sensors are noisy. Its observations
arrive a few control steps late: a camera exposes, a filter smooths, a USB bus
and a message queue each cost a millisecond, and by the time the policy sees the
world the world has moved. And the physical robot's masses, joint friction, and
gravity never quite match the numbers the simulator used. That collection of
discrepancies is the **reality gap**, and it is why a policy that scores 500/500
in sim can tip over on the bench.

This chapter builds hardware intuition **without hardware**. We take a policy
trained in the clean sim, and we re-run it while injecting the three
perturbations that dominate the gap (one family at a time) and we *measure* how
far it degrades. The goal is not a number; it is a reflex. By the end you should
look at a new task and guess which perturbation will hurt it most, the way an
experienced roboticist does.

## Three perturbations, built from scratch

There is no wrapper hiding the injection. Each perturbation is a few lines you
can read, and each acts on the policy's *interface with the world*, never on its
weights. The reality gap is entirely outside the network.

**Sensor noise** adds gaussian noise to the observation the policy reads.
**Latency** delays that observation through a ring buffer, so the policy acts on
a stale snapshot: where the world *was*, not where it is. **Model mismatch**
scales the eval env's mass, joint damping, and gravity away from the values
training saw, by editing the compiled MuJoCo model in place.

```
[include-by-region: perturb.py#perturb]
```

A design decision worth pausing on: the policy is *always* a plain
`obs -> action` function, and the perturbations wrap that observation stream (or
the env's physics). This is why the chapter's scripted fallback controller (the
same hand-tuned balancer and IK-reacher from the `common/` envs, rebuilt here to
read the *observation* instead of the raw simulator state) degrades under noise
and latency exactly as a learned policy does. A controller that only ever sees
observations has no privileged access to truth, and neither does your robot.

## Load a policy you trained, and perturb it

No policy binary ships in this repo (the `.pt` files are gitignored, root
invariant 5). You point `perturb.py` at a checkpoint you trained with chapter
2.1 or 2.2; with no checkpoint (and always under `--smoke`, for hermetic CI) it
falls back to the scripted baseline, which tells the same story.

```
[include-by-region: perturb.py#policy]
```

The eval loop is deliberately close to the ones you already know (reset on
held-out seeds, act, step) with the perturbation spliced into perception:

```
[include-by-region: perturb.py#eval]
```

## Run it: the degradation curves

```
python curriculum/phase2_reinforcement/ch2.6_perturb/perturb.py \
    --seed 0 --task cartpole --ckpt outputs/ch2.1-ppo/ppo_agent.pt
```

The headline run sweeps each perturbation family from clean to broken and reports
the success rate falling along the way. On a CPU laptop the full three-family
sweep takes about **2.77 min (measured)**. For the chapter-2.1 PPO balancer
(seed 0):

```
clean baseline: success 1.00  mean_return 500.0000
  sensor_noise    (     obs_noise)  success@ 0:1.00  0.01:1.00  0.02:1.00  0.05:1.00  0.1:1.00  0.2:1.00
  latency         ( latency_steps)  success@ 0:1.00  1:1.00  2:1.00  4:0.00  8:0.00
  model_mismatch  ( gravity_scale)  success@ 0.5:1.00  0.75:1.00  1:1.00  1.25:1.00  1.5:1.00  2:1.00
worst perturbation for this policy: latency  (success drop +1.00 from clean baseline)
```

Read that carefully, because it is more interesting than "the policy gets worse."
This policy **shrugs off sensor noise** (even a large jitter on the pole angle
barely moves it) and it **shrugs off gravity mismatch**, balancing fine under
half or double the gravity it trained on. It learned a genuinely robust feedback
controller. But it falls **off a cliff** at observation latency: solid through 2
steps of delay, and dead by 4 (80 ms). Balance is a *stability* problem, and
stability problems die on delay: there is a delay margin, and past it the
corrections arrive too late and amplify instead of damp.

## The mirror: run the reacher too

Switch tasks and the lesson inverts. The chapter-2.2 SAC reacher, perturbed the
same way (seed 0):

```
clean baseline: success 0.70  mean_final_dist 0.0386
  sensor_noise    (     obs_noise)  success@ 0:0.70  ...  0.1:0.75  0.2:0.35
  latency         ( latency_steps)  success@ 0:0.70  ...  4:0.70  8:0.50
  model_mismatch  (    mass_scale)  success@ 0.5:0.70  1:0.70  1.5:0.70  2:0.70  3:0.70
worst perturbation for this policy: sensor_noise
```

The reacher is the cartpole's opposite. It **tolerates latency** (a settling
task can afford to arrive a little late) and it is instead brittle to **sensor
noise**, which jitters its notion of where the target is. Gravity mismatch is a
genuine *no-op* here: the arm moves in the plane perpendicular to gravity, so
scaling gravity exerts no torque on it at all (mass is its sharp knob, and even
that the learned controller mostly absorbs).

That is the whole intuition this chapter exists to build: **which perturbation
bites is a property of the task's dynamics, not of the algorithm.** Nobody could
have told you "fear latency" or "fear noise" in general; you have to know what
kind of control problem you are solving. A balance problem fears delay. A reach
problem fears noise. You now have a way to find out for any policy, before it
ever touches a robot.

## Break it (optional, not graded)

`--break` pins the observation latency to an extreme 16 steps (320 ms) and runs a
single eval. The clean-trained policy has no memory to fill that gap and fails
outright. It is a teaching toggle, not a graded bug-hunt: per the RL doctrine
(the chapter-2.1 spike), single-run effects are noise, so the graded exercises
assert a seed-robust *structural* fact instead: which perturbation wins, and
that the latency cliff is real across seeds.

## Determinism and honesty

Perturbed eval is noisy in two senses, and we are honest about both. The
*perturbation* RNG is its own seeded stream, so a fixed `--seed` gives
byte-identical results twice over on CPU (CI checks this). But the *degradation
numbers* still wobble seed to seed: the exact breaking latency moves between 2
and 4 steps depending on which clean-sim policy you trained. So this chapter, like
chapter 1.6 taught, reports the *shape* (solid → knee → dead) as the robust
finding, and leaves the exact cliff step as something you read, not a single
number you trust.

## The seam into 2.7

There is a reason this is "Lab I." You have now measured that a policy trained in
*one* clean world is defenseless against a gap it was never shown. And, tellingly,
the gap it fears most (latency, for the balancer) is not one you can noise-
filter your way across. No amount of better sensing gives a policy back a
stability margin it never learned to have.

Chapter 2.7 is the answer: **domain randomization**. Instead of training in one
perfect world, train across a *distribution* of imperfect ones (randomize the
masses, the delays, the noise during training) so the policy meets the reality
gap in sim, where falling over is free. You have just built the measuring
instrument that will prove whether it worked.

## Read the real thing

Unlike its siblings, this lab was not carved from one canonical repo: there is no
single "reality-gap benchmark" file to read against, because measuring the gap is
done paper by paper, robot by robot. So read the two that turned these sliders into
named, measured quantities. Tan et al., *Sim-to-Real: Learning Agile Locomotion for
Quadruped Robots* (RSS 2018, arXiv 1804.10332), is the canonical treatment of the
two knobs you swept here: it closes the Minitaur reality gap by *system-identifying*
an accurate actuator model and by explicitly *modeling control latency*, the same
delay whose cliff you just measured. Then Hwangbo et al., *Learning Agile and Dynamic
Motor Skills for Legged Robots* (Science Robotics 2019, arXiv 1901.08652), replaces
hand-tuned system identification with a learned *actuator network* on ANYmal, the
production answer to the model-mismatch slider. Read them for the vocabulary that
turns the mass, delay, and noise sliders you just swept into quantities an engineer
measures and compensates for on a real robot.
