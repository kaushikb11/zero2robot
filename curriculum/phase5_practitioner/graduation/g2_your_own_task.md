# G2 — Define and Build Your Own Task

<!-- graduation-bridge module: prose only. No artifact, no exercises, no toy, no wall-clock, no meta.yaml. -->

## The leap this module exists to make un-scary

Every task in this course was ours. PushT, the ALOHA bimanual rig, the cartoon
quadruped — we picked them, we wrote the scripted expert, we recorded the demos,
we set the tolerance that decides success. You built the policies, but you built
them against a target someone else had already scoped, measured, and made
reproducible. That scaffolding is invisible until it is gone.

Now you want to make a robot do a thing *you* care about — sort your resistor
bins, fold a specific towel, stack your own blocks — and every one of those
authored decisions lands back on your desk at once. What is the task, exactly?
How will you know it worked? Where does the data come from? Which of the eight
algorithms you built do you even reach for first? This is the step that humbles
people, and it is the step almost no course teaches. Karpathy's *Neural
Networks: Zero to Hero* is the best from-scratch ML course in existence, and it
ends the way most do: you have built the mechanisms, now go build something —
the "go build something" left as an exercise. That gap between owning the
mechanisms and shipping your own project is exactly where most learners stall.
This module is the scaffold across it. Not new machinery — you have all of it —
but the loop, made explicit, run on *your* task instead of ours.

## The scaffold: a loop you run on your own task

Every step below is a lesson you already did, pointed at a target you chose.

**1. Pick a scoped, measurable task.** The single most common beginner mistake
is picking a task you cannot measure or cannot afford. "Make my robot tidy my
desk" is not a task; it is a research program. "Push a single wooden cube into a
10 cm target square, from starts anywhere in a 30 cm annulus" is a task — it fits
in sim or on the cheap arm from **G1**, it has an unambiguous success test, and
it is small enough that fifty demos is a meaningful dataset. Scope down until you
can say, in one sentence, what the robot does and when it has done it. You can
always widen later. You cannot debug a target you cannot state.

**2. Define success and an honest eval FIRST — before any training.** This is
chapter 1.6, and it goes first for a reason: you cannot tell whether an idea
helped until you can measure the thing it was supposed to help. Write the seeded
eval suite before you write the model. Decide the number of held-out rollouts,
fix the reset seeds so they are new starts by construction, and — this is the
part 1.6 drills — report a rate *with its interval*, never a hero rollout. A
success rate over twenty episodes is a coin flipped twenty times; the band around
it is embarrassingly wide, and "0.40 beat 0.25" may be pure noise. Build the
Wilson interval into your eval on day one and every later claim you make about
your own policy carries its error bar for free.

**3. Collect and curate data — because the data is the policy.** You produce the
demos with the **G1** arm or the ch0.4 teleop loop, in the same LeRobot format
every chapter consumed. Then remember chapter 1.2: *the data is the policy, and
bad demos poison it.* The runs where you fumbled, chased the object into a wall,
and timed out with the task half-done are labeled `(observation, action)` pairs,
and behavior cloning believes labels. The strongest filter is the bluntest one —
keep the episodes that reached the goal, drop the ones that did not — and 1.2's
warning stands: a quality *signal* is not a quality *objective*. Do not curate on
some clever internal-agreement score; your filter will find the tidy, easy subset
and hand you a policy that dies the instant the task gets hard.

**4. Baseline with the dumbest thing that works.** Chapter 1.1: fit a three-layer
MLP to your demos with MSE and roll it out. Do this *before* anything fancier. It
is fast, it is honest, and its failures are diagnostic. It will covariate-shift —
drift a little, land in a state no demo covered, guess worse, drift further — and
watching *where* it breaks tells you what to do next. Only escalate when the data
and the task demand it: reach for ACT or diffusion (1.3, 1.4) when your demos are
genuinely multimodal and the MLP is averaging two good actions into a bad third;
reach for a fine-tuned VLA (the P1 reading track, **G1**) when the task needs
language or broad visual generalization the MLP cannot carry. A bigger model on
top of a covariate-shift problem is wasted money.

**5. Iterate with corrections where it fails.** When the baseline drifts off the
demo manifold, chapter 4.2 is the fix that does not need a reward function: roll
out your policy, record the drifted states it actually visits, ask the expert
(your hand on the teleop, this time) what to do *there*, aggregate, retrain. You
are fixing the *distribution*, not the model — labeling the states your policy
fails in, which are exactly the states your original demos never covered. Watch
for 4.2's other lesson: more rounds is not strictly better, so select the best
round on held-out eval, do not just take the last.

**6. Reward design, only if you go RL.** If your task has a reward and no expert,
chapter 2.4 applies — and its warning is the load-bearing one. *The reward is the
program you write for the robot, and it does exactly what you wrote, not what you
meant.* Reward "torso height" hoping for a walk and PPO will hand you a robot that
rears up and goes nowhere, its reward climbing 10× while the distance you actually
wanted stays at zero. Shape the path, not just the destination, and measure the
behavior you care about *separately* from the reward that trained it.

## The pitfalls are the course's lessons, aimed at you

**Covariate shift** (1.1). Your policy chooses its own future inputs. One slightly
off action leads to a slightly off state where the policy is slightly worse, and
the spiral compounds. No loss curve can see it, because loss is computed on states
the *demonstrator* chose. Only a rollout eval — or DAgger corrections — catches it.

**Reward hacking** (2.4). If you go RL, your agent will optimize the literal number
you wrote. A plausible-sounding proxy is a trap; the fix is never "optimize harder"
(that games the proxy better), it is a truer program. Always eval the intended
behavior, not the reward.

**Overfitting tiny data** (1.2 / 3.7). Your first dataset will be small, and a small
dataset tiles almost none of the state space. A network that nails your training
frames can fail everywhere else, and capacity spent past "can represent the
demonstrator" only buys a sharper copy of the same mistakes. More *valid* coverage
beats a bigger model — this is the whole of 3.7's data-engine argument.

**The eval that lies** (1.6). The most dangerous number is the one that looks fine.
A single twenty-episode suite can read 0.30 when the true rate is 0.23 — you got a
lucky twenty. Normalization stats computed on the easy week describe a world that
ends where the easy demos ended, and every curve looks healthy while the policy is
quietly broken. Seed your suites, pool your episodes, and put a band on it.

**Fabricating numbers.** The failure mode no algorithm causes and every practitioner
must refuse: reporting a rate you did not measure, a "~90%" from memory, a hero
rollout as if it were a rate. This course pins wall-clock to a CSV and success to a
seeded suite for exactly this reason. On your own project no CI enforces it — you
do. A number you cannot reproduce from a seed is not a result.

## If you don't have a task or data yet

You do not need to invent one from nothing. **Open X-Embodiment** (1M+ trajectories
across 22 robot embodiments, the corpus 3.7 wrangles) and **LIBERO** (the held-out
manipulation benchmark 1.6 reads against) both ship real tasks with real data and
real success criteria — pick one task family and treat its eval as your target. Or
generate your own the way the whole course did: drive the **G1** arm or the ch0.4
teleop loop and record fifty demonstrations of the smallest task you actually want.
Borrowed or homemade, the loop is identical.

## The loop is the whole skill

Your first project will humble you. The task will be vaguer than you thought, the
data messier, the baseline worse, and the eval will tell you something you did not
want to hear. That is not the project failing — that *is* the job. The eight
algorithms were never the skill. The skill is the loop:
**define → data → baseline → eval → iterate**, run honestly, with error bars,
until the robot does the thing. You built every piece of that loop against our
tasks. G2 is where you point it at yours.

**Back to G1** for the real robot the demos and rollouts run on. **Forward to G3**
for how to stay current after this course ends — the field moves weekly, and the
loop above is what stays true when the checkpoints change.
