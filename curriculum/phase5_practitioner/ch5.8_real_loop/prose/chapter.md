# 5.8: The Real Loop — Teleoperate, Record, Train, Deploy on a Real Arm's Body

<!-- objectives: rendered from meta.yaml, do not duplicate here -->
<!-- DRAFT prose (agent-authored). Voice is human-owned; author does the voice pass. -->

## The whole course, run once, on a real robot's body

Every method in this book ran against an environment we wrote. PushT, ALOHA-cube, the
cartpole — small, clean, ours. That was the right call for learning a mechanism: when the sim
is a hundred lines you can read, you always know exactly what the algorithm saw. But it left
one honest question open, and it is the only question Phase 5 exists to answer: *does any of
this survive contact with a real robot?*

This chapter runs the **entire LeRobot loop** — drive, record, train, deploy, evaluate — on
the **SO-101's real morphology**: the same \$150 arm the graduation bridge (G1) sends you to
buy, loaded straight from `google-deepmind/mujoco_menagerie`. Six `sts3215` position-servo
joints, the manufacturer's link lengths, the manufacturer's joint limits, a bundled wrist
camera. Not a simplified stand-in — the real body, in MuJoCo, free-tier, on your laptop. And
the format and the control loop are byte-for-byte what you would run on the metal.

Open `real_loop.py`. Nine regions: **setup**, **fetch**, **env**, **expert**, **record**,
**model**, **train**, **deploy**, **report**. The middle seven are the loop.

## First, fetch the real robot — without committing a single mesh

```
[include-by-region: real_loop.py#fetch]
```

The SO-101 model is an MJCF plus eighteen STL meshes, about 17 MB. Invariant #5 is absolute:
**no binaries in git.** So we fetch it — once — into a gitignored cache, and skip the download
on every run after. Reproducibility without a committed blob comes from three checks, cheapest
first: the two XMLs we actually *parse* are **sha-pinned** (a changed robot definition trips an
assertion, not a silent surprise); the binary meshes are **size-checked** (a rate-limited HTML
error page is a few hundred bytes, a real mesh is tens of KB); and the final proof of a good
fetch is that **MuJoCo compiles the model** — `nu=6, nq=13` or we stop.

We pull through a CDN mirror keyed by a git ref, and that ref is **un-pinned on purpose**.
This is the same doctrine G1 states for the LeRobot CLI: the upstream moves weekly, so we hold
a *template*, not a frozen commit. An author bumps `--menagerie_ref` and the sha-checks tell
them immediately whether the robot definition moved under them.

## The env is the real arm — we do not simplify it

```
[include-by-region: real_loop.py#env]
```

`SO101ReachEnv` is a thin wrapper, and *thin* is the point. Unlike the Phase-1 envs we
authored line by line, here the physics, the kinematics, and the actuators are the
manufacturer's. We add exactly two things: a box, respawned on the floor in a reachable patch
each reset (via its free joint — the same trick ch1.3's env used to place its cube), and an
`obs`/`step` contract that matches PushT and ALOHA so the behavior-cloning loop below is
*unchanged* from ch1.1. The observation is nine numbers — six joint angles and the box's
`(x, y, z)`. The action is six **joint-position targets**: the SO-101's servos track them,
exactly as a real STS3215 does.

## Driving is teleoperation, scripted so CI can diff it

```
[include-by-region: real_loop.py#expert]
```

On real hardware you would grab a *leader* arm and move it; a *follower* mirrors you, and the
joint angles it reports are your actions (that is G1's `lerobot-teleoperate`). We can't replay
a human hand deterministically, so a tiny controller stands in. The reach is almost entirely a
fixed *lower-to-the-table* pose — shoulder-lift, elbow, wrist held at constants — steered by
the **one joint that must know where the box is**: `shoulder_pan`, aimed at the box's azimuth.
It commands the target pose every step and lets the position servos interpolate. That detail
matters more than it looks: a *stable feedback law* (drive toward a fixed goal) is something a
clone can learn from any starting state, where an open-loop *trajectory* (do X at step 3) drifts
the moment the clone's timing slips. Exercise 3 has you write that `shoulder_pan` line.

## Recording writes the real format — the same one hardware writes

```
[include-by-region: real_loop.py#record]
```

This is not a schema we invented and keep in sync. It is `LeRobotDataset.create → add_frame →
save_episode → finalize` — the **actual pinned `lerobot` API**, the identical call `record.py`
made in ch0.4 and `lerobot-record` makes on hardware. Format parity with everything you have
trained on all course is free *by construction*. We record **state-only** (`use_videos=False`)
so the run is byte-reproducible on one CPU — and, honestly, because the missing camera is part
of the reality gap we are about to name. We store the observation we *acted on* (pre-step), the
off-by-one ch0.4 warned you about.

## Training is ch1.1's behavior cloning, retargeted to six joints

```
[include-by-region: real_loop.py#model]
```

The clone never sees the expert. It loads the **recorded dataset back off disk** and fits
`observation → action` with plain MSE — the whole point of the loop is that training consumes
your *recording*, not your recorder. The network is ch1.1's three-layer MLP, retargeted 9 → 6.

One change is load-bearing, and it is a lesson about robots specifically. The six joints move
over wildly different ranges — shoulder-lift swings more than a radian on every reach, while
the box-dependent `shoulder_pan` correction is a few tenths. Compute the MSE on raw joint
angles and the big joints *drown* the tiny pan signal — the network learns to sweep down and
ignore where the box is, and the clone reaches the same spot every time. So we take the loss in
**normalized action space**, where every joint weighs equally and the pan signal survives. (You
will feel the raw-space failure if you look for it; the normalized loss is why the clone works.)

```
[include-by-region: real_loop.py#train]
```

## Deploying is four lines, and evaluating is ch1.6

```
[include-by-region: real_loop.py#deploy]
```

`obs → policy → d.ctrl → step`. That is the entire deployment surface — the sim mirror of the
`get_observation → select_action → send_action` loop G1 shows you on real hardware. Everything
hard you built in Phase 1 exists to make one of those lines good; the rest is the robot.

And we evaluate it the ch1.6 way: a **success rate over held-out box placements** (seeds the
clone never trained on), against two baselines that must fail — a no-op that holds the rest
pose, and a random flail. The headline is a **direction**, not a number: the recorded-then-cloned
policy reproduces the reach *clearly above* both baselines, on every seed. That is the whole
claim — **the loop closes end-to-end on the real arm's body.** It is a claim about the loop,
not about manipulation being solved, and we report the order because the exact rate shifts with
the platform's contact and servo settling (ch1.6). The clone matching the scripted expert is
the loop working, not a performance flex — a scripted reach is an easy, low-variance target,
which is exactly the right bar for a mechanism demo.

## What the twin gives you — and what it cannot

Here is the honest core, and it is the reason this chapter is the *second-to-last* thing you
do, not the last.

**What you just did transfers to hardware unchanged.** You drove a real arm's morphology,
recorded the real LeRobot dataset format, trained a policy on it, and redeployed it through the
real control loop. Buy the SO-101 and every one of those skills is already yours — that is the
entire thesis of G1.

**What the twin cannot give you is the reality gap**, and you must not let a clean 100% here
paper over it. Servo backlash. Friction that isn't in the model. Latency between reading an
observation and the motor moving. Camera noise and lighting and a background that shifts at
6pm. Calibration drift between two "identical" arms. The Menagerie model even tells you this
itself: its servo gains are a *calculated approximation*, **not** the real STS3215 gains — the
MJCF says so in a comment. This chapter is **morphology plus the loop, not torque-level
fidelity**, and it reaches for a box rather than pinching and lifting it precisely because a
frictional grasp is where the gap bites hardest. Those gaps are not a footnote. They are the
whole reason real hardware exists, and closing them is what your weeks on a real arm will be
spent on. They stay reading — that is G1.

## Break it the way the metal breaks

```
python real_loop.py --seed 0 --break obs_swap
```

The most common way a working policy fails on a real robot is not a bad model — it is a **bad
wiring between your recorder and your deploy script.** `--break obs_swap` simulates exactly
that: the deploy code reads `box_x` and `box_y` in the *opposite* order than the recording
wrote them. Training is untouched — same dataset, same clean loss near `1e-5`. And the arm
confidently reaches the *wrong way*, its success collapsing from ~1.00 to ~0.20. No loss curve
could have warned you, because training never saw the swap; the bug lives entirely at the
seam where your policy meets the world. This is the ch0.4 lesson — *record obs must equal deploy
obs* — now with a physical consequence, and it is why G1 insists your calibration `id` matches
across record, train, and rollout. Exercise 2 makes you generate this failure and explain why
every training metric you watched stayed green.

## Where you are

You have run the whole loop, from a blank robot model to a deployed policy, on a real arm's
body. The last gap — the reality gap — is the one thing sim could not hand you, and it is the
one thing worth crossing on real hardware. Two bridges remain: **G1**, where you buy the arm
and run this exact loop on the metal, and **G2/G3**, where you decide what *your* robot should
do. You did not need to wait for a robot to learn robotics. You needed the robot to find out you
already had.
