# 0.1: The Simulation Loop

## See it work

Grab the red box and drag it somewhere else: behind the pusher, on top of the pusher, half off the table. Let go. It falls, lands, and the blue pusher keeps grinding forward and shoves it wherever it now happens to be. A few seconds into the replay you'll see the box lurch sideways on its own: that's a force we injected mid-run, and the simulation absorbs it without complaint, the same way it absorbed your mouse.

Nothing in this scene is smart. There is no policy, no goal, no reward. What you're looking at is the bare loop that every chapter of this book (behavior cloning, diffusion policies, 4096 parallel quadrupeds) runs inside. This chapter is about seeing that loop with nothing stacked on top of it.

## The problem

You've seen simulated robots before: locomotion clips, manipulation demos, the works. Here's a question that separates watching from building: when a simulated box rests on a simulated table, what is the computer actually doing, hundreds of times a second, to keep it there?

If your answer is a shrug, everything downstream stays fuzzy. When a training run in chapter 2.4 produces a quadruped that vibrates through the floor, you need to know whether to suspect your reward or the physics. That debugging session, and a dozen like it later in the book, bottoms out in the same two data structures and one function call.

So that's what we build: a complete, runnable simulation (load a world, step it through time, poke it mid-run, and read out what happened) in about 160 lines, and the loop at the heart of it is under twenty.

## Build

MuJoCo splits the world into two objects, and the split is the single most important idea in this chapter. **mjModel** is everything that never changes while the simulation runs: geometry, masses, joint layout, actuator gearing. It's compiled once from a description file and then treated as read-only. **mjData** is everything that does change: positions, velocities, contact forces, the current time. `mj_step` reads the model, mutates the data, and that's the whole game. One consequence you'll cash in later: a single model can drive many independent datas. Hold that thought until chapter 2.3, where it becomes 4096 robots training at once.

### Setup

One thing to look for: there's a random number generator here, and it is the only source of randomness in the entire file.

```
[include-by-region: sim_loop.py#setup]
```

The flags follow a convention you'll see in every chapter: free-tier defaults first, and a `--smoke` mode that runs short and fixed so CI can compare two runs byte-for-byte. The `--seed` flag controls exactly one thing in this file: the direction and strength of the shove we'll set up below. Physics itself needs no seed: given the same model and the same inputs, MuJoCo on CPU produces the same trajectory every time. Determinism isn't something we add; it's something we protect, by routing every random choice through one seeded generator.

### Scene

The scene is handed to you this chapter. In 0.2 you'll write your own from scratch. Read it top to bottom anyway; it's short.

```
[include-by-region: sim_loop.py#scene]
```

Three things live in this world. A floor. A red box connected to the world by a `freejoint`: all six degrees of freedom, which is the XML way of saying "loose on the table". And a blue pusher that can only slide along x, because that's the single joint we gave it, driven by a single motor. That asymmetry is deliberate: the box can go anywhere physics sends it; the pusher can only do the one thing its actuator allows. Robots live on the pusher's side of that line.

Below the XML sit the two constructor calls that give this chapter its title: `MjModel.from_xml_string` compiles the description, `MjData(model)` allocates the state. Note the one honest asterisk on "never changes": we overwrite `model.opt.timestep` from a flag right after compiling. The model is frozen *during* stepping; between runs it's yours to edit. Break It below abuses exactly this.

### Perturb

Before the loop starts, we decide how we're going to interfere with it.

```
[include-by-region: sim_loop.py#perturb]
```

`xfrc_applied` is mjData's slot for external forces: a force and torque you inject on any body, added on top of whatever physics is already doing. It's how a mouse-drag in the viewer works, and it's how we'll simulate disturbances for the rest of the book. The number is worth a look: the box weighs half a kilogram, so at MuJoCo's default friction coefficient of 1.0 the floor resists sliding with roughly 5 N (mass × g × μ), and we sample a shove between 6 and 12 N: enough to win, not enough to launch it.

One property of `xfrc_applied` matters more than all the others: it persists. MuJoCo does not clear it after a step. Set it once and it pushes forever, which is either exactly what you want or a bug you'll hunt for an afternoon. Exercise 2 makes you find that bug on purpose.

### Loop

Here it is: the loop the whole book runs inside.

```
[include-by-region: sim_loop.py#loop]
```

Read the body of the `for` loop as a rhythm you'll repeat for 33 more chapters: write your intent into `data.ctrl`, write any outside interference into `data.xfrc_applied`, call `mj_step`, look at what changed. Today `ctrl` is a hardcoded 1.0: full throttle on the slide motor. In chapter 1.1 a neural network writes that line instead, and nothing else about the loop changes.

Why `mj_step` and not our own integration? Because a step is not `position += velocity * dt`. Inside that one call MuJoCo detects collisions, solves for contact forces, applies actuator torques, and only then integrates. You could hand-roll the integration for a floating box; the moment two things touch, you couldn't. (In chapter 3.3 you *will* hand-roll it, and contact will take three chapters; that's the point.)

Two small things in the loop repay attention. First, `mj_forward` before the loop. `mj_step` computes the same derived quantities (body poses, contact lists) but only as a side effect of advancing time; `mj_forward` computes them *without* stepping, so the first frame we log is the true rest pose at t=0, before anything has moved. It's how you inspect a world you haven't stepped yet, and you'll reach for it whenever you need to read a freshly-loaded state. Second, the `.copy()` when we snapshot the box position: `data.xpos` is a view into mjData's memory, and the next `mj_step` overwrites it in place. Saving a reference instead of a copy is the other classic first-week bug.

The rerun calls log the two moving bodies against a `sim_time` timeline, into `world/objects/box` and `world/robot/pusher`, the same entity paths every chapter uses, so the debugging tool you learn here is the one you'll still be using at the capstone. MuJoCo hands us quaternions as wxyz and rerun wants xyzw; the reindex on that line is the first of many small format seams you'll cross in robotics, and not the last quaternion convention issue in this book. Chapter 0.3 is about the rest of them.

### Inspect

The run is over. Where did everything end up?

```
[include-by-region: sim_loop.py#inspect]
```

The box was declared first in the XML, so its free joint owns the first slots of the state vectors: `qpos[0:7]` and `qvel[0:6]`. Seven position numbers, six velocities. That mismatch isn't a bug: a quaternion carries four numbers but only three degrees of freedom, so positions need one more slot than velocities. For now, treat it as a rule; chapter 0.3 lives inside that gap. The pusher's single slide joint takes the next slot, `qpos[7]`, and that's the entire state of this world: eight numbers.

The metrics file is deliberately boring: final positions rounded to six decimals, written with sorted keys. Boring is the feature: run the smoke config twice with the same seed and the two files are byte-identical, and CI holds every chapter in this book to that standard.

## Run it

```
python sim_loop.py
```

<!-- wall-clock table auto-rendered from wallclock.csv -->

The run prints where the box ended up and drops a recording at `outputs/ch0.1-sim-loop/sim_loop.rrd`. Open it:

```
rerun outputs/ch0.1-sim-loop/sim_loop.rrd
```

Scrub the `sim_time` timeline. You should see the pusher close the gap, the box slide ahead of it, and, halfway through, the sideways lurch of the shove, with the force arrow visible under `world/objects/box/shove` for exactly 0.1 s. With seed 0 the shove is +9.8 N in y and the box ends up about 13 cm off its original line. Change `--seed` and the shove changes with it; nothing else does.

If you don't see the lurch, first check you didn't pass `--no-perturb`; second, look at the entity tree: if `world/objects/box` isn't there, you're looking at an old recording.

## Break it

The timestep is 0.002 s: five hundred `mj_step` calls per simulated second, two thousand for the four seconds we just ran. Steps cost wall-clock, so a bigger timestep is the obvious economy: fewer steps to buy the same span of simulated time. Let's take that economy 25× too far:

```
python sim_loop.py --timestep 0.05 --no-perturb
```

`--no-perturb` matters here: we strip out our own interference so that anything strange is the simulator's doing, not ours. And it is strange. Scrub the recording. The moment the pusher reaches the box, the box *hops into the air* off a flat, straight, horizontal push (z peaks around 0.11 m against 0.05 at rest) and by the end of the run it has drifted a quarter of a meter sideways, despite the fact that no lateral force exists anywhere in this scene. The energy came from nowhere. That's the signature to memorize: motion with no force to explain it means the integrator is manufacturing energy, and it manufactures it at contacts, where forces are stiffest and big steps hurt most.

The diagnosis walk in rerun: pick the box in the entity tree, watch its z track. At 0.002 s it's a flat line at 0.05 after settling. At 0.05 s it spikes at first contact. Same model, same scene, same code: the only thing that changed is how far each step extrapolates before physics gets a say. When a later chapter's robot jitters, dances, or launches itself, this is the first dial to check, and now you know its failure signature on sight.

(Why 0.002 and not 0.0002, then? Ten times the steps for the same simulated second. The timestep is a genuine trade (stability against wall-clock) and exercise 1 makes you find the edge yourself before you're allowed to trust the default.)

## Exercises

Three, in `exercises/`: a bug-hunt where the metrics come out almost right and only your model of what persists across `mj_step` explains the gap, and two where you commit to a prediction before the run is allowed to answer. <!-- rendered from exercises/ -->

## What's next

You stepped a world, but you didn't build one: the XML at the top of this file was handed to you, and it's a toy: one box, one pusher, geometry chosen by someone else. The moment you want a different table, a goal region, or a T-shaped block instead of a cube, you're editing MJCF yourself, and MJCF has opinions about bodies, joints, and what's attached to what. Next chapter you build, from an empty file, the exact PushT scene the following fifteen chapters train on.
