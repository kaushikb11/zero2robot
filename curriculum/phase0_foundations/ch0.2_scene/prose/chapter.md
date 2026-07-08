# 0.2: Bodies, Joints, and MJCF

## See it work

There's a switch on this scene labeled "weld." Leave it on and watch: the red pusher slides north into the blue T-block and shoves it clear across the table, and the T travels like what it is — one solid object, bar and stem locked together. Now flip the switch off and run it again. This time the "T" never even gets assembled: the two rectangles settle with a 0.0912 m gap where the weld should hold them at 0.06 m, sitting as two loose pieces before the pusher has so much as touched them.

Nothing about the physics changed between those two runs. What changed is one decision in the scene description — whether the T's two rectangles are one body or two — and that single decision is most of what this chapter is about. Last chapter you stepped a world someone handed you. This chapter you write the world, and MJCF, MuJoCo's XML for describing worlds, has firm opinions about what's attached to what.

## The problem

You can step a simulation now. You did it last chapter: load a model, call `mj_step`, read `qpos`. But the model came pre-built — a floor, a cube, a pusher, geometry chosen for you. That's fine exactly once. The moment you want the actual task this book trains on — a T-shaped block, a specific goal pose, a pusher that moves in a plane instead of on a rail — nobody hands you the XML. You write it.

And MJCF is not a drawing. It's a description of a *kinematic tree*: bodies nested inside bodies, each connected to its parent by joints that spell out how it's allowed to move. Get the tree right and physics does the rest for free — contact, friction, momentum, all of it. Get it wrong in a way that still parses, and you get a scene that runs, produces numbers, and lies to you: a T-block that isn't rigid, a block that tips into the third dimension when it should stay flat, a pusher that can only move along one axis. None of those throw an error. They just quietly do the wrong physics.

So this chapter builds the PushT scene from an empty file — about 150 lines of MJCF and the Python to load it — and then does the thing that separates authoring from guessing: it reads back the tree MuJoCo actually compiled, and pushes the block to prove it moves the way you meant.

## Build

The scene grows one part at a time, and the file is organized to match: a region per part, in the order you'd build them if you were setting a table. Ground first, then the things that sit on it.

### Setup

One thing to notice up front: there's a single random number generator, and it seeds exactly one thing — where the block starts.

```
[include-by-region: scene.py#setup]
```

The flags follow the convention you saw last chapter: free-tier defaults first, a `--smoke` mode that runs a fixed length so CI can compare two runs byte-for-byte, and `--out` for artifacts. Two are new. `--break split-tee` is the deliberate-failure switch from the hook, and we'll pull it in Break It. And `--seed` here plays the role a *reset* plays in a real task: it draws the block's starting position and angle, the way PushT drops the block somewhere new at the start of every episode. Physics itself needs no seed — same scene, same inputs, same trajectory on CPU — so routing that one random choice through one generator is all it takes to keep the whole file reproducible.

### Ground

The table is the one geom in this scene that isn't quite what it looks like.

```
[include-by-region: scene.py#ground]
```

It's a plane at z=0, but its collision is switched off — `contype` and `conaffinity` both zero. The block never actually touches it. That sounds backwards until you remember what we're simulating: a flat pushing task, quasi-static, where a real robot would feel sliding friction against a tabletop. Rather than model true 3D contact between block and table — which is genuinely hard, and which chapter 3.x spends three chapters earning — we fake the tabletop friction with joint `frictionloss` and `damping` on the block itself, and let the table be a backdrop. The four walls, by contrast, are real collision geometry: they box the block into a 0.40 m workspace so a hard shove can't send it off the edge into the void.

### Tee

This is the region the whole chapter turns on, so read it slowly.

```
[include-by-region: scene.py#tee]
```

The T-block is **one body carrying two geoms** — a wide bar and a tall stem, offset so their union looks like a T. That "one body, two geoms" is the entire idea. When two geoms share a body, MuJoCo treats them as a single rigid object: they share one set of joints, and they can never, under any force, move relative to each other. There is no explicit `<weld>` tag here — the rigidity is a consequence of the tree structure. Putting both geoms in one body *is* the weld. You'll feel the alternative in Break It, and it's not pretty.

The joints are the other half of the region. The block connects to the world through three of them — a slide along x, a slide along y, and a hinge about the vertical axis — and that specific set is called a *planar joint*: it grants exactly the three degrees of freedom a flat task needs. The block can glide anywhere on the table and spin in place, and it has no way to lift, tip, or roll, because you gave it no joint that permits those. Its state is three numbers, `[x, y, yaw]`. Degrees of freedom aren't something physics decides; they're something you grant, one joint at a time, and granting the wrong ones is a bug you'll write more than once (exercise 1 makes you feel it).

### Pusher

The pusher is the robot, such as it is, and it's deliberately less capable than the block.

```
[include-by-region: scene.py#pusher]
```

It's a cylinder on two slide joints — x and y — and notice what's missing: no hinge. The pusher cannot rotate, because you didn't give it a joint that lets it. The block can spin; the pusher can't. That asymmetry is the whole shape of a robot-in-a-world: the block goes wherever physics sends it, but the pusher can only do what its actuators allow.

And the actuators are worth a careful look, because they're *velocity* servos, not raw motors. `ctrl` is a target speed in meters per second, and `kv` is the gain — how hard the servo works to hit that speed. Commanding `vy = 1.0` means "try to move north at 1 m/s," and MuJoCo solves for whatever force achieves it, up to the `forcerange` cap. That's the interface a policy will write to in chapter 1.1: not forces, but desired velocities. Exercise 3 sweeps `kv` and shows you what a mushy gain does to a push.

### Target

The goal is the simplest body in the scene: it does nothing.

```
[include-by-region: scene.py#target]
```

It's a translucent green T with no joint at all, which means it's welded to the world and never moves, and its geoms have collision switched off, so the block passes right through it. It's there to be looked at and, later, measured against — the task's reward is the distance between the block's pose and this one. The `<site>` inside it is a massless, collision-free marker: a named reference frame you attach for bookkeeping, and something you'll lean on constantly once chapter 0.3 makes you fluent in frames.

### Build

Now the parts become a world.

```
[include-by-region: scene.py#build]
```

The regions concatenate into one MJCF string, `from_xml_string` compiles it, and — this is the payoff — the next few lines read the tree *back out of the compiled model*, not out of the XML you think you wrote. That distinction matters: the XML is your intent; the compiled `mjModel` is what MuJoCo actually built, and when they differ, the model is right and you are wrong. So we print it. The kinematic tree comes out as three bodies under the world — target, tee, pusher — each hanging directly off the root. Five joints, in `qpos` order: slide, slide, hinge, slide, slide. `nq = nv = 5`: five position numbers, five velocities, and — unlike last chapter's free-jointed box — no quaternion in sight, so no mismatch between the two counts. Two actuators. That printout is the first thing you check when a scene misbehaves, and it's how you'll catch Break It in the act.

Then a short, deterministic demonstration: set the block's start pose from the seed, let it settle for a few steps, and drive the pusher north into it. The block travels from y≈0.031 to y≈0.376 — the pusher is actuated, the block is not, and yet the block moves, because contact does what contact does. And the whole time, we track one number: the distance between the bar geom and the stem geom, in world space. In a properly welded T that number is 0.06 m and it never changes, no matter how hard the push. That constant is the weld, made into a measurement.

## Run it

```
python scene.py
```

<!-- wall-clock table auto-rendered from wallclock.csv -->

It prints the kinematic tree, runs the push, and drops a recording at `outputs/ch0.2-scene/scene.rrd`. Open it:

```
rerun outputs/ch0.2-scene/scene.rrd
```

Scrub the `sim_time` timeline. You should see the block sitting at its seeded start, then the pusher gliding up from the south and driving it north as one clean rigid piece, bar and stem locked, spinning slightly as it goes. The three entity paths are the same ones every chapter uses — `world/objects/tee`, `world/objects/target`, `world/robot/pusher` — so the debugging habits you build here transfer straight through to the capstone.

If the tree printout doesn't say three bodies and five joints, stop and read your XML against the compiled model before trusting anything downstream — a scene that compiles wrong is a scene that lies quietly.

## Break it

Here's the failure the whole chapter has been pointing at. Author the T as **two** bodies instead of one:

```
python scene.py --break split-tee --no-rerun
```

`--break split-tee` puts the bar and the stem in separate bodies, each with its own planar joint set. Everything still parses. Everything still runs. But look at the tree printout: it's a *different scene* now — four bodies where there were three, eight joints where there were five, `nq = nv = 8`. You didn't mean to add three degrees of freedom, but you did, because every body needs its own way to move and you gave the stem its own.

And the weld invariant gives it away — before the push even lands. In the welded scene the bar–stem gap holds at 0.06 m start to finish; its deviation from 0.06 is exactly zero at every step of the run. In the split scene the gap is already wrong at rest: the block settles at 0.0912 m, half again the 0.06 m it should be.

Here's the actual mechanism, and it's worth getting right, because the reason is more mundane than "it came apart." The reset seeds the *bar* body's joints — it nudges the bar a few centimeters north to its start pose, exactly as a PushT reset would. But the stem is a separate body now, with its own joints, and nobody seeded those. So the bar walks north to its seeded position while the stem stays home, and the "T" settles as two rectangles a gap apart. It can't even be *placed* as one piece: assembling it correctly would mean positioning two bodies, and the reset only knew about one. That settled gap is the signature to memorize — when a thing you meant to be one rigid object can't even be set down as one, the rigidity was never there. You drew a T, but you built two blocks.

What happens *during* the push is a red herring, and the code is careful not to be fooled by it. The pusher comes up from the south, meets the loose stem first, and just shoves it around near the bar; the two halves actually drift back toward 0.06 m for a moment before the run ends. So the rigidity verdict never looks at the final gap — it watches the *peak* deviation from 0.06 over the whole run. For the welded T that peak is 0.0 (the gap is 0.06 at every single step); for the split T it is 0.031 m or more. Two loose pieces can wander back together by accident, but a gap that was *ever* off 0.06 was never a weld.

The diagnosis walk in rerun: put the welded and split recordings side by side and scrub to the very first settled frame, before the pusher moves. Welded, the stem sits locked to the bar at 0.06 m and stays there through the whole push. Split, the stem is already sitting apart from the bar in that first frame — the weld gap is visibly wrong before anything gets pushed. Same geoms, same code — the only difference is which body each geom lives in, and that's the difference between a rigid object and a pile of parts. When a later chapter's gripper or multi-link arm behaves like it's made of loosely associated pieces, this is the first thing to check: are the parts you meant to weld actually sharing a body?

## Exercises

Three ways to convince yourself you can author a tree and predict what it will do — predict before you run. <!-- rendered from exercises/ -->

## What's next

You built the scene, and you proved it's rigid, but you did it by watching a distance stay constant — a crude proxy for the thing you actually care about, which is *pose*: where the block is and how it's turned, in what frame, relative to what. The moment you tried to describe the block's orientation you reached for a single number, `yaw`, and got away with it only because this task is flat. Real robots aren't flat. The instant the block can tip, or the camera sits at an angle, or the gripper approaches from the side, you need rotations that don't collapse to one number, and you need to know which frame every quantity lives in. That's the next chapter, and it's where the quaternion you dodged twice now finally comes due.
