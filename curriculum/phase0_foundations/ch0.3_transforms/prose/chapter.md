# 0.3: Spatial Reasoning Without Tears

<!-- objectives: rendered from meta.yaml, do not duplicate here -->
<!-- Numbers in this chapter are seed-0 CPU reference-run values; full provenance
     (versions, date) lives in meta.yaml, which the exercise checks read. -->

## See it work

Drag the block frame around the table. Turn it, slide it into a corner. Watch the three little arrows that ride on it: red, green, blue, the block's own x, y, and z. Off to the side there's a single dot, the pusher, sitting still in the world. But look at the number under it: the pusher's coordinates keep changing as you drag, even though the pusher never moves. You're not moving the pusher. You're moving the frame you're measuring it from.

That's the whole subject of this chapter in one gesture. A robot never asks "where is the block?" in the abstract. It asks "where is the block *in the gripper's frame*", "where is the gripper *in the arm's frame*". Every one of those questions is a rotation and a translation stacked on another rotation and translation, and there are exactly three ways everyone gets it wrong. Flip the toggle marked "xyzw" and watch the block's arrows swing to a wrong orientation on a single click. That's bug number one, and by the end of this chapter you'll recognize it on sight.

## The problem

In chapter 0.1 you crossed a quaternion seam without stopping to look at it. One line in the rerun logging said the quiet part out loud:

> MuJoCo stores quaternions wxyz; rerun wants xyzw, hence the reindex.

You reindexed four numbers and moved on. That worked because someone had already figured out which convention each library used and written the swap for you. The moment you're the one composing a block pose with a gripper pose, the moment the transform is *yours*, that borrowed knowledge runs out, and a wrong answer doesn't announce itself. There's no exception, no NaN, no stack trace. The block just ends up pointing the wrong way, or a centimeter off, and your policy trains on quietly corrupted coordinates for six hours before you notice.

The reason this is dangerous is that rotation math has no natural error signal. Add two positions in the wrong order and you'll often see something obviously broken. Compose two rotations in the wrong order and you get *a perfectly valid rotation*, just not the one you wanted. So the only defense is an answer key you trust absolutely. We have one: MuJoCo ships the same operations as C functions (`mju_mulQuat`, `mju_quat2Mat`, `mju_rotVecQuat`), and it's the library that drives every simulation in this book. So that's what we build: the rotation toolkit from scratch, in numpy, checked line by line against MuJoCo until the two agree to the fifteenth decimal, and then, deliberately, each of the three classic bugs, so you can measure exactly what each one costs.

## Build

The file is about 260 lines, most of that the comments that walk through each operation (the code itself is a little over 160), and there's no physics in it at all, no `mj_step`, no `mjData`. It's five short regions: the setup, then the two quaternion operations everything else is built from, then the two ways to spend a quaternion on a vector, then the `Frame`, then a demo that proves the whole thing against MuJoCo and draws it in rerun.

### Setup

One thing to look for: the only randomness in the file is the generator that draws test quaternions, and it feeds exactly one thing: the answer-key comparison against MuJoCo.

```
[include-by-region: transforms.py#setup]
```

The flags match every chapter's shape (`--seed`, `--smoke`, `--out`, `--no-rerun`) plus one that's specific to this chapter: `--break`, which takes the name of a bug to inject. Its default is `none`, and when it's `none` the code is correct and agrees with MuJoCo; the whole Break It section is just this one flag set to something else. The convention decision that governs the entire chapter is stated in the docstring and never revisited: a quaternion is `[w, x, y, z]`, scalar first, the MuJoCo order. Every function below assumes it, and bug number one is what happens when a caller doesn't.

### Quaternions

Two operations generate every rotation you'll ever compute. Here they are.

```
[include-by-region: transforms.py#quaternions]
```

`quat_multiply` is the Hamilton product, written out term by term, and it is worth reading all four lines even though you will never read them again. Two facts hide in it. First, it is *not* componentwise: you cannot multiply quaternions the way you'd multiply two arrays, and numpy will happily let you try and hand back garbage. Second, it does not commute: `quat_multiply(a, b)` is a different rotation from `quat_multiply(b, a)`, in general, and that asymmetry is bug number two waiting to happen. `quat_conjugate` is almost too simple to notice (flip the sign of the vector part), but for a unit quaternion it *is* the inverse rotation, which is the fact the `Frame`'s inverse leans on later. The comment earns its place here: it's only the inverse because the quaternion is unit-length, and every rotation quaternion is, so we never pay for the general case.

### Rotations

There are two ways to actually apply a quaternion to the world, and you'll want both.

```
[include-by-region: transforms.py#rotations]
```

`quat_to_matrix` bakes the rotation into a 3×3 matrix: the form MuJoCo hands back from `mju_quat2Mat`, and the one you want when you're about to rotate a hundred vectors and don't want to pay for the quaternion sandwich each time. Every entry is quadratic in the components; it's the sandwich product collected into a matrix once, algebraically.

`rotate_vector` is the sandwich itself: promote the vector `v` to a quaternion with a zero scalar part, compute `q * (0, v) * conj(q)`, and read off the vector part of the result. That's it. That's what "rotate a vector by a quaternion" *is*, and it's exactly what `mju_rotVecQuat` computes. It's also the single most bug-prone line in robotics, because if the `q` you pass in is in the wrong component order, this function runs without complaint and returns a wrong vector. Hold that thought for four paragraphs.

### Frames

A rotation alone isn't enough: the block isn't just turned, it's turned *and* somewhere. A `Frame` is the pair.

```
[include-by-region: transforms.py#frames]
```

Read the name `world_from_tee` left to right as a machine: it takes a point written in the tee's coordinates and returns that point in the world's. That naming convention is the single most useful habit in this chapter, because it makes composition checkable by eye. `frame_a.compose(frame_b)` is only meaningful when the "from" of `a` matches the "to" of `b`: `world_from_tee.compose(tee_from_pusher)` reads cleanly (the `tee`s meet in the middle); `tee_from_pusher.compose(world_from_tee)` does not, and that mismatch is bug number three, visible in the *names* before you ever run anything.

Three methods carry it. `transform_point` rotates first and translates second, and the comment says why order matters: translate first and you'd rotate the offset along with the point. `compose` multiplies the rotations and sends the child's origin through the parent transform. `inverse` conjugates the rotation and works out where the parent's origin lands once the rotation is undone. The `-rotate_vector(inverse_rotation, translation)` line is the part people get wrong, because it is *not* just negating the translation; the translation has to be rotated into the new frame first.

### Demo

Now we prove it and draw it. The proof comes first.

```
[include-by-region: transforms.py#demo]
```

`max_error_against_mujoco` runs every from-scratch operation on 512 random quaternions and points and records the worst disagreement with the matching `mju_*` function. The result is the whole argument of the chapter in four numbers: quaternion multiply agrees to 2.2e-16, quat-to-matrix to 5.0e-16, rotate-vector to 1.3e-15, and a full `Frame` round-trip (transform a point out and back) to 1.8e-15. Those are all machine epsilon: the distance between adjacent doubles. It's not that our code is *close* to MuJoCo; it's that the two are computing bit-for-bit the same thing, and the tiny residue is just floating-point noise. That's what "you can trust this" means, quantitatively.

Then the concrete question, the one this chapter exists to answer: the block sits at some pose, the pusher is somewhere in the world, and you want the pusher's position *as the block sees it*, in the block's own frame. That's one `inverse` and one `transform_point`, and it's the exact move a reward function makes when it measures error in the block's frame, or a policy makes when its input is relative. With seed 0 the pusher at world `[0.20, 0.10, 0]` lands at `[0.167, 0.031, 0]` in the tee frame, and composing straight back recovers `[0.20, 0.10, 0]` to the last decimal. The round-trip is the point: frames are only useful if going there and back is exact.

The rerun logging draws each frame as its three basis arrows and then, along a short timeline, sweeps an extra 15° yaw onto the tee frame twenty-four times, tracing where the block's long-axis tip lands after each composition. The tip walks an arc: that's "watch compositions" made literal, one `compose` per frame of the timeline.

## Run it

```
python transforms.py
```

<!-- wall-clock table auto-rendered from wallclock.csv (ch0.3-transforms: cpu-laptop, t4, l40s all MEASURED) -->

There is no training here and no GPU to wait on: the whole run is pure numpy against MuJoCo's C functions, and it finishes in about a second on a laptop. It prints the four agreement numbers, the pusher's coordinates in the tee frame, and drops a recording at `outputs/ch0.3-transforms/transforms.rrd`. Open it:

```
rerun outputs/ch0.3-transforms/transforms.rrd
```

You're looking at two coordinate frames (`world/frames/world` at the origin and `world/frames/tee` turned and offset), each as a red/green/blue arrow triple, plus the pusher as a single dot. Scrub the `compose_step` timeline and the tee's long-axis tip traces its arc. What healthy looks like: the world frame's arrows lie exactly along the grid axes, and the tee's are rigidly turned by the same angle you set in the code, all three still at right angles to each other. If any pair of arrows stops looking perpendicular, a rotation has stopped being a rotation. That's your first thing to check, and it's exactly what the next section breaks on purpose.

## Break it

The three bugs everyone writes are all one flag away. Start with the one chapter 0.1 warned you about:

```
python transforms.py --break quat-convention --no-rerun
```

This takes the block's orientation quaternion (correct, in `[w, x, y, z]` order) and reindexes it to `[x, y, z, w]` before handing it to `rotate_vector`, which still reads it as `[w, x, y, z]`. Exactly the wxyz/xyzw seam from 0.1, except now nobody wrote the swap for you. The four agreement numbers are unchanged (the toolkit is still correct), but the convention check that read `0` a moment ago now reads `0.097`. That's the signature to memorize: about ten centimeters of error on a bar only six centimeters long, so the rotated tip misses by more than the bar's own length. A vector that should have turned cleanly now points somewhere else entirely. In the recording (drop the `--no-rerun`) the tee's arrows visibly point the wrong way while the world frame's stay put. No exception, no warning: a confident wrong answer, which is the most expensive kind.

The other two live behind the same flag. `--break compose-order` composes `world_from_tee` and `tee_from_pusher` in the reversed order and measures the gap: `0.091`. The subtle part, and the reason this bug survives code review, is that the two *rotations* here are both yaws about z, so they commute exactly: the resulting orientations are bit-for-bit identical. It's only the *translation* that moves, because composition sends the child's origin through the parent's rotation and doing that in the wrong order puts it in the wrong place. "The rotations match, so the transforms match" is a lie your intuition will tell you, and the 9 cm gap is the price. The third bug (point-versus-frame, transforming a point *by* a frame when you meant to express it *in* that frame) has its own flag, `--break point-vs-frame`: it reads the pusher's world position through `world_from_tee` forward instead of through its inverse, and the pusher lands 14 cm from where it actually is. Same demo, three flags, three silent wrong answers with three different signatures.

## Exercises

Three ways to convince yourself you actually hold the math: predict before you run, hunt the convention bug, and write the rotation yourself. <!-- rendered from exercises/ -->

## What's next

You can now put a point in any frame and get it back out, exactly. But every pose in this chapter was a number you typed. Real poses come from somewhere: a joystick, a SpaceMouse, your own hand dragging a simulated gripper while MuJoCo records where it went. Next chapter you wire up teleoperation, and the first thing you'll do with the stream of poses coming off the controller is exactly what you built here, compose them into the robot's frame, except now they're arriving thirty times a second and they're yours.
