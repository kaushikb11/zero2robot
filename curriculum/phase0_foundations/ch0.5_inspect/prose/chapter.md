# 0.5: Seeing Like a Robot (rerun)

## See it work

Point it at the dataset you made last chapter and scrub:

```
python inspect.py --dataset ../ch0.4_record/outputs/ch0.4-record/dataset
rerun outputs/ch0.5-inspect/inspect.rrd
```

Drag the `sim_time` slider and your recording comes back to life: the pusher creeps behind the T, shoves, corrects, and walks the block onto the green target; then the timeline jumps and the next demonstration starts from a fresh spawn. In the panel on the side, two little traces, `eval/pos_err` and `eval/ang_err`, slide toward zero exactly when the block settles. That is your dataset, and for the first time you are *looking* at it instead of trusting that it exists.

You are also, right now, seeing like a robot: everything on screen was reconstructed from ten numbers per frame. Not a video, ten floats. The block's position, its orientation, the target. This chapter is about reading those numbers the way the robot does, and it is the single most useful debugging skill in the book: every chapter after this one logs to rerun, and every mysterious training failure you will ever hit starts with opening the `.rrd` and asking "wait, what did the policy actually see?"

## The problem

You ended 0.4 with a dataset and an uneasy question: is it any good? You recorded some episodes, but a directory of parquet files answers none of the things you actually care about. How many of your demonstrations reached the target? How long are they? Is the orientation you *think* you recorded the orientation that's actually in there? "I collected data" and "I know what's in my data" are different sentences, and the gap between them is where silent training failures live.

There is a specific trap hiding in that gap. The dataset does **not** store success. It does not store reward. Chapter 0.4 wrote exactly two things per frame, `observation.state` and `action`, because that is all a policy trains on. So the success rate you want is not a column you can read off; it is something you have to *reconstruct* from the observations, by applying the same rule the environment used. To know your data, you have to be able to decode it. And decoding has sharp edges: the block's orientation, for one, is stored not as an angle but as a `(sin, cos)` pair, and there is exactly one correct way to turn that back into an angle.

`inspect.py` is the tool for all of this: load a LeRobot v3 dataset back, read its structure, reconstruct the success hiding in it, and replay it to rerun so you can read an episode like a story.

## Build

The shape of the file: **load, reconstruct, replay.** Load a dataset into plain numpy, reconstruct the errors and the success signal the dataset never stored, and log the whole thing to rerun on the canonical entity paths. There is also a small provisioning helper so the chapter runs even before you have data of your own.

### Setup

One thing to notice before anything else: the file has to defend its own name.

```
[include-by-region: inspect.py#setup]
```

This artifact is called `inspect.py`, and `inspect` is a standard-library module. When you run `python inspect.py`, Python helpfully puts the script's own directory first on the import path, which means the next time *anything* (numpy, mujoco) does `import inspect`, it finds this file instead of the standard library and everything falls over. So the very first act of the file is to remove its own directory from the path and put the repo root there instead. It is a small, deliberate seam, called out here rather than hidden, because a debugging tool that trips over its own filename teaches the wrong lesson. With that done, we import the PushT environment for one reason: its success tolerances. We are going to reconstruct "did this reach the target," and we must use the *same* `POS_TOL` and `ANG_TOL` the recorder's world used, not a second copy that could drift.

### Load

Reading a dataset back is two facts, and no more.

```
[include-by-region: inspect.py#load]
```

The first fact is the **schema**: what every frame is. `observation.state` is ten floats, `action` is two: the contract you held in 0.4, read back from `meta/info.json`. The second fact is the **episode index**: lerobot's `meta.episodes` carries, for each demonstration, the `[from, to)` range of rows it occupies. That is the entire definition of an episode, a contiguous run of frames, and it is the fact everyone forgets exactly once (there's a `--break` for forgetting it, below). We slice the frame table by those ranges into plain numpy arrays so the rest of the file never has to touch a tensor.

### Success: seeing like a robot

Here is the heart of the chapter.

```
[include-by-region: inspect.py#success]
```

The orientation of the block is stored as `sin(yaw)` and `cos(yaw)`: the sin/cos trick from 0.3, which exists precisely so the value never jumps as the angle wraps past ±π. The one correct way back to an angle is `atan2(sin, cos)`, arguments in that order. Swap them and you get the *reflected* angle, and every orientation you read from then on is wrong: the block looks rotated in the replay and the error you compute against it is garbage. That is not a hypothetical; it is `--break yaw-swap`, and it is the whole reason "seeing like a robot" is a skill and not a slogan.

With the decode in hand, `frame_errors` reconstructs the environment's `pos_err` (how far the block is from the target) and `ang_err` (how far its yaw is from the target's) from a single observation. And `episode_reached` answers the question you came for: did this demonstration reach the goal? Note *how* it answers. The recorder stops the instant the env latches success (it breaks on `done`), so the last frame it stored is the block sitting on the target. Reading "reached" is therefore reading whether the episode *ended* in tolerance. The dataset never says "success"; this decode is the reading.

### Inspect

The walk turns a list of episodes into the handful of numbers you actually wanted.

```
[include-by-region: inspect.py#inspect]
```

Lengths, the reconstructed success rate, and each demonstration's terminal error. This is what "looking at your dataset" concretely means, and it is the same summary you will want for every dataset in the book, before you train on it, so a bad batch of demos fails your eyes here instead of your policy three chapters later.

### Rerun

And then we replay it, so you can read it rather than tabulate it.

```
[include-by-region: inspect.py#rerun]
```

Every stored observation is logged onto the same entity paths every chapter uses (`world/objects/tee`, `world/robot/pusher`) on the `sim_time` timeline, with `policy/action` and the reconstructed `eval/pos_err` / `eval/ang_err` riding along as scalars. Because it uses the *same* decode as the reconstruction, `--break yaw-swap` doesn't just change a number, it visibly rotates the block in the viewer, so the picture and the numbers disagree in front of you. That is the debugging instinct this chapter is trying to install.

## Run it

With no dataset of your own yet, it makes a small one to inspect, six scripted-expert demonstrations, with a little noise turned on because real teleop is never clean:

```
python inspect.py --episodes 6
```

<!-- wall-clock table renders from wallclock.csv -->

It prints the schema, the episode lengths, and the reconstructed `reached target: N/6`, then drops an `.rrd` you can open with `rerun`. Pointing it at your real 0.4 dataset is the same command with `--dataset`. With seed 0 the six-episode stand-in records lengths `[65, 89, 136, 52, 82, 143]` and all six reach the target: a cleaner run than the one you recorded by hand will be, which is exactly why you learn to check.

## Break it

Two ways to read the same, correct data wrongly, each with an honest signature you can measure:

```
python inspect.py --episodes 3 --break yaw-swap
python inspect.py --episodes 3 --break drop-boundaries
```

`yaw-swap` decodes the orientation as `atan2(cos, sin)`, the reflected angle. Nothing about the recorded data changes; only the reading. The signature is unmistakable: the schema and every episode length are byte-identical to a correct run, but the reconstructed success rate collapses from 3-of-3 to 0-of-3, because every block now reads as rotated ~90° off, and its `ang_err` sails past `ANG_TOL`. The block in the rerun replay points visibly wrong. This is the failure mode of trusting a decode you didn't check.

`drop-boundaries` forgets that demonstrations *have* boundaries: it ignores the episode index and treats the whole frame table as one long episode. The signature: your three episodes collapse into one long one, the "success rate" becomes a meaningless 1-out-of-1, and in the replay the block teleports between demonstrations because the timeline never resets. It is the bug you write the first time you load someone else's dataset and assume it's one trajectory.

## Exercises

Three ways to make sure you can *read* a dataset, not just open one. Predict before you run.

## What's next

You can now record a dataset (0.4) and read it (0.5), which means you have, for the first time, everything a policy needs: examples, in a known format, that you've actually looked at. Phase 1 is where those examples become a policy. Chapter 1.1 takes this exact dataset and fits the dumbest thing that works, behavior cloning, and the very first thing it will do when the policy misbehaves is send you back here, to open the `.rrd` and ask what the policy saw. The inspector you just built is not a Phase-0 detour; it is the instrument you debug the rest of the book with.
