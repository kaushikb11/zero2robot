# 0.4: Teleoperation & Your First Dataset

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## See it work

Grab the pusher and drag it into the back of the T-block. Push. The block slides, catches on a corner, spins a little; you correct, come at it from the other side, and walk it onto the green target. The moment it settles, the counter in the corner ticks over: *episode 1 recorded.* You just made a training example, not by labeling anything, just by doing the task once and letting the recorder watch.

That is teleoperation, and it is where almost every robot dataset in this book begins. No policy here yet, no learning: just you, a mouse, and a recorder writing down what your hand did. What you're looking at is the raw material chapter 1.1 turns into a policy: a list of moments, each one an observation of the world paired with the action you took in it. This chapter is about capturing that list and writing it to disk in the exact format the rest of the book expects.

## The problem

You can build a scene (0.2), read poses across frames (0.3), and step physics forward (0.1). Nothing you've built has *learned* anything, and it can't, because learning needs examples and you don't have any. A policy is a function from observation to action; to fit that function you need pairs: thousands of "the world looked like this, so I did that." Where do the pairs come from? From someone doing the task. From you.

But "record what I did" hides a surprising amount of structure. What exactly is an observation: the pixels, the block's pose, both? What is one action: a mouse position, a velocity, a motor command? When does one episode end and the next begin? And once you've answered all that, how do you write it so that chapter 1.1's trainer, and chapter 0.5's inspector, and the curation pass in 1.2 can all read it without a single line of custom glue?

The robotics community settled these questions with a format (LeRobot v3), and this chapter's job is to record real episodes and write them in it. We build `record.py`: it captures episodes two ways (you driving locally, or a bundle exported from the browser after a real mouse session) and funnels both through one canonical writer, so the dataset you make is byte-for-byte the same shape as the reference datasets the book trains on.

## Build

`record.py` has a spine worth saying out loud before you read a line of it: **two inputs, one output.** You can get episodes by driving the sim locally, or by ingesting a recording the browser made. Both paths end at the *same* four library calls that write the dataset. That shared ending is not a convenience. It is the entire reason your homemade dataset is guaranteed to match the training format, and we'll come back to it.

### Setup

One thing to look for: we import the PushT environment instead of redefining it, and there is exactly one random number generator.

```
[include-by-region: record.py#setup]
```

The environment is shared reference code (decision 004): it owns the obs/action semantics, so importing it means a locally-recorded episode and a scripted-expert episode describe the same world in the same numbers. The flags follow the book's convention, free-tier defaults, a `--smoke` mode that runs short and fixed so CI can diff two runs byte-for-byte, and here two mode switches: `--episodes N` for local recording and `--from-interchange PATH` for ingesting a browser bundle. The single generator seeds both the environment reset and a small "wobble" on the recorded actions, so `--seed 0` twice produces identical data, and a different seed produces a different, still reproducible, dataset.

### Features

Before you record anything, you declare the *shape* of what you're recording.

```
[include-by-region: record.py#features]
```

This is the schema, and it is the one promise your dataset makes to everything downstream. `observation.state` is ten floats (the pusher's position, the block's position and orientation, the target's) laid out in the order chapter 1.1's behavior cloning will slice. `action` is two floats, a target velocity. The names and shapes here are copied, deliberately and verbatim, from `gen_demos.build_features` and `pusht_env.py`. The repetition is the point: when the trainer assumes `observation.state` is `float32[10]`, this is the line that has to agree, and you can see it agree. The yaw comes in as a sin/cos pair rather than a raw angle for a reason you met in 0.3: an angle wraps at ±π and a network hates the discontinuity; sine and cosine are smooth everywhere.

### Teleop

Here is the local recorder, the "you driving" path.

```
[include-by-region: record.py#teleop]
```

Two honesties up front. First, real interactive teleop, a human dragging the pusher and feeling the contact, lives in the *browser* demo at the top of this chapter, not here. A live mouse can't be replayed byte-for-byte, and this book's determinism contract needs `--smoke` to reproduce exactly, so `scripted_drive` stands in for your hand: get behind the block, then shove it toward the target. Second, and this is the important one: the stand-in is deliberately crude. It drives the block's *position* toward the goal but never rotates it into alignment, so it essentially never trips the environment's success latch. Every local episode plays out to the full time limit with the block parked near the target but turned the wrong way. That is not a bug to fix; it is the texture of real teleop data, which is full of near-misses and half-finished intentions. A policy that only ever saw a flawless demonstrator is a policy in for a shock. What matters is that every control step emits a real `(observation, action)` pair, and a list of those pairs is all an episode is.

Read the loop's shape, because it is the shape of every data collector in robotics: record the observation you're looking at *now*, choose an action, then step the world. We store the pre-step observation, so the observation we keep is always the one we acted on, never the terminal frame the episode ended in. That off-by-one is the same bookkeeping `gen_demos` uses, and datasets that get it wrong are a real and annoying bug.

### Ingest

The other input: a recording the browser already made.

```
[include-by-region: record.py#ingest]
```

When you teleoperate in the browser, it can't write a LeRobot dataset directly. The v3 format is a pile of parquet files and quantile statistics that belong to the Python library, and reimplementing all of it in TypeScript would be a standing bug farm (that trade-off is decision 008). So the browser writes something small and stable instead: an *interchange*, a JSON manifest with the episode arrays inline, plus one PNG per frame if you recorded images. `load_interchange` reads that bundle. Notice what it does *not* do: it never hardcodes "ten floats." It reads the feature spec the browser declared and trusts it, which is why this same converter will handle an ALOHA-style scene later without a line changing. The browser, which runs the scene, is the one that knows the shape.

### Write

And here is the ending both paths share.

```
[include-by-region: record.py#write]
```

Four calls: `create` the dataset with your feature schema, `add_frame` for every frame, `save_episode` at each episode boundary, `finalize` to compute statistics and seal it. This is, line for line, the sequence in `gen_demos.py`. That is the whole trick. Because the pinned `lerobot` library does the actual writing (the parquet columns, the `meta/info.json`, the per-feature quantiles, the `CODEBASE_VERSION` stamp), your dataset's format matches the training datasets *by construction*, not because we carefully re-derived it and hope it stays in sync. The one thing to notice in the loop: we never pass a `timestamp`. The library derives it from the frame index and the fps, and handing it one would be redundant at best and wrong at worst.

### Run

The bottom of the file wires the two inputs to the one writer and records what happened.

```
[include-by-region: record.py#run]
```

The dataset lands in `{out}/dataset`; `metrics.json` sits beside it, deliberately *outside* the dataset directory so it can't pollute the format. That `metrics.json` is what CI actually diffs: the dataset itself carries a uuid and absolute paths and is not byte-stable, but the structural facts (how many episodes, how many frames, the obs/action dimensions, the first and last observation) are, and two `--smoke --seed 0` runs produce identical ones. The `log_rerun` helper replays your recorded observations onto the same `world/objects/tee` and `world/robot/pusher` entity paths every chapter uses, so you can scrub your own episode on the `sim_time` timeline, and it works identically whether the observations came from your local drive or a browser bundle, because by this point an episode is just arrays.

## Run it

Record three episodes locally:

```
python record.py --episodes 3
```

<!-- wall-clock table renders from wallclock.csv -->

It prints where the dataset went and drops a recording at `outputs/ch0.4-record/record.rrd`. Open it:

```
rerun outputs/ch0.4-record/record.rrd
```

Scrub the `sim_time` timeline and you'll see the pusher chase the block and worry it toward the green target, and then just keep worrying, because the crude local driver noses the block onto the target *spot* but never squares up its rotation, so no episode latches success and each one plays out to the full length. With the default settings that is 300 frames per episode: `--episodes 3` writes **3 episodes, 900 frames**. (The `--smoke --seed 0` run CI diffs is shorter and fixed: **2 episodes, 40 frames each, 80 total.**)

To convert a browser recording instead, point `record.py` at the bundle the playground handed you:

```
python record.py --from-interchange ~/Downloads/my_teleop_session
```

Same writer, same output layout, a dataset at `{out}/dataset` that loads with `LeRobotDataset` and passes the identical schema check as a scripted-expert dataset. If it doesn't load, the first thing to check is the interchange version string: the converter asserts `z2r-teleop-1` and refuses anything else rather than write a subtly wrong dataset.

## Break it

The feature schema is a contract, and contracts break quietly. Change one number, declare `observation.state` as nine floats instead of ten, and watch what happens.

The recorder still runs. It might even write a file. But the dataset it produces no longer matches what chapter 1.1's behavior cloning expects, and the failure doesn't surface here where you made it; it surfaces three chapters later as a shape-mismatch deep inside a training loop, or worse, as a policy that trains without complaint on the wrong columns and simply never works. That distance between the mistake and the symptom is exactly why the golden-parity test exists: it writes a dataset from `record.py` and diffs its schema against a `gen_demos` dataset, so a drifted feature shape fails *now*, loudly, in CI, instead of silently downstream. Exercise 2 makes you plant this bug and feel the parity check catch it.

## Exercises

Three ways to make sure you own the format, not just the file. Predict before you run.

## What's next

You have a dataset, but you haven't *looked* at it. Is it any good? How many of your episodes actually reached the target, how long are they, are there frames where the action is nonsense because your hand slipped? Right now you can't say, because a directory of parquet files is not something you can eyeball. Next chapter you load this exact dataset back and inspect it: the distribution of episode lengths, the success rate hiding in it, the handful of bad demonstrations that will quietly poison training if you leave them in. Your first dataset exists; 0.5 is where you find out what's in it.
