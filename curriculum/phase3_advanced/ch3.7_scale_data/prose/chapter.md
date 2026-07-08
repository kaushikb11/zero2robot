# 3.7: Datasets at Scale

## The datasets you will never download

Open X-Embodiment is more than a million trajectories from twenty-two different
robots. DROID is seventy-six thousand demonstrations across five hundred scenes.
These are the datasets real robot policies are trained on, and you cannot fit
either one on a Colab T4. OXE alone is multiple terabytes. So this chapter does
not ask you to. It makes a sharper argument: every hard problem those datasets
pose is already sitting in the two tiny datasets *you* made, and you can feel all
of them (and fix one of them) offline, on a laptop, in a few minutes.

There are two problems, and they are the whole chapter:

1. **Cross-embodiment wrangling.** OXE is not one robot. It is dozens, with
   different action spaces, different sensors, different formats, all poured into
   one training run. You have exactly this in miniature: your PushT pusher acts
   in 2 numbers, your ALOHA bimanual rig acts in 6. Mixing them surfaces the real
   questions: how do you normalize across robots, and what can a shared policy
   actually carry from one to another?
2. **Data as the bottleneck.** Chapter 1.2 told you the data *is* the policy. At
   scale, that means the highest-leverage thing you can build is not a better
   network but a *data engine*: a way to turn a few demos into many valid ones.
   You will build a from-scratch MimicGen and measure that it works.

## Cross-embodiment, made concrete

```
[include-by-region: scale_data.py#wrangle]
```

Load your two datasets the way any training stack does (straight off disk, into
plain numpy) and the heterogeneity is immediate. PushT actions live in 2 dims
(pusher velocity); ALOHA actions live in 6 (two arms, two grippers). Both
*observations* are ten numbers, and that symmetry is a trap: the layouts do not
line up. Index 2 is the block's *x* in PushT and the right gripper's *closedness*
in ALOHA: a length and a unitless open/close command that happen to share a
slot and share nothing else. Ten-number states that mean different things,
column by column, are the norm the moment you mix robots.

Two consequences fall out, and both scale straight up to OXE:

- **Normalization is per embodiment.** A pusher's 1 m/s and a gripper's
  open/close command do not share a scale. Compute one global normalizer over
  the mixed pile and you crush one robot's signal into the other's. Real
  cross-embodiment training normalizes per dataset for exactly this reason, and
  the per-embodiment min/max this region prints is the same statistic, by hand.
- **A shared policy transfers structure, not semantics.** To put both
  embodiments in one action tensor you pad the narrow one up to the widest action
  dim and carry an `action_mask` marking the real dims: the same zero-pad +
  mask you built in ch1.7. The model always emits 6 numbers; the mask tells the
  loss which ones a given example actually constrains, so a PushT row never
  trains the four ALOHA dims it left at zero. What crosses the embodiment gap is
  the *shape* of the problem (an MLP, an action head), never the meaning of raw
  dimension 2, which is the honest answer to "what does a cross-embodiment model
  share?" and the reason OXE-scale models still need per-robot heads.

No policy is trained on the mixed pile here; that is ch1.8's job. The point is
that you have now hit, on your own data, the format-wrangling and normalization
reality that a single paragraph in the OXE paper glosses over.

## A data engine you can actually run: MimicGen from scratch

```
[include-by-region: scale_data.py#augment]
```

MimicGen's idea is simple and powerful: a manipulation task is *object-centric*,
so a demonstration recorded for one object pose can be transformed into a valid
demonstration for a nearby pose. Generate enough poses and a handful of human
demos becomes thousands.

Our single-env analog is the most honest version of that idea available to us,
because we have something MimicGen does not: a scripted expert that solves *any*
start. So for each of your source demos, we read its initial object and pusher
pose, perturb them, drop the environment into that perturbed start through the
public MuJoCo model, and **re-solve it with the same expert that made the
originals**. The trajectory that comes out is real (the true solver acting in
the true physics), not a fabricated action sequence stitched onto a new start.

The critical line is the filter: we keep an augmented demo **only if the expert
still succeeds**. A perturbed start the solver cannot finish is not a demo, it is
noise, and it never joins the pile. Measured over the default config, that filter
passes 96–99% of attempts: the perturbations are wide enough to add coverage,
mild enough to stay solvable. That yield is itself a reading on how far you can
push the object before the task stops being the task.

## Does more data help? Measure it.

```
[include-by-region: scale_data.py#measure]
```

This is the payoff, and it is deliberately the ch1.1 recipe, unchanged: the same
small BC MLP, normalization baked in as buffers, the same rollout eval. Two arms
are trained: once on your 12 source demos, once on those 12 plus everything the
augmentation produced. Both arms start from **identical weights** (the loop
re-seeds torch before each) and see batches in the **same order**, so the only
thing that differs between them is the *data*. That is what makes this a
measurement and not an anecdote, and it is why turning augmentation off
(`--aug_per_demo 0`) reproduces the source-only arm to the digit.

The measured result, seed sweep 0–2 at the default config:

| seed | source-only | source + augmented | delta |
|------|-------------|--------------------|-------|
| 0    | 0.02        | 0.30               | +0.28 |
| 1    | 0.08        | 0.20               | +0.12 |
| 2    | 0.14        | 0.26               | +0.12 |

Augmentation helps on **every seed**. Read the table honestly, because both
halves matter:

- The **ordering is the rock**: source << augmented, always, +0.12 to +0.28.
  That is the ch1.2 thesis scaled: the policy never changed, only the data did,
  and more valid data bought more success.
- The **absolute numbers are modest** (2–30%), on purpose. Twelve demos tile
  almost none of the PushT spawn annulus, so source-only BC is coverage-starved;
  we chose that regime so the *data* effect is visible instead of being drowned
  by an already-saturated policy. This is the free-tier reality, not a
  state-of-the-art result, and the chapter does not pretend otherwise. With more
  source demos you would expect the gap to shrink as source-only stops starving,
  which is itself the lesson about *where* augmentation earns its keep: exactly
  where data is scarce.

One seed would have told you almost nothing here (seed 0's +0.28 next to seed
1's +0.12), which is exactly the ch2.1 warning: read RL-flavored metrics across
seeds, never off a single draw. The rerun recording logs the two success rates
against training-set size: the data-scale curve, for your own eyes.

## Read the real thing

Our data engine is one region of `scale_data.py`: `augment`. For each source
demo it reads the first state, perturbs the block and pusher pose, drops the env
into that start through the public MuJoCo model (`set_pusht_start`), and
**re-solves with the same scripted expert** (`solve_from`). The honesty gate is a
single line, `if ok:`. The trajectory joins the pile only when the expert still
finishes. Because we own a solver that solves *any* start, every kept demo is a
real solution in real physics. That ownership is also the whole simplification:
one task, one object, one embodiment, a solver on tap.

`meta.yaml` pins the real thing to `NVlabs/mimicgen@ea09885` (tag `v1.0.0`). Read
it against what you just wrote.

**The object-centric transform.** MimicGen has no solver on tap, so instead of
re-solving it *replays*, geometrically. `DataGenerator.generate()` in
`mimicgen/datagen/data_generator.py` segments each source demo into per-object
subtasks, and for each one calls `transform_source_data_segment_using_object_pose`
in `mimicgen/utils/pose_utils.py` (line 261): it takes the end-effector poses that
the source demo recorded relative to the object and re-expresses them relative to
the object's *new* pose in the current scene. The transformed waypoints
(`WaypointTrajectory`) are then executed open-loop through the robot's controller.
Same object-centric bet our perturb makes (a demo for one object pose is valid
for a nearby one), but done in SE(3) pose space and run through the real
controller rather than handed back to an expert.

**The success filter, at scale.** Ours is `if ok`. Theirs is the top of
`mimicgen/scripts/generate_dataset.py`: after each `generate()`,
`success = bool(generated_traj["success"])`, and only on success is the episode
written and merged into the output dataset. The loop runs under `guarantee_success`
(keep attempting until you have collected N *successful* trajectories), so the
reported `success_rate = num_success / num_attempts` is the production twin of our
`aug_yield`. Same gate, industrial scale: thousands of attempts, per-episode HDF5s
merged at the end.

**What they add, and why.** Real tasks are multi-step (pick *then* place *then*
insert), so MimicGen segments per subtask and picks a source segment per object
frame (`select_source_demo`, `mimicgen/datagen/selection_strategy.py`); our single
push has one segment and needs none of it. And their cross-embodiment story is
*robot transfer*, not our normalization: `transform_first_robot_pose` plus the env
interface's `target_pose_to_action` (`mimicgen/env_interfaces/base.py`) let the
same object-relative waypoints drive a *different arm* through its own IK: the
transfer is geometric, in pose space. Be honest about the seam: the per-embodiment
min/max, zero-pad, and `action_mask` this chapter builds are the OXE/OpenVLA answer
to heterogeneous *action vectors*, a different mechanism than MimicGen's pose-space
transfer. The chapter reproduces both realities on your own tiny data; the paper
solves one of them at a million trajectories.

**Read next:** open `mimicgen/utils/pose_utils.py` and read
`transform_source_data_segment_using_object_pose`, then follow it up into
`DataGenerator.generate()` in `mimicgen/datagen/data_generator.py`, then out to the
`success` check in `mimicgen/scripts/generate_dataset.py`. That path (transform,
execute, keep-if-success) is exactly the three moves your `augment` region
compresses into a perturb, a re-solve, and an `if ok`.

## Where this goes

You have now, offline and from scratch, hit the cross-embodiment wrangling that
OXE is built on and run the MimicGen-style data engine that turns small demo sets
into large ones, and you *measured* that it works instead of taking it on faith.
The Scale Lab (meta.yaml) is this same picture at a million trajectories: stream
a real OXE subset on a bigger tier and the normalization, the padding, and the
"what transfers" question are identical, just larger. The read-the-real-thing
segment points you at the primary sources (Open X-Embodiment, DROID, MimicGen),
which you can now read as descriptions of code you have already written.
