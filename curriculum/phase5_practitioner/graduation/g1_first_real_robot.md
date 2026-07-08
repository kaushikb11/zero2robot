# G1: Your First Real Robot

<!-- graduation bridge module: prose only. No artifact, no exercises, no toy, no wall-clock, no meta.yaml. -->
<!-- Upstream (LeRobot) moves weekly. Commits/paths are UN-PINNED on purpose; verify at the docs. Lead with SO-101; SO-100 is deprecated. Never trust a SHA in this file. There isn't one. -->

*A graduation bridge. You read this one, then you go buy a robot. It is deliberately cheap to update: the hardware and the CLI are theirs, moving on `main`, and the exact strings belong to the docs, not to this file.*

## The last gap is the only one that counts

You have built every method in this course against a simulator. Behavior cloning (1.1), a curated dataset (1.2), ACT with action chunking (1.3), the evaluation harness that reports a rate with an interval instead of a hero rollout (1.6), domain randomization (2.7), DAgger corrections (4.2), a LoRA fine-tune of a real VLA (5.6). Every one of them ran because MuJoCo handed you a clean `qpos` and took a clean `ctrl` back. That is the last thing a real robot will do for you.

Crossing sim→real is the gap that turns a person who *understands* robot learning into a person who *does* it. It is also, in 2026, cheaper to cross than at any point in the field's history, and here is the part that should make you grin: **the arm on the other side runs the exact stack you already built.** A ~$230 SO-101 leader-plus-follower rig records into the same `LeRobotDataset` format your ch0.4 `record.py` wrote. It trains with a BC or ACT policy, the same two you coded from a blank file. It deploys behind a control loop that is four function calls long. You are not learning a new system. You are plugging your system into a motor.

## The loop, mapped to what you built

Hugging Face's [LeRobot](https://huggingface.co/docs/lerobot) is the reference stack for low-cost arms, and its real-robot pipeline is a scripted CLI that maps one-to-one onto the pieces you already own. The commands (verify the exact names against the docs, they move):

- **`lerobot-find-port` → `lerobot-setup-motors` → `lerobot-calibrate`.** Pure hardware bring-up, with no simulator equivalent. This is the new material, and it is the point of the whole exercise (see below).
- **`lerobot-teleoperate`: this is your ch0.4.** You drive a *leader* arm by hand; a *follower* arm mirrors it. That is your recorder's teleop step, except the joint angles come from a physical twin instead of a keyboard.
- **`lerobot-record`: this writes the format you have used all course.** Every episode lands as a `LeRobotDataset` and pushes to the Hub under `${HF_USER}/my-dataset`. Byte-for-byte the contract from ch0.4 and the reading-track data module, the same `meta/info.json`, the same tabular-plus-video layout. The docs' own advice is your ch1.2 lesson verbatim: record ~50 episodes, ~10 per object location, keep the cameras fixed; *"you should be able to do the task yourself by only looking at the camera images."* Data quality is the policy.
- **`lerobot-train`: this is your BC/ACT, or your 5.6 fine-tune.** `--policy.type=act` loads the full ACT, the action-chunking transformer you wrote in ch1.3, now *with* the CVAE latent that ch1.3 deliberately cut for clarity; it auto-adapts to your robot's motor count and cameras from the recorded dataset. Swap in `smolvla` or `pi0` and you are running the 5.6 fine-tune path on real demonstrations.
- **`lerobot-rollout`: this is your ch1.6 eval, on hardware.** It deploys the trained checkpoint and runs the task. The 1.6 discipline does not relax here; it *matters more*. Report a success rate over N trials with its interval, not the one clip you filmed for the tweet.

Under the CLI is a Python control loop so short you can hold all of it in your head:

```python
robot.connect()
while True:
    observation = robot.get_observation()   # cameras + joint state
    action = policy.select_action(observation)  # your BC / ACT / VLA
    robot.send_action(action)                # motors move
```

`connect → get_observation → select_action → send_action`. That is the entire deployment surface. Every hard-won thing in Phase 1 exists to make `select_action` good; the other three lines are the robot. (In teleop mode `select_action` is just `teleop.get_action()`, the leader arm *is* the policy.)

## What you need, and what it costs

The arm is 3D-printed parts plus off-the-shelf servos, from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (the repo predates the rename; **SO-101 is the current flagship; SO-100 is deprecated in the docs, so build SO-101**). What you need:

- **Feetech STS3215 servos**: six for a single follower arm, more for a leader (the leader uses a mix of gear ratios). These are the whole bill of materials that matters.
- **3D-printed structural parts**: STL files in the repo, printable in PLA+ on a hobby FDM printer, or bought pre-printed.
- **A laptop**: to run the CLI, record, and (for small policies) even train. A GPU helps for training but is not required to *record and deploy*.

Honest cost, from the repo's own BOM:

- **~$122** for a single follower arm (`Total $121.94` in the README), enough to *deploy* a policy you trained from someone else's dataset.
- **~$230** for the leader + follower pair (`$229.88`), the full teleop-and-record rig, and what you want if you plan to collect your own demonstrations.
- **Realistically ~$120–400 all-in.** The low end is self-printing every part and sourcing servos yourself; the high end is an assembled or "no-3D-printing-required" kit from a vendor (Seeed, Hiwonder, and others sell dual-arm kits around $220–240, plus ~$35 if you need the printed parts too). Prices and vendors shift constantly. Treat these as a 2026 snapshot, not a quote.

This is the one honest line the rest of the course never had to write: **this module is not free tier.** Every built chapter completes on a Colab T4 or a CPU laptop. This one requires you to buy or build a physical arm. There is no simulator substitute for the thing that makes it worth doing, which is exactly the next section.

## What real hardware teaches that sim cannot

If sim could teach it, we would have kept you in sim. It cannot. Three lessons live only on the metal.

**Calibration, and cross-robot transfer.** Sim handed you joint angles in a fixed, noiseless frame. A real servo does not know where zero is until you tell it, and two "identical" SO-101s disagree by degrees. `lerobot-calibrate` writes a per-arm calibration file keyed by an `id`, and you must use the *same* `id` when you teleoperate, record, and evaluate. Mismatch it and a policy that worked yesterday drives the arm into the table. This is also where transfer becomes concrete: a dataset recorded on *your* follower can be replayed on *another* SO-101 (`lerobot-replay`), and it mostly works, because what crosses the robot gap is the *calibrated joint trajectory*, the shape, not the raw motor counts. That is the ch3.7 "shape, not semantics" lesson you felt on tiny data, now with a physical second arm proving it.

**The reality gap is an OOD gap, and it is brutal.** Your sim policy hit its numbers because train and test came from the same generator. Move a lamp, change the tablecloth, nudge the camera, and a real VLA falls off a cliff. This is now the consensus finding of the 2025–26 benchmark wave, not an anecdote: **RobotArena ∞** (Jangir et al., 2025; arXiv:2510.23571, and note it is a distinct paper from RoboArena, arXiv:2506.18123, the DROID pairwise eval cited in G3) evaluates VLA policies via real-to-sim translation and reports that performance *degrades under perturbation of textures and object placements*. Robustness to distribution shift remains open. **LIBERO-Plus** (arXiv:2510.13626) is blunter: under modest perturbations of camera viewpoint and initial state, success drops *from 95% to below 30%*, and the models "tend to ignore language instructions completely," leaning on visual shortcuts a benchmark score hides. Your ch1.6 held-out-seed lesson was the miniature; this is the full-scale version. The gap between your recording conditions and your kitchen at 6pm *is* the OOD gap, and it is where practitioners spend their weeks.

**And the strategies for closing it are also things you already built.** This is the reward for the whole arc:

- **Domain randomization, your ch2.7.** Randomize lighting, object pose, camera jitter, distractors *at record time and in augmentation* so the policy stops trusting the background. You wrote the mechanism; now you aim it at your real camera.
- **Sim + real co-training.** Mix your ~50 real episodes with a larger sim or public corpus (the reading-track data module's mixture-weighting lesson: a mixture beats either source alone, but not monotonically; you tune the weights).
- **Camera and system-ID calibration.** Match the sim camera intrinsics/extrinsics to the real rig, and identify the real servo dynamics, so the two distributions overlap instead of merely resembling each other.

None of these is new to you. That is the entire thesis of G1: you did not need to wait to learn robots on real hardware. You needed the real hardware to *find out you already had.*

## Honest rails

- **Not free tier.** Stated plainly and restated: this requires buying or building an arm. Everything before it runs on a laptop; this does not.
- **Un-pinned, on purpose.** LeRobot's CLI, flag names, robot-type strings (`so101_follower`, `so101_leader`), and the Python API move weekly. Every command in this module is a *template to check against the [LeRobot docs](https://huggingface.co/docs/lerobot) and `--help`*, not a guarantee. There is no SHA in this file. Do not trust one. The [`il_robots` tutorial](https://huggingface.co/docs/lerobot/il_robots) is the current source of truth for the record→train→rollout loop; the [SO-ARM100/101 repo](https://github.com/TheRobotStudio/SO-ARM100) for the BOM and assembly.
- **SO-101, not SO-100.** SO-100 is deprecated in the docs. LeRobot also supports **Koch** (another low-cost arm) and **LeKiwi** (a mobile base with an arm on top): same dataset format, same loop, different hardware. Lead with SO-101 unless you have a reason not to.
- **The benchmark numbers are theirs, under their protocol.** The 95%→<30% figure is LIBERO-Plus's, under LIBERO-Plus's perturbations; RobotArena ∞'s degradation is under its real-to-sim eval. They are the *direction* of the truth about your robot, not a promise about it.

## What's next

G1 gets your policy onto a real arm doing a task someone else defined. Two more bridges close the send-off:

- **G2: Define Your Own Task.** The step from "reproduce the lego-in-a-bin demo" to "decide what *your* robot should do, design the dataset for it, and eval it honestly." The task definition is the research; the code you already have.
- **G3: Keep Going.** Where the field is heading and how to stay current when the stack moves weekly: reading papers as code, tracking the data pyramid, contributing back.

You built every method in this course from a blank file. The robot on your desk runs all of them. That was always the point.
