# P1: Fine-Tune a Real Foundation Policy (Reading Track)

<!-- reading-track module: prose only. No artifact, no exercises, no wall-clock, no meta.yaml. -->
<!-- Upstream commits are UN-PINNED here on purpose; the author pins a verified SHA at drop. -->

## Why this is a reading module, not a chapter

Every built chapter in this course obeys the free-tier floor: it runs to completion on a
Colab T4 or a CPU laptop, and it builds exactly one file you can read top to bottom. This
module breaks both rules on purpose, and says so plainly.

Fine-tuning a published vision-language-action checkpoint (taking `lerobot/smolvla_base`
or `lerobot/pi0` and adapting it to *your* robot and *your* task) is the literal daily job
of a robot-learning engineer in 2026. It is also not a from-scratch artifact and not free
tier. The smallest useful thing here is a 450M-parameter policy whose LoRA fine-tune wants a
24 GB GPU; a full fine-tune of a π-class model wants an 80 GB A100. There is no honest way
to shrink that to a T4 laptop and still call it the real job. So we do the thing the course
has done every time the real workflow outgrows the floor (the TinyTorch concession): we
**read the real thing** instead of pretending to rebuild it. You already own the mechanisms:
attention (1.8), the flow-matching action head (1.5), the LeRobot dataset format (1.9), and
LoRA from scratch (5.6 LoRA From Scratch, a durable single). This module is where those mechanisms meet
a real 450M-param policy and a real GPU. Nothing new is *built*. What is new is the job.

## The workflow, concretely

The real loop has four steps, and only the middle two involve any machine learning.

**1. Load a checkpoint and run it zero-shot.** Both stacks ship pretrained weights you pull
by name. In LeRobot the base policy is `lerobot/smolvla_base` (the 450M model from the SmolVLA
paper, arXiv:2506.01844); in openpi it is `pi0_base` / `pi05_base`, served straight from
`gs://openpi-assets/checkpoints/`. (The frontier has kept moving: Physical Intelligence's
π0.6 added RL-style self-improvement via RECAP in November 2025, but its weights are not yet
open in openpi, so the `pi0_base` / `pi05_base` code above stays current; NVIDIA's GR00T N1.5
is now loadable through LeRobot; and Google's Gemini Robotics On-Device, from June 2025, is
fine-tunable through its own SDK with 50 to 100 demonstrations. Names to know; the workflow
below is unchanged.) Point it at your robot and run it. It will mostly fail:
your camera angle, your gripper, your lighting, your object are all off-distribution. **That
failure is the syllabus.** The gap between the zero-shot rollout and the task you want is
exactly the thing fine-tuning closes, and looking at *where* it fails (grasps but mis-places?
never grasps? ignores the instruction?) tells you how much data and which knobs you need.

**2. Record ~50 demonstrations.** The SmolVLA guide is blunt about this: ~50 episodes is the
starting point, and they found 25 was not enough: "the data quality and quantity is
definitely a key." Record several repetitions of every variation you care about (5 cube
positions × 10 episodes, in their example). This is the same `LeRobotDataset` your ch1.9
recorder has been writing all along; push it to the Hub under `${HF_USER}/mydataset`.

**3. Freeze most of the model; fine-tune the action expert.** This is the one modeling
decision, and it is the memory lever. A VLA is a big frozen vision-language backbone plus a
small action expert. You rarely retrain the backbone. In LeRobot's
`configuration_smolvla.py` the defaults already encode this:

- `freeze_vision_encoder: bool = True`
- `train_expert_only: bool = True`
- `train_state_proj: bool = True`

`train_expert_only=True` freezes the SmolVLM2 backbone and updates only the action expert and
the small projections. That is what makes a 450M policy fine-tunable on a single consumer
card. **LoRA** is the other first-class lever, and as of LeRobot v0.5.0 it is native to the
core training pipeline too: the PEFT config lives at the policy level, so you can low-rank
adapt SmolVLA without touching the trainer. openpi has offered the same choice all along as a
**LoRA** `TrainConfig`: instead of freezing whole towers you inject low-rank adapters (the
exact mechanism from the durable-core LoRA chapter) and train only those. bf16 weights and
gradient checkpointing are the other two standard
levers: trade compute for memory so activations fit. Grep `TrainConfig` in
`src/openpi/training/config.py` to see the registered LoRA-vs-full variants side by side.

The recipe those knobs sit inside has a name worth reading: **OpenVLA-OFT**
(Optimized Fine-Tuning, arXiv:2502.19645). It replaces autoregressive token-by-token decoding
with parallel decoding of a whole action chunk fed to a continuous L1-regression head, and
reports the LIBERO average climbing from 76.5% to 97.1% at ~26x higher action throughput.
Those three moves (parallel decoding, action chunking, a regression head instead of
discretized action tokens) are now effectively the default in SmolVLA and the π-class models
you fine-tune here, so OFT is the single best paper for *why* these policies fine-tune the way
they do.

**4. Train, then eval on your robot.** The real command, from the LeRobot SmolVLA docs (run
it if you have a 24 GB GPU):

```bash
cd lerobot && lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/mydataset \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true
```

20k steps is ~4 hours on a single A100; start `--batch_size` small and raise it while loading
stays fast. `lerobot-train --help` is the real menu of knobs. Then evaluate on the actual
hardware (`lerobot-rollout --policy.path=${HF_USER}/my_smolvla --robot.type=so101_follower
--robot.port=/dev/ttyACM0 --strategy.type=base --task="..."`) because a held-out *simulation* number is not the deliverable here; a robot
that does the task is. (The 1.6 discipline still applies: report a rate with its interval,
not a hero rollout.)

The openpi path has the same shape with a JAX accent: convert your data
(`examples/libero/convert_libero_data_to_lerobot.py`), compute normalization stats, then
train.

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero \
  --exp-name=my_experiment --overwrite
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

## The memory tiers: read these before you rent a GPU

openpi publishes the honest numbers, and they are the whole reason this is a reading module:

| Mode | Memory | Example GPU |
|------|--------|-------------|
| Inference | > 8 GB | RTX 4090 |
| Fine-tune (LoRA) | > 22.5 GB | RTX 4090 (24 GB) |
| Fine-tune (full) | > 70 GB | A100 (80 GB) / H100 |

Inference fits on a hobbyist card. **LoRA fine-tuning fits on a single 4090** with almost no
headroom (22.5 of 24 GB). Full fine-tuning does not: it needs a datacenter card, and past
one GPU you are into `fsdp_devices` model parallelism. This table is the map of what you can
and cannot do at your desk, and it is why "just fine-tune it" is a budget decision before it
is a code decision.

## Read the real thing (paths verified; SHAs pinned at drop)

Upstream moves fast, so commits are left un-pinned in this draft. The author pins a verified
SHA per repo at drop time. Every path below was fetch-checked to exist.

**LeRobot: `huggingface/lerobot`, `src/lerobot/policies/smolvla/`.** This is the policy you
fine-tune, and it is the grown-up version of your ch1.8 `vla.py`. Read four files against what
you built:

- `configuration_smolvla.py`: the config dataclass with the freeze flags above, plus
  `vlm_model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"`, `chunk_size = 50`,
  `n_action_steps = 50`. Every fine-tuning decision is a field here.
- `modeling_smolvla.py`: `VLAFlowMatching`, whose `embed_prefix()` lays out image + language
  + state tokens (your `TinyVLA.fuse`) and whose `sample_actions()` integrates noise→action
  (your ch1.5 sampler). This is the code your `lerobot-train` command actually optimizes.
- `smolvlm_with_expert.py`: the frozen SmolVLM2 backbone + action-expert wiring; the thing
  `train_expert_only=True` freezes and leaves trainable, respectively.
- `processor_smolvla.py`: the pretrained subword tokenizer and normalization steps that make
  *your* dataset speak the model's dialect.

**openpi: `Physical-Intelligence/openpi`.** Read the ops surface, not a re-derivation:

- `src/openpi/training/config.py`: `TrainConfig`, `LeRobotLiberoDataConfig`, and the
  LoRA-vs-full config variants. This file *is* the fine-tuning API.
- `scripts/train.py`, `scripts/compute_norm_stats.py`, `scripts/serve_policy.py`: the three
  commands above, and the whole train→serve loop.
- `examples/libero/convert_libero_data_to_lerobot.py`: the concrete "get your data into the
  format" step everyone actually spends a day on.

## Where this sits in the course

5.6 LoRA From Scratch, a durable single, builds the **mechanism**: LoRA from scratch (two low-rank matrices,
a scaling factor, and the observation that you can adapt a frozen weight by learning
`W + BA`). That chapter proves you understand *what* a LoRA adapter is on a small model you
can hold in your head. This reading module is where that same adapter is bolted onto a real
450M-parameter policy that a real robot runs, and where you learn that the hard parts of the
job are not the adapter math but the data (50 good episodes), the memory budget (22.5 of
24 GB), and the honest eval (on the robot, with an interval). You keep the mechanism; you gain
the job.

## Honest caveats

- **Not free tier, and not from-scratch.** Stated up front, restated here. If you don't have a
  24 GB GPU, the reading still stands on its own; the command is there for when you do.
- **Un-pinned upstream.** Both repos move weekly. Flag names, defaults, and the memory numbers
  can drift between the author's pinned SHA and whatever you install. Treat every command as a
  template to check against `--help`, not a guarantee. The paths and the *shape* of the
  workflow are what this module promises to keep current; exact strings are the reader's to
  verify against their checkout.
