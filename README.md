# zero2robot

**Learn embodied AI by building a robot brain from scratch. No robot required.**

**[Read it online](https://zero2robot.com)** · **[Play in your browser](https://play.zero2robot.com)** · **[Newsletter: Physical AI Weekly](https://physicalaiweekly.substack.com)**

zero2robot is a 43-chapter course (across six phases) that teaches robot learning the
way you'd read a textbook, except every page is a single runnable file you study line
by line and run yourself. You start from a bare simulation loop and build up, one chapter
at a time, to a vision-language-action policy you train, evaluate, export, and drive in
your browser, then on to the from-scratch practitioner's stack and a real-arm graduation.
No framework to excavate, no black boxes.

zero2robot is free and open, and it always will be.

- **Read**: each chapter is one file (`bc.py`, `act.py`, `diffusion.py`, `ppo.py`, …),
  ≤450 lines, no clever abstractions. The code *is* the product. You read every line.
- **Run**: every artifact runs on a free Colab T4 or a CPU laptop. `--smoke` for a fast
  deterministic pass, `--seed` everywhere, `--out` for artifacts. GPU owners get optional
  Scale Labs; nobody is ever gated behind hardware.
- **Try**: chapters ship with live browser demos (MuJoCo-WASM + ONNX). Drag the block
  out of the region the demonstrations covered and watch a confident policy fail. That
  failure has a name, and the course is built around understanding it.

## Honest state

The buildable curriculum arc is **complete: 43 chapters across six phases** are built,
finalized, and re-verified end to end (Phase 0 foundations through HIL-SERL post-training,
the Phase 5 practitioner's stack, and the real-arm graduation). The interactive textbook is
live at **[zero2robot.com](https://zero2robot.com)** and the MuJoCo-WASM playground at
**[play.zero2robot.com](https://play.zero2robot.com)**; you can also run everything locally
(below).

What is honestly **not** finished, and is not claimed done: some GPU and Scale-Lab
wall-clock cells are still unmeasured (they read `TBD`, never a guess). Wall-clock times in
chapter prose are measured on real hardware (`curriculum/common/wallclock.csv`), never
estimated.

## Quickstart

```bash
git clone https://github.com/kaushikb11/zero2robot.git
cd zero2robot
uv sync                                   # or: uv venv --python 3.11 .venv && uv pip install -e ".[dev,export]"

# Chapter 0.1 — the simulation loop (instant, CPU)
.venv/bin/python curriculum/phase0_foundations/ch0.1_sim_loop/sim_loop.py --smoke

# Generate demos (deterministic, offline), then Chapter 1.1 — behavior cloning
.venv/bin/python curriculum/common/envs/pusht/gen_demos.py --episodes 100 --seed 0 --out outputs/pusht-demos --no-video
.venv/bin/python curriculum/phase1_imitation/ch1.1_bc/bc.py --smoke
```

Every artifact accepts `--smoke`, `--seed`, and `--out`. The demo datasets are
regenerated locally from seeded generators (no download needed); see
[`datasets/datasets.yaml`](datasets/datasets.yaml) and
`python scripts/fetch_datasets.py --check`.

Run the interactive textbook locally:

```bash
cd site && npm install && npm run dev      # renders chapters from curriculum/**/prose
```

Check your work on any chapter's exercises (instant, offline, no accounts):

```bash
python -m grader.check 1.1                  # runs that chapter's public exercise checks
```

## The phase map (43 chapters across six phases)

The main line is **0 → 1 → 2 → 4 (post-training) → 5 (practitioner + graduation)**: a
complete "empty file → real robot" journey on its own. **Phase 3 is an optional *Depth*
interlude, off the main line:** self-contained electives you take when the itch strikes,
never a wall before the payoff (Phase 4 requires nothing from it).

| Phase | Chapters | What you build |
|---|---|---|
| **0: Foundations** | 0.0–0.5 (6) | A quickstart on-ramp (train a tiny BC policy first, understand it later), then the simulation loop, MJCF scenes, spatial transforms, browser teleop → your first LeRobot dataset, reading episodes with rerun. |
| **1: The Imitation Spine** | 1.1–1.9 (9) | Behavior cloning, data curation, ACT (action chunking), diffusion & flow-matching policies, a real eval harness, and a from-scratch tiny VLA (data + training). |
| **2: The RL Spine** | 2.1–2.8 (8) | PPO and SAC from a blank file, 4096-robot MJX parallelism, reward design, quadruped locomotion, sim-to-real intuition labs, and a ROS-shaped runtime, without ROS. |
| **3: Depth** *(optional interlude)* | 3.1–3.9 (9) | Dreamer-style world models, a physics engine built from scratch (dynamics → constraints → contact), your engine vs MuJoCo, datasets at scale, reading the frontier, and sampling-based MPC (CEM/MPPI) that plans through the engine you built. Off the main line. Take it when the itch strikes. |
| **4: Post-Training** | 4.2, 4.3 + primer (3 built) | Take a trained policy and make it reliable: DAgger corrections on your own policy's failures, an offline-RL primer (BC vs AWAC), and HIL-SERL post-training. |
| **5: The Practitioner's Stack** | 5.1–5.8 (8) + graduation | Perception from scratch (a ViT, then contrastive vision-language), the production two-tower VLA shape and a FAST action tokenizer, LoRA and INT8 quantization by hand, and the real-arm teleop → record → train → deploy loop, plus a "go build" graduation bridge to your first real robot. |

Minimum viable paths (see `curriculum/`): a **20-hour core** (0.1–0.5, 1.1–1.4, 1.6, 1.9)
or the **completionist** run of all 43 built chapters (the 34-chapter core at
~118 learner-hours, plus Phase 5's practitioner arcs).

## Pointers

- **`curriculum/`**: the chapters. This is the product. Start at `phase0_foundations/ch0.1_sim_loop/`.
- **`site/`**: the interactive textbook (renders chapter prose + code + live demos).
- **`playground/`**: the MuJoCo-WASM browser playground and teleop UI.
- **`grader/`**: the exercise auto-checker (`python -m grader.check`).
- **`notebooks/`**: generated Colab variants of chapter code (regenerated, never hand-edited).
- **`datasets/`, `checkpoints/`**: pointers/manifests only; artifacts live on the
  Hugging Face Hub. No binaries in git.
- **[`ARCHITECTURE.md`](ARCHITECTURE.md)**: the repo map, the doctrine, and the CI gates.
- **[`CONTRIBUTING.md`](CONTRIBUTING.md)**: how the pieces fit and what a PR must pass.

## License

See [`LICENSE`](LICENSE). zero2robot is single-voice by design; chapter prose is not
accepted externally (see `CONTRIBUTING.md`).
