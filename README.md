# zero2robot

Learn embodied AI by building a robot brain from scratch. No robot required.

- **Read**: every chapter is a single runnable file you study like a textbook.
- **Try**: every chapter ships with a live browser demo. Teach a robot with your mouse.
- **Free**: everything completes on a laptop or free Colab. GPU owners get optional Scale Labs.

> **Status: pre-launch (building Season 1).** The interactive site and hosted playground are
> not up yet — the links below are aspirational until Drop 1. Today you run the chapters
> locally.

## Run it locally today

```bash
uv venv --python 3.11 .venv && uv pip install -e ".[dev,export]"

# Chapter 0.1 — the simulation loop (instant, CPU)
.venv/bin/python curriculum/phase0_foundations/ch0.1_sim_loop/sim_loop.py --smoke

# Generate demos, then Chapter 1.1 — behavior cloning
.venv/bin/python curriculum/common/envs/pusht/gen_demos.py --episodes 100 --seed 0 --out outputs/pusht-demos --no-video
.venv/bin/python curriculum/phase1_imitation/ch1.1_bc/bc.py --smoke
```

Every artifact accepts `--smoke` (fast deterministic run), `--seed`, and `--out`. Run `make check`
before any PR. The playground spike lives in `playground/` (`npm install && npm run dev`).

## Coming at Drop 1

An interactive textbook site (`site/`) and a hosted MuJoCo-WASM playground where you drive the
robot in your browser and watch a policy you trained recover the task — no install, no GPU.
