# infra/scale_labs/: learner-facing Scale Lab runners

**Optional. Never on a required path.** Every REQUIRED learner path in
zero2robot completes free on a Colab T4 or a CPU laptop (root invariant 1). The
runners here are the OPTIONAL Scale Labs: they run a chapter's *unchanged*
artifact at its full, GPU-scale configuration on a real datacenter GPU via
Modal, so you can see the headline number (4096 robots, real-VLA, fused int8)
that the free tier only points at.

Each runner:

- runs the chapter artifact **unchanged** (same file, GPU flags only),
- **reuses the same pinned Modal image and pins as the project's internal GPU
  wall-clock runner** (no re-pin, no new dependency),
- uses the honest **L40S** Scale-Lab GPU tier (Modal has no consumer RTX 4090;
  L40S is the same Ada Lovelace architecture, the closest honest analog), and
  never mislabels another GPU as "4090",
- **always points at the free alternative first** and is honest about cost.

This is tooling, not a product dependency: no chapter, the grader, the
playground, or the site imports anything here.

## Cost and accounts (read before you run)

A Scale Lab needs a **Modal account** and may **cost beyond Modal's free monthly
credit**. A single ch2.3 4096-env run is a few GPU-minutes; check Modal's current
per-second GPU pricing before you run. You never need this to finish a chapter,
an exercise, or the course. If you only want the *lesson*, run the free CPU
version below.

## Honesty on reproducibility

The free-tier CPU path is **bitwise-deterministic** under `--seed`. A GPU run is
only **statistically reproducible** (jax/XLA and cuDNN GPU nondeterminism): same
seed gives the same qualitative result and metrics within a seeded band, not
byte-for-byte. That is expected and is itself a Scale-Lab lesson (see ch1.6).

## Runners

### `ch2.3_mjx_scale.py`: 4096 robots at once (MJX PPO)

The clearest GPU win in the course: ch2.3 runs `ppo_mjx.py` with thousands of
MuJoCo-XLA envs stepping together. On CPU-jax the throughput cliff plateaus
around 256 envs; on a GPU it keeps climbing to the 4096-robot headline.

**Free alternative first (no account, no cost):**

```
python curriculum/phase2_reinforcement/ch2.3_mjx/ppo_mjx.py --seed 0
python curriculum/phase2_reinforcement/ch2.3_mjx/ppo_mjx.py --sweep 64,256,1024
```

**Scale Lab (needs Modal + GPU credit):**

```
# the headline: full 4096-robot PPO run on an L40S
modal run infra/scale_labs/ch2.3_mjx_scale.py

# the wall-clock cliff on GPU: throughput vs num_envs, past the CPU plateau
modal run infra/scale_labs/ch2.3_mjx_scale.py --sweep 256,1024,4096,8192

# a specific env count / seed / GPU tier
modal run infra/scale_labs/ch2.3_mjx_scale.py --num-envs 4096 --seed 0 --gpu L40S
```

`metrics.json` (and the rerun `.rrd`, with `--rerun`) is written back under
`outputs/ch2.3-mjx-scale/` locally.

> Authored but **untested end-to-end** (no Modal account/GPU in the authoring
> environment). Treat the first real `modal run` as the smoke test.

### `ch3.7_scale_data_scale.py`: the data engine, scaled up

Runs `scale_data.py` unchanged with the chapter's own knobs pushed up (more
source demos, a bigger augmentation multiplier, more eval) so the data-scale
curve extends past what a laptop wants to sit through. Honest scope: this is NOT
OXE/DROID at true scale (that stays referenced, never fetched: the no-binaries
floor); it pushes the learner's OWN offline data engine. The augmentation
re-solve is CPU-bound MuJoCo, so the GPU mainly speeds the BC training + eval.

```
python curriculum/phase3_advanced/ch3.7_scale_data/scale_data.py --seed 0   # free
modal run infra/scale_labs/ch3.7_scale_data_scale.py                        # scale lab
```

### `ch5.3_pixels_scale.py`: the pixels-only rollout CPU only floors

The chapter's PROBE headline (aligned < random val MSE) reproduces free on CPU;
the closed-loop pixel ROLLOUT floors 0/20 at free-tier. This runs `pixels.py`
unchanged with the `--dim 384` "4090" knob + more data/epochs/eval, so the
rollout has the capacity to actually roll out. Whether it lifts off the floor is
an open, measured question here, not a promise (a pretrained backbone is the fix).

```
python curriculum/phase5_practitioner/ch5.3_pixels/pixels.py --seed 0        # free
modal run infra/scale_labs/ch5.3_pixels_scale.py                             # scale lab
```

### `ch5.4_vla_shape_scale.py`: the two-tower VLA, scaled up

Runs `vla_shape.py` unchanged with `--model_dim 128 --layers 4` + more
data/epochs. Honest scope: the deeper upgrades the meta describes (ch5.2's
aligned encoder in the vision slot; a real pretrained VLA) would CHANGE the
artifact and stay referenced, not built. The PushT rollout is expected to keep
flooring (the frozen-random backbone is why); the routing-collapse headline
(`flow_mse_gap > 0`) reproduces free on CPU.

```
python curriculum/phase5_practitioner/ch5.4_vla_shape/vla_shape.py --seed 0   # free
modal run infra/scale_labs/ch5.4_vla_shape_scale.py                          # scale lab
```

### `ch5.7_quantize_scale.py`: a bigger policy's int8 triangle

ch5.7 is a CPU story on purpose (from-scratch numpy int8, no fused kernel, no
`--device`). A GPU does NOT flip the latency verdict (naive int8 stays
not-faster, which IS the lesson). This runs `quantize.py` unchanged on a bigger
policy so the size + per-channel-recovery direction is measured at scale; the
default L40S sits idle for the unchanged artifact (pass `--gpu none` for CPU-only).
The real fused-int8 runtime study (onnxruntime QLinear / TensorRT) is the
author's `scale_lab_ref`, not wired into the artifact.

```
python curriculum/phase5_practitioner/ch5.7_quantize/quantize.py --seed 0     # free
modal run infra/scale_labs/ch5.7_quantize_scale.py --gpu none                # scale lab (CPU-only)
```

> All four are authored but **untested end-to-end** (no Modal account/GPU in the
> authoring environment). Treat each first real `modal run` as its smoke test.

## Roadmap

The decision-019 roadmap targets are now all present: ch2.3 (MJX), ch3.7
(datasets), ch5.3 (rollout), ch5.4 (two-tower VLA), ch5.7 (quantize). Each new
chapter with a GPU-scale headline gets its own `chX.Y_*_scale.py` here, reusing
this same image/pins recipe and the same free-first / honest-cost framing.
