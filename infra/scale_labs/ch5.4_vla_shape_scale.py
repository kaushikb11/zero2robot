"""zero2robot Scale Lab runner: ch5.4 the production VLA shape, scaled up on a GPU.

=============================================================================
 OPTIONAL SCALE LAB. You do NOT need this to complete ch5.4 or the course.
=============================================================================
The free way is the chapter default: `vla_shape.py --seed 0` on a CPU laptop (or
a Colab T4) trains the two-tower prefix/suffix VLA at the deliberately tiny free
config (model_dim 64, 2 blocks, H=8) in about 1.3 minutes, and it is what the
chapter, the exercises, and the grader all use. Every REQUIRED learner path
completes free on a Colab T4 or a CPU laptop (root invariant 1).

The chapter's GATED HEADLINE, the routing-collapse DIRECTION (cutting the
suffix->prefix cross-attention raises the held-out flow MSE, flow_mse_gap > 0),
already reproduces free on CPU and is gated there. This runner runs the SAME
`vla_shape.py`, unchanged, with the chapter's OWN visible upscale (`--model_dim
128 --layers 4`, more episodes / epochs / eval) on a datacenter GPU, so you can
watch the same two-tower mechanism at a larger capacity. It needs a Modal account
and may cost beyond Modal's free monthly credit (check Modal's current
per-second GPU pricing before you run).

  FREE ALTERNATIVE (do this first, no account, no cost):
    python curriculum/phase5_practitioner/ch5.4_vla_shape/vla_shape.py --seed 0

HONEST SCOPE (what this is NOT): the two DEEPER upgrades ch5.4's meta.yaml
describes (step 1: ch5.2's ALIGNED encoder in the vision slot; step 2: a real
pretrained VLA, pi0 / SmolVLA, a SigLIP tower + a pretrained-LM tokenizer + this
exact prefix/suffix flow-expert) would CHANGE the artifact and stay REFERENCED,
not built, here too. This runner does NOT swap the encoder or load a pretrained
VLA: it only pushes the UNCHANGED artifact's own knobs (model_dim, layers,
episodes, epochs) up on a GPU. So the PushT rollout is expected to keep flooring
(the frozen RANDOM backbone is why, exactly as the chapter warns): the pixels are
NOT load-bearing until an aligned/pretrained backbone goes in the vision slot,
which is the author's scale_lab_ref to wire. This runner asserts nothing.

Honesty on reproducibility (root invariant 2): the free-tier CPU run is
statistically reproducible under --seed (flow_mse_full and final_train_loss
match; flow_mse_cut is stable to ~1e-5 from CPU BLAS reduction order), and the
`--smoke` metrics.json is byte-identical twice-run. This GPU run is only
STATISTICALLY reproducible (cuDNN/torch GPU nondeterminism): same seed gives the
same qualitative result within a band, NOT byte-for-byte. Read the DIRECTION
(flow_mse_gap > 0), never an exact MSE.

-----------------------------------------------------------------------------
 What this reuses (decision 014) and why nothing here is a new dependency
-----------------------------------------------------------------------------
This is a *runner*, not a product dependency: no chapter, the grader, the
playground, or the site imports it. It reuses decision 014's Modal wall-clock
image recipe verbatim (the same `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
base + the apt GL/EGL/ffmpeg libs + the EXACT pyproject pins the wall-clock lane
installs). ch5.4 is a torch/mujoco chapter (NOT the MJX/jax path), so the right
image is the torch wall-clock image, not ch2.3's jax[cuda12] variant. No re-pin,
no new dependency.

L40S is the honest Scale-Lab GPU tier adopted in decision 014 (Modal has no
consumer RTX 4090; the L40S is the same Ada Lovelace architecture in a datacenter
variant, the closest honest analog to the "consumer GPU" ch5.4's scale story
imagines). Runs are labelled `l40s`, never `4090`.

-----------------------------------------------------------------------------
 Usage (Modal CLI authed via ~/.modal.toml or MODAL_TOKEN_ID/SECRET)
-----------------------------------------------------------------------------
  # the scaled two-tower: model_dim 128, 4 blocks, more data on an L40S
  modal run infra/scale_labs/ch5.4_vla_shape_scale.py

  # your own scale knobs / seed / GPU tier
  modal run infra/scale_labs/ch5.4_vla_shape_scale.py \
      --model-dim 128 --layers 4 --episodes 200 --epochs 300 --eval-episodes 50 --seed 0

metrics.json is written back under outputs/ch5.4-vla-shape-scale/ locally.

NOTE: this script is authored but UNTESTED end-to-end here (no Modal account /
GPU in the authoring environment). Treat the first real `modal run` as the
smoke test.
"""

from __future__ import annotations

import os
import pathlib

import modal

# Repo root: meaningful only LOCALLY (source mount + where results are written
# back). Same guard as ch2.3_mjx_scale.py and infra/modal/modal_wallclock.py.
_HERE = pathlib.Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent

# Point Z2R_MOUNT_ROOT at a stable rsync snapshot when parallel sessions are
# editing the tree (Modal's mount hasher aborts on a mid-hash change).
MOUNT_ROOT = os.environ.get("Z2R_MOUNT_ROOT", str(REPO_ROOT))

CH_ARTIFACT = "curriculum/phase5_practitioner/ch5.4_vla_shape/vla_shape.py"

# The EXACT pins decision 014's wall-clock image installs (torch lane), copied
# verbatim. ch5.4 needs torch + mujoco (rendering the 64x64 frames). Not a
# re-pin, not a new dependency.
WALLCLOCK_PINS = [
    "mujoco==3.10.0",
    "torch==2.10.0",
    "numpy==2.4.6",
    "rerun-sdk==0.26.2",
    "lerobot==0.4.4",
    "pyyaml==6.0.3",
    "onnx~=1.17",
    "onnxruntime~=1.20",   # CPU onnxruntime, exactly as the wall-clock image (decision 014)
]

MOUNT_IGNORE = [
    ".venv",
    ".git",            # this runner reads no git provenance (unlike the wall-clock lane)
    "**/__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "**/node_modules",
    "site/dist",
    "site/.astro",
    "site/.build",
    "playground/dist",
    "notebooks",
    "outputs",
    ".gstack",
    "**/*.onnx",
    "**/*.rrd",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11"
    )
    # Same apt set as decision 014's wall-clock image (git dropped: no provenance
    # read here). EGL/GL/ffmpeg for headless MuJoCo rendering; the build toolchain
    # for lerobot's linux-only evdev C extension (decision 002 linux-lane finding).
    .apt_install(
        "libegl1",
        "libgl1",
        "libglib2.0-0",
        "libosmesa6",
        "ffmpeg",
        "build-essential",
        "clang",
        "linux-libc-dev",
    )
    .pip_install(*WALLCLOCK_PINS)
    .env(
        {
            "PYTHONPATH": "/repo",   # guarantees `import curriculum...` from any cwd
            "MUJOCO_GL": "egl",      # headless render backend for the 64x64 frames
        }
    )
    .add_local_dir(MOUNT_ROOT, "/repo", ignore=MOUNT_IGNORE, copy=False)
)

app = modal.App("z2r-scale-ch5.4-vla-shape", image=image)


# One GPU, one run. timeout=3600s covers the scaled two-tower training + the
# rollout eval. gpu="L40S" is the honest Scale-Lab tier (decision 014); override
# via --gpu but keep the label honest (see the warning in main()).
@app.function(gpu="L40S", timeout=3600)
def run_scale(model_dim: int, layers: int, episodes: int, epochs: int,
              eval_episodes: int, seed: int) -> str:
    """Run the UNCHANGED ch5.4 vla_shape.py at a larger scale config on this GPU.
    stdout streams live to the `modal run` logs (the flow_mse full/cut gap + the
    PushT rollout success + CI). Returns metrics.json text."""
    import subprocess
    import sys

    print("--- z2r Scale Lab: ch5.4 two-tower VLA on GPU ---")
    subprocess.run(["nvidia-smi"], check=False)

    out_dir = "outputs/ch5.4-vla-shape-scale"
    cmd = [
        sys.executable,
        CH_ARTIFACT,
        "--model_dim", str(model_dim),
        "--layers", str(layers),
        "--episodes", str(episodes),
        "--epochs", str(epochs),
        "--eval_episodes", str(eval_episodes),
        "--seed", str(seed),
        "--device", "cuda",   # detect_device() would also pick cuda; explicit is honest
        "--out", out_dir,
        "--no-rerun",         # the .rrd is a free local-run thing; open it there
    ]
    print(f"[run] {' '.join(cmd)}")
    # Do NOT capture: stream progress to the Modal logs. check=False so we still
    # return whatever metrics landed if the run errors late.
    result = subprocess.run(cmd, cwd="/repo")
    if result.returncode != 0:
        print(f"!! vla_shape.py exited {result.returncode}")

    metrics = pathlib.Path("/repo") / out_dir / "metrics.json"
    return metrics.read_text() if metrics.exists() else ""


@app.local_entrypoint()
def main(
    model_dim: int = 128,    # the chapter's own "T4: 128" tower-width knob
    layers: int = 4,         # the chapter's own "T4: 4" block-count knob
    episodes: int = 200,
    epochs: int = 300,
    eval_episodes: int = 50,
    seed: int = 0,
    gpu: str = "L40S",
):
    """Launch the ch5.4 Scale Lab on a real GPU via Modal.

    model_dim: tower width. The chapter's knob is `T4: 128`; heads stay at the
               artifact default 4, which divides 128.
    layers:    shared attention blocks (the chapter's `T4: 4`).
    episodes:  scripted-expert PushT demos (more than the free-tier 60).
    epochs:    training epochs.
    eval_episodes: held-out PushT rollout episodes.
    gpu:       Modal GPU tier. L40S is the honest Scale-Lab default (decision 014).
    """
    if gpu.upper() != "L40S":
        # Honesty guardrail (decision 014): another GPU is fine to run, but it is
        # NOT a 4090 and its numbers are not the recorded l40s Scale-Lab band.
        print(
            f"[note] gpu={gpu} is not the recorded Scale-Lab tier (l40s). Fine to "
            f"run, but its timings are yours alone, not the published l40s band, "
            f"and never a '4090' (decision 014)."
        )

    print(
        "\n=== zero2robot Scale Lab: ch5.4 two-tower VLA (OPTIONAL, costs GPU time) ===\n"
        "Free alternative (no account, no cost):\n"
        "  python curriculum/phase5_practitioner/ch5.4_vla_shape/vla_shape.py --seed 0\n"
        "The routing-collapse headline (flow_mse_gap > 0) already reproduces free\n"
        "on CPU. This lab only scales the UNCHANGED artifact's own knobs up. It\n"
        "does NOT swap in an aligned encoder or a pretrained VLA (those change the\n"
        "artifact and stay referenced), so the PushT rollout is expected to keep\n"
        "flooring: the frozen-random backbone is why.\n"
    )

    fn = run_scale if gpu.upper() == "L40S" else run_scale.with_options(gpu=gpu)
    print(
        f"Running ch5.4 on a {gpu} (model_dim={model_dim}, layers={layers}, "
        f"episodes={episodes}, epochs={epochs}, eval_episodes={eval_episodes}, "
        f"seed={seed})...\n"
    )

    metrics_text = fn.remote(model_dim, layers, episodes, epochs, eval_episodes, seed)

    if not metrics_text.strip():
        print("\nNo metrics.json returned. Check the container logs above.")
        return

    out = REPO_ROOT / "outputs" / "ch5.4-vla-shape-scale" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(metrics_text)
    print(f"\nWrote metrics -> {out}")
    print(metrics_text)
    print(
        "\nReminder: GPU numbers are STATISTICAL (same seed gives the same "
        "qualitative result within a band), not bitwise. Read the routing lesson "
        "as the DIRECTION flow_mse_gap > 0, never an exact MSE. The PushT rollout "
        "floors until an aligned/pretrained backbone goes in the vision slot."
    )
