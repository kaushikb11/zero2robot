"""zero2robot Scale Lab runner: ch5.3 pixels, the closed-loop rollout that CPU only floors.

=============================================================================
 OPTIONAL SCALE LAB. You do NOT need this to complete ch5.3 or the course.
=============================================================================
The free way is the chapter default: `pixels.py --seed 0` on a CPU laptop (or a
Colab T4) trains the tiny ViT, the compact contrastive alignment, the
control-usefulness probe, the BC head, and the pixels-only rollout, and it is
what the chapter, the exercises, and the grader all use. Every REQUIRED learner
path completes free on a Colab T4 or a CPU laptop (root invariant 1).

The chapter's REPRODUCIBLE HEADLINE, the control-usefulness PROBE direction
(aligned features have lower held-out action-regression val MSE than random),
already reproduces free on CPU and is gated there. This Scale Lab is ONLY for the
HIGHER bar: the closed-loop pixel ROLLOUT. At free-tier scale (a 64x64 frame, a
dim-96 from-scratch ViT, single-frame BC) that rollout FLOORS at 0/20 for BOTH
encoders (the honest ceiling the chapter documents). This runner runs the SAME
`pixels.py`, unchanged, with the chapter's OWN `--dim 384` "4090" knob and more
data / epochs / eval on a datacenter GPU, so the rollout has the capacity to
actually roll out instead of sitting on the floor. It needs a Modal account and
may cost beyond Modal's free monthly credit (check Modal's current per-second GPU
pricing before you run).

  FREE ALTERNATIVE (do this first, no account, no cost):
    python curriculum/phase5_practitioner/ch5.3_pixels/pixels.py --seed 0

HONESTY (root invariant 2, and the chapter's own framing): a bigger from-scratch
ViT is still NOT a pretrained SigLIP/DINOv2 backbone. Whether the rollout lifts
off the floor at this tier is an OPEN, MEASURED question, not a promise: this
runner asserts nothing (it runs the artifact and writes its metrics.json back).
The definitive fix, a real pretrained aligned backbone, is the read-the-real-thing
(OpenVLA / openpi), reached by reference, not built here. Reproducibility: the
free CPU run is bitwise-deterministic on the SAME machine under --seed; this GPU
run is only STATISTICALLY reproducible (cuDNN/torch GPU nondeterminism) and
MuJoCo rasterization is not bitwise across arches, so absolute rollout numbers are
platform-sensitive. Read the DIRECTION, never a byte-exact %.

-----------------------------------------------------------------------------
 What this reuses (decision 014) and why nothing here is a new dependency
-----------------------------------------------------------------------------
This is a *runner*, not a product dependency: no chapter, the grader, the
playground, or the site imports it. It reuses decision 014's Modal wall-clock
image recipe verbatim (the same `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
base + the apt GL/EGL/ffmpeg libs + the EXACT pyproject pins the wall-clock lane
installs, including the CPU onnxruntime pixels.py exports + parity-checks
against). ch5.3 is a torch/mujoco chapter (NOT the MJX/jax path), so the right
image is the torch wall-clock image, not ch2.3's jax[cuda12] variant. No re-pin,
no new dependency.

L40S is the honest Scale-Lab GPU tier adopted in decision 014 (Modal has no
consumer RTX 4090; the L40S is the same Ada Lovelace architecture in a datacenter
variant, the closest honest analog to the `--dim 384` "4090" knob). Runs are
labelled `l40s`, never `4090`.

-----------------------------------------------------------------------------
 Usage (Modal CLI authed via ~/.modal.toml or MODAL_TOKEN_ID/SECRET)
-----------------------------------------------------------------------------
  # the higher bar: a dim-384 ViT + more data / epochs / eval on an L40S
  modal run infra/scale_labs/ch5.3_pixels_scale.py

  # your own scale knobs / seed / GPU tier
  modal run infra/scale_labs/ch5.3_pixels_scale.py \
      --dim 384 --episodes 200 --bc-epochs 600 --eval-episodes 50 --seed 0

metrics.json is written back under outputs/ch5.3-pixels-scale/ locally.

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

CH_ARTIFACT = "curriculum/phase5_practitioner/ch5.3_pixels/pixels.py"

# The EXACT pins decision 014's wall-clock image installs (torch lane), copied
# verbatim. ch5.3 needs torch + mujoco (rendering) + onnx/onnxruntime (the
# contract-v1 export + parity check, which belong on CPU per decision 014). Not a
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

app = modal.App("z2r-scale-ch5.3-pixels", image=image)


# One GPU, one run. timeout=3600s covers alignment + the probe + BC for both
# encoders + the larger pixel rollout at dim 384. gpu="L40S" is the honest
# Scale-Lab tier (decision 014); override via --gpu but keep the label honest.
@app.function(gpu="L40S", timeout=3600)
def run_scale(dim: int, depth: int, episodes: int, align_epochs: int,
              probe_epochs: int, bc_epochs: int, eval_episodes: int, seed: int) -> str:
    """Run the UNCHANGED ch5.3 pixels.py at a larger scale config on this GPU.
    stdout streams live to the `modal run` logs (the probe val_mse gap + the two
    rollout success rates + CIs). Returns metrics.json text."""
    import subprocess
    import sys

    print("--- z2r Scale Lab: ch5.3 pixels-only rollout on GPU ---")
    subprocess.run(["nvidia-smi"], check=False)

    out_dir = "outputs/ch5.3-pixels-scale"
    cmd = [
        sys.executable,
        CH_ARTIFACT,
        "--dim", str(dim),
        "--depth", str(depth),
        "--episodes", str(episodes),
        "--align_epochs", str(align_epochs),
        "--probe_epochs", str(probe_epochs),
        "--bc_epochs", str(bc_epochs),
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
        print(f"!! pixels.py exited {result.returncode}")

    metrics = pathlib.Path("/repo") / out_dir / "metrics.json"
    return metrics.read_text() if metrics.exists() else ""


@app.local_entrypoint()
def main(
    dim: int = 384,          # the chapter's own "4090: 384" ViT-width knob
    depth: int = 6,
    episodes: int = 200,
    align_epochs: int = 80,
    probe_epochs: int = 300,
    bc_epochs: int = 600,
    eval_episodes: int = 50,
    seed: int = 0,
    gpu: str = "L40S",
):
    """Launch the ch5.3 Scale Lab on a real GPU via Modal.

    dim:            ViT width. The chapter's knob is `T4: 96 | 4090: 384`; 384 is
                    the "4090" (here L40S) config. heads stay at the artifact
                    default 3, which divides 384.
    depth:          attention blocks (more capacity for the harder rollout bar).
    episodes:       rendered demos (more coverage than the free-tier 90).
    align/probe/bc_epochs: contrastive, probe, and BC-head training lengths.
    eval_episodes:  pixels-only rollout episodes (the Wilson-interval sample).
    gpu:            Modal GPU tier. L40S is the honest Scale-Lab default (decision 014).
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
        "\n=== zero2robot Scale Lab: ch5.3 pixels rollout (OPTIONAL, costs GPU time) ===\n"
        "Free alternative (no account, no cost):\n"
        "  python curriculum/phase5_practitioner/ch5.3_pixels/pixels.py --seed 0\n"
        "The PROBE headline (aligned < random val MSE) already reproduces free on\n"
        "CPU. This lab only chases the HIGHER bar: the pixels-only rollout, which\n"
        "floors 0/20 at free-tier. Whether a bigger ViT lifts it is an OPEN,\n"
        "MEASURED question here, not a promise. A pretrained backbone is the fix.\n"
    )

    fn = run_scale if gpu.upper() == "L40S" else run_scale.with_options(gpu=gpu)
    print(
        f"Running ch5.3 on a {gpu} (dim={dim}, depth={depth}, episodes={episodes}, "
        f"eval_episodes={eval_episodes}, seed={seed})...\n"
    )

    metrics_text = fn.remote(dim, depth, episodes, align_epochs,
                             probe_epochs, bc_epochs, eval_episodes, seed)

    if not metrics_text.strip():
        print("\nNo metrics.json returned. Check the container logs above.")
        return

    out = REPO_ROOT / "outputs" / "ch5.3-pixels-scale" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(metrics_text)
    print(f"\nWrote metrics -> {out}")
    print(metrics_text)
    print(
        "\nReminder: GPU numbers are STATISTICAL (same seed gives the same "
        "qualitative result within a band), not bitwise, and MuJoCo raster is "
        "platform-sensitive. Read success_gap and the rollout rates as a "
        "DIRECTION, never a byte-exact %."
    )
