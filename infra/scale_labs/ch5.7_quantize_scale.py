"""zero2robot Scale Lab runner: ch5.7 quantize, a bigger policy on a datacenter tier.

=============================================================================
 OPTIONAL SCALE LAB. You do NOT need this to complete ch5.7 or the course.
=============================================================================
The free way is the chapter default: `quantize.py --seed 0` on a CPU laptop runs
the whole from-scratch int8 story (symmetric per-tensor vs per-channel weights,
static activation calibration, the full-integer path, the deployment triangle
with Wilson CIs) in about 0.15 minutes, and it is what the chapter, the
exercises, and the grader all use. Every REQUIRED learner path completes free on
a Colab T4 or a CPU laptop (root invariant 1).

  FREE ALTERNATIVE (do this first, no account, no cost):
    python curriculum/phase5_practitioner/ch5.7_quantize/quantize.py --seed 0

=============================================================================
 READ THIS FIRST: what a GPU does and does NOT buy for ch5.7
=============================================================================
ch5.7 is a CPU story ON PURPOSE. quantize.py builds int8 by hand in numpy (a
scale r/127, round, clamp, an int8 @ int8 -> int32 matmul, dequantize): there is
NO fused int8 kernel, NO --device knob, and the banner is hard-coded to cpu. So a
GPU does NOT flip the chapter's honest latency verdict: naive int8 with no fused
kernel is often SLOWER than fp32 (dequant overhead), and that is the whole point.

The TRUE "fused int8 runtime" Scale Lab the meta.yaml describes (export this
policy and run it under a fused int8 kernel: onnxruntime QLinear/QNN, or TensorRT
on an NVIDIA GPU, then MEASURE the latency the laptop CPU honestly cannot show) is
a STUDY that is NOT wired into the artifact (the chapter's scale_lab_ref is
PENDING, an author task, because building it would add a runtime step to the
from-scratch file). This runner does NOT edit the artifact to add that path.

What this runner DOES: it runs the SAME `quantize.py`, unchanged, at a LARGER
policy config (a wider MLP, more demos, a bigger eval) on a datacenter container,
so the size + action-error direction (per-tensor int8 spikes the error,
per-channel recovers it, at ~4x smaller size) is measured on a bigger policy. The
latency verdict (int8 NOT faster without a fused kernel) is EXPECTED TO HOLD, and
that expectation holding is itself the honest lesson. This runner asserts nothing.

Because the unchanged artifact is CPU-only numpy, the default L40S GPU sits IDLE
for it (it is the tier the eventual TensorRT fused-kernel study will need). If you
only want the bigger-policy quantization triangle and not to pay for an idle GPU,
pass `--gpu none` to run CPU-only.

Honesty on reproducibility (root invariant 2): unusually for a Scale Lab, this
stays BITWISE-deterministic under --seed, because the artifact runs CPU numpy +
exact int8/int32 integer arithmetic + bitwise MuJoCo resets (no GPU
nondeterminism to caveat). Only the measured latency_ms is wall-clock and not
bit-reproducible (it reflects the container CPU).

-----------------------------------------------------------------------------
 What this reuses (decision 014) and why nothing here is a new dependency
-----------------------------------------------------------------------------
This is a *runner*, not a product dependency: no chapter, the grader, the
playground, or the site imports it. It reuses decision 014's Modal wall-clock
image recipe verbatim (the same `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
base + the apt GL/EGL/ffmpeg libs + the EXACT pyproject pins the wall-clock lane
installs). ch5.7 needs torch (to train the BC policy) + mujoco + numpy. No re-pin,
no new dependency. In particular NO torch.quantization, NO onnxruntime PTQ, NO
bitsandbytes: the int8 path stays the artifact's own from-scratch numpy.

L40S is the honest Scale-Lab GPU tier adopted in decision 014 (Modal has no
consumer RTX 4090). It is the default here as the tier the fused-int8 study will
need, NOT because the current unchanged artifact uses it. Runs never mislabel a
GPU as `4090`.

-----------------------------------------------------------------------------
 Usage (Modal CLI authed via ~/.modal.toml or MODAL_TOKEN_ID/SECRET)
-----------------------------------------------------------------------------
  # a bigger policy's quantization triangle on the datacenter tier
  modal run infra/scale_labs/ch5.7_quantize_scale.py

  # CPU-only (no idle GPU), your own knobs / seed
  modal run infra/scale_labs/ch5.7_quantize_scale.py --gpu none \
      --hidden-dim 512 --demos 200 --eval-episodes 48 --seed 0

metrics.json is written back under outputs/ch5.7-quantize-scale/ locally.

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

CH_ARTIFACT = "curriculum/phase5_practitioner/ch5.7_quantize/quantize.py"

# The EXACT pins decision 014's wall-clock image installs (torch lane), copied
# verbatim. ch5.7 needs torch (BC training) + mujoco + numpy; the int8 path is
# pure numpy in the artifact. Not a re-pin, not a new dependency.
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
            "MUJOCO_GL": "egl",      # headless render backend
        }
    )
    .add_local_dir(MOUNT_ROOT, "/repo", ignore=MOUNT_IGNORE, copy=False)
)

app = modal.App("z2r-scale-ch5.7-quantize", image=image)


# One run. timeout=3600s is ample (the quantize triangle is fast even at a wider
# MLP). gpu="L40S" is the DEFAULT tier (the fused-int8 study's target), but the
# unchanged numpy artifact runs on the container CPU; pass --gpu none for CPU-only.
@app.function(gpu="L40S", timeout=3600)
def run_scale(hidden_dim: int, demos: int, calib_episodes: int, epochs: int,
              eval_episodes: int, seed: int) -> str:
    """Run the UNCHANGED ch5.7 quantize.py at a larger policy config. stdout
    streams live to the `modal run` logs (the round-trip errors, the deployment
    triangle, and the honest latency verdict). Returns metrics.json text."""
    import subprocess
    import sys

    print("--- z2r Scale Lab: ch5.7 quantize (from-scratch numpy int8) ---")
    subprocess.run(["nvidia-smi"], check=False)   # informational; the artifact uses CPU numpy

    out_dir = "outputs/ch5.7-quantize-scale"
    cmd = [
        sys.executable,
        CH_ARTIFACT,
        "--hidden_dim", str(hidden_dim),
        "--demos", str(demos),
        "--calib_episodes", str(calib_episodes),
        "--epochs", str(epochs),
        "--eval_episodes", str(eval_episodes),
        "--seed", str(seed),
        # NOTE: quantize.py has NO --device flag (int8 is a CPU story by design).
        "--out", out_dir,
        "--no-rerun",         # the .rrd is a free local-run thing; open it there
    ]
    print(f"[run] {' '.join(cmd)}")
    # Do NOT capture: stream progress to the Modal logs. check=False so we still
    # return whatever metrics landed if the run errors late.
    result = subprocess.run(cmd, cwd="/repo")
    if result.returncode != 0:
        print(f"!! quantize.py exited {result.returncode}")

    metrics = pathlib.Path("/repo") / out_dir / "metrics.json"
    return metrics.read_text() if metrics.exists() else ""


@app.local_entrypoint()
def main(
    hidden_dim: int = 512,   # wider than the free-tier 128 so the triangle is measured on a bigger policy
    demos: int = 200,
    calib_episodes: int = 16,
    epochs: int = 500,
    eval_episodes: int = 48,
    seed: int = 0,
    gpu: str = "L40S",
):
    """Launch the ch5.7 Scale Lab via Modal.

    hidden_dim:     BC MLP width (bigger than the free-tier 128).
    demos:          scripted-expert episodes to train on.
    calib_episodes: held-out calibration + eval episodes.
    epochs:         BC epochs.
    eval_episodes:  held-out rollouts per config (the Wilson-interval sample).
    gpu:            Modal tier. L40S is the default (the fused-int8 study's target);
                    pass `none`/`cpu` to run CPU-only and not pay for an idle GPU.
    """
    cpu_only = gpu.lower() in ("none", "cpu", "")
    if cpu_only:
        # The unchanged artifact is CPU-only numpy int8; a CPU function is the
        # honest, cheaper choice when you only want the bigger-policy triangle.
        fn = run_scale.with_options(gpu=None)
        tier_label = "CPU-only (no GPU)"
    elif gpu.upper() == "L40S":
        fn = run_scale
        tier_label = "L40S (idle for the unchanged numpy artifact; the fused-int8 study's tier)"
    else:
        # Honesty guardrail (decision 014): another GPU is fine, but never a 4090.
        print(
            f"[note] gpu={gpu} is not the recorded Scale-Lab tier (l40s). Fine to "
            f"run, but it is never a '4090' (decision 014), and the unchanged "
            f"numpy artifact will not use it anyway."
        )
        fn = run_scale.with_options(gpu=gpu)
        tier_label = gpu

    print(
        "\n=== zero2robot Scale Lab: ch5.7 quantize (OPTIONAL, costs compute time) ===\n"
        "Free alternative (no account, no cost):\n"
        "  python curriculum/phase5_practitioner/ch5.7_quantize/quantize.py --seed 0\n"
        "This runs the UNCHANGED from-scratch numpy artifact on a bigger policy. A\n"
        "GPU does NOT flip the latency verdict (int8 has no fused kernel here, by\n"
        "design): naive int8 stays not-faster, which IS the lesson. The real\n"
        "fused-int8 runtime study (onnxruntime QLinear / TensorRT) is the author's\n"
        "scale_lab_ref, not wired into the artifact.\n"
    )
    print(
        f"Running ch5.7 on {tier_label} (hidden_dim={hidden_dim}, demos={demos}, "
        f"eval_episodes={eval_episodes}, seed={seed})...\n"
    )

    metrics_text = fn.remote(hidden_dim, demos, calib_episodes, epochs, eval_episodes, seed)

    if not metrics_text.strip():
        print("\nNo metrics.json returned. Check the container logs above.")
        return

    out = REPO_ROOT / "outputs" / "ch5.7-quantize-scale" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(metrics_text)
    print(f"\nWrote metrics -> {out}")
    print(metrics_text)
    print(
        "\nReminder: this Scale Lab stays BITWISE-deterministic under --seed "
        "(CPU numpy + exact integer arithmetic); only latency_ms is wall-clock. "
        "The size + per-channel-recovery direction is the headline; int8 is "
        "expected to stay NOT faster (no fused kernel), which is the honest point."
    )
