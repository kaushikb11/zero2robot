"""zero2robot Scale Lab runner: ch3.7 datasets-at-scale, the data engine pushed up on a GPU.

=============================================================================
 OPTIONAL SCALE LAB. You do NOT need this to complete ch3.7 or the course.
=============================================================================
The free way is the chapter default: `scale_data.py --seed 0` on a CPU laptop (or
a Colab T4) runs the whole cross-embodiment + MimicGen-style augmentation + BC
measure loop at the coverage-starved default (source_episodes 12, aug_per_demo 8,
eval_episodes 50) in minutes, and it is what the chapter, the exercises, and the
grader all use. Every REQUIRED learner path completes free on a Colab T4 or a CPU
laptop (root invariant 1).

This runner is ONLY the optional Scale Lab: it runs the SAME `scale_data.py`,
unchanged, with the chapter's OWN visible scale knobs pushed up (more source
demos, a bigger augmentation multiplier, more BC epochs, a larger eval) on a
datacenter GPU via Modal, so you can extend the data-scale curve (rollout success
vs training-set size) past what a laptop wants to sit through. It needs a Modal
account and may cost beyond Modal's free monthly credit (check Modal's current
per-second GPU pricing before you run). If you just want the lesson, the CPU
default in the chapter teaches the same thesis (data is the policy, scaled),
measured small.

  FREE ALTERNATIVE (do this first, no account, no cost):
    python curriculum/phase3_advanced/ch3.7_scale_data/scale_data.py --seed 0

HONEST SCOPE (what this is NOT): this is NOT Open X-Embodiment / DROID at true
scale. The chapter's meta.yaml is explicit that streaming a real OXE subset would
violate the free-tier + no-binaries floor (OXE is >1M trajectories / multi-TB),
so it stays REFERENCED, never fetched, here too. This runner downloads nothing:
it pushes the learner's OWN offline data engine (the perturb-and-re-solve
augmentation on the learner's PushT + ALOHA demos) harder. The augmentation
re-solves run the scripted expert inside MuJoCo, which is CPU-bound: decision 014
measured that these partly-CPU-bound curricula saturate a GPU fast, so here the
GPU mainly speeds the torch BC training and the larger eval, and the wall-clock
win is modest and honest, not a headline speedup.

Honesty on reproducibility (root invariant 2): the free-tier CPU run is
BITWISE-deterministic under --seed (the smoke metrics.json is byte-identical
twice-run). This GPU run is only STATISTICALLY reproducible (cuDNN/torch GPU
nondeterminism): same seed gives the same qualitative result and metrics within a
seeded band, NOT byte-for-byte. That is expected (see ch1.6).

-----------------------------------------------------------------------------
 What this reuses (decision 014) and why nothing here is a new dependency
-----------------------------------------------------------------------------
This is a *runner*, not a product dependency: no chapter, the grader, the
playground, or the site imports it. It reuses decision 014's Modal wall-clock
image recipe verbatim (the same `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
base + the apt GL/EGL/ffmpeg libs + the EXACT pyproject pins the wall-clock lane
installs). ch3.7 is a torch/mujoco/lerobot chapter (NOT the MJX/jax path), so the
right image is the torch wall-clock image, not ch2.3's jax[cuda12] variant. No
re-pin, no new dependency.

L40S is the honest Scale-Lab GPU tier adopted in decision 014 (Modal has no
consumer RTX 4090; the L40S is the same Ada Lovelace architecture in a datacenter
variant, the closest honest analog). Runs are labelled `l40s`, never `4090`.

-----------------------------------------------------------------------------
 Usage (Modal CLI authed via ~/.modal.toml or MODAL_TOKEN_ID/SECRET)
-----------------------------------------------------------------------------
  # the data-scale run: a bigger source set + augmentation + eval on an L40S
  modal run infra/scale_labs/ch3.7_scale_data_scale.py

  # your own scale knobs / seed / GPU tier
  modal run infra/scale_labs/ch3.7_scale_data_scale.py \
      --source-episodes 48 --aug-per-demo 24 --epochs 800 --eval-episodes 100 --seed 0

metrics.json is written back under outputs/ch3.7-scale-data-scale/ locally.

NOTE: this script is authored but UNTESTED end-to-end here (no Modal account /
GPU in the authoring environment). Treat the first real `modal run` as the
smoke test.
"""

from __future__ import annotations

import os
import pathlib

import modal

# Repo root: meaningful only LOCALLY (source mount + where results are written
# back). Modal re-imports this module inside the container at a different path;
# parents[2] may not exist there, so fall back to a harmless placeholder. Same
# guard as ch2.3_mjx_scale.py and infra/modal/modal_wallclock.py.
_HERE = pathlib.Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent

# Point Z2R_MOUNT_ROOT at a stable rsync snapshot when parallel sessions are
# editing the tree (Modal's mount hasher aborts on a mid-hash change).
MOUNT_ROOT = os.environ.get("Z2R_MOUNT_ROOT", str(REPO_ROOT))

CH_ARTIFACT = "curriculum/phase3_advanced/ch3.7_scale_data/scale_data.py"

# The EXACT pins decision 014's wall-clock image installs (torch lane), copied
# verbatim: same mujoco/torch/numpy/rerun-sdk/lerobot pins as pyproject, plus the
# CI-only pyyaml/onnx/onnxruntime the wall-clock image carries. ch3.7 needs torch
# + mujoco + lerobot (LeRobotDataset). Not a re-pin, not a new dependency.
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

# Only ignore heavy/irrelevant trees; keep the curriculum/ source the artifact
# needs (device.py, the pusht + aloha envs, gen_demos). Mirrors ch2.3's list.
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
    "outputs",         # the artifact auto-generates its own tiny demo sets in-container
    ".gstack",
    "**/*.onnx",
    "**/*.rrd",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11"
    )
    # Same apt set as decision 014's wall-clock image: git is dropped (no
    # provenance read here); EGL/GL/ffmpeg for headless MuJoCo rendering;
    # build-essential + clang + linux-libc-dev so lerobot's linux-only evdev C
    # extension builds (decision 002 linux-lane finding).
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

app = modal.App("z2r-scale-ch3.7-scale-data", image=image)


# One GPU, one run. timeout=3600s covers the (CPU-bound) augmentation re-solves
# plus the two GPU BC trainings and the larger rollout eval. gpu="L40S" is the
# honest Scale-Lab tier (decision 014); override via --gpu but keep the label
# honest (see the warning in main()).
@app.function(gpu="L40S", timeout=3600)
def run_scale(source_episodes: int, aug_per_demo: int, epochs: int,
              eval_episodes: int, hidden_dim: int, seed: int) -> str:
    """Run the UNCHANGED ch3.7 scale_data.py at a larger scale config on this GPU.
    stdout streams live to the `modal run` logs (the augmentation yield + both BC
    success rates). Returns metrics.json text."""
    import subprocess
    import sys

    print("--- z2r Scale Lab: ch3.7 datasets-at-scale on GPU ---")
    subprocess.run(["nvidia-smi"], check=False)

    out_dir = "outputs/ch3.7-scale-data-scale"
    cmd = [
        sys.executable,
        CH_ARTIFACT,
        "--source_episodes", str(source_episodes),
        "--aug_per_demo", str(aug_per_demo),
        "--epochs", str(epochs),
        "--eval_episodes", str(eval_episodes),
        "--hidden_dim", str(hidden_dim),
        "--seed", str(seed),
        "--device", "cuda",   # detect_device() would also pick cuda; explicit is honest
        "--out", out_dir,
        "--no-rerun",         # the .rrd is a free local-run thing; open it there
    ]
    print(f"[run] {' '.join(cmd)}")
    # Do NOT capture: let the artifact stream progress to the Modal logs so the
    # learner watches the augmentation + training live. check=False so we still
    # return whatever metrics landed if the run errors late.
    result = subprocess.run(cmd, cwd="/repo")
    if result.returncode != 0:
        print(f"!! scale_data.py exited {result.returncode}")

    metrics = pathlib.Path("/repo") / out_dir / "metrics.json"
    return metrics.read_text() if metrics.exists() else ""


@app.local_entrypoint()
def main(
    source_episodes: int = 48,
    aug_per_demo: int = 24,
    epochs: int = 800,
    eval_episodes: int = 100,
    hidden_dim: int = 512,
    seed: int = 0,
    gpu: str = "L40S",
):
    """Launch the ch3.7 Scale Lab on a real GPU via Modal.

    source_episodes: coverage-starved source demos per embodiment, pushed up from
                     the free-tier 12 so the data-scale curve extends.
    aug_per_demo:    MimicGen-style re-solves per source demo (the data engine).
    epochs:          BC epochs per arm (source-only and source+augmented).
    eval_episodes:   held-out rollout episodes per arm.
    hidden_dim:      BC MLP width.
    gpu:             Modal GPU tier. L40S is the honest Scale-Lab default (decision 014).
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
        "\n=== zero2robot Scale Lab: ch3.7 datasets-at-scale (OPTIONAL, costs GPU time) ===\n"
        "Free alternative (no account, no cost):\n"
        "  python curriculum/phase3_advanced/ch3.7_scale_data/scale_data.py --seed 0\n"
        "This does NOT download OXE/DROID (that stays referenced, not fetched): it\n"
        "pushes YOUR offline data engine up. The augmentation re-solve is CPU-bound\n"
        "MuJoCo, so the GPU mainly speeds the BC training + the larger eval.\n"
    )

    fn = run_scale if gpu.upper() == "L40S" else run_scale.with_options(gpu=gpu)
    print(
        f"Running ch3.7 on a {gpu} (source_episodes={source_episodes}, "
        f"aug_per_demo={aug_per_demo}, epochs={epochs}, eval_episodes={eval_episodes}, "
        f"seed={seed})...\n"
    )

    metrics_text = fn.remote(source_episodes, aug_per_demo, epochs,
                             eval_episodes, hidden_dim, seed)

    if not metrics_text.strip():
        print("\nNo metrics.json returned. Check the container logs above.")
        return

    out = REPO_ROOT / "outputs" / "ch3.7-scale-data-scale" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(metrics_text)
    print(f"\nWrote metrics -> {out}")
    print(metrics_text)
    print(
        "\nReminder: GPU numbers are STATISTICAL (same seed gives the same "
        "qualitative result within a band), not bitwise. The free CPU run is the "
        "bitwise-deterministic one. The lesson (data is the policy, scaled) is the "
        "ORDERING (augmented > source), not the exact number."
    )
