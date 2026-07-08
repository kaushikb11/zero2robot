"""zero2robot Scale Lab runner: ch2.3 MJX PPO at 4096 robots on a real GPU.

=============================================================================
 OPTIONAL SCALE LAB. You do NOT need this to complete ch2.3 or the course.
=============================================================================
The free way is the chapter default: `ppo_mjx.py` on CPU-jax (num_envs 64)
runs in about half a minute on a laptop and on a Colab T4, and it is what the
chapter, the exercises, and the grader all use. Every REQUIRED learner path
completes free on a Colab T4 or a CPU laptop (root invariant 1).

This runner is ONLY the optional Scale Lab: it runs the SAME `ppo_mjx.py`,
unchanged, with `--platform gpu --num_envs 4096` on a real datacenter GPU via
Modal, so you can watch the wall-clock cliff keep climbing past the CPU plateau
(~256 envs) to the 4096-robot headline. It needs a Modal account and may cost
beyond Modal's free monthly credit (a 4096-env run is a few GPU-minutes; check
Modal's current per-second GPU pricing before you run). If you just want to
learn the lesson, the CPU-jax sweep in the chapter teaches the same tradeoff
(throughput vs gradient quality), measured small.

  FREE ALTERNATIVE (do this first, no account, no cost):
    python curriculum/phase2_reinforcement/ch2.3_mjx/ppo_mjx.py --seed 0
    python curriculum/phase2_reinforcement/ch2.3_mjx/ppo_mjx.py --sweep 64,256,1024

Honesty on reproducibility (root invariant 2): the free-tier CPU-jax run is
BITWISE-deterministic under --seed. This GPU run is only STATISTICALLY
reproducible (jax/XLA GPU nondeterminism): same seed -> same qualitative result
and metrics within a seeded band, NOT byte-for-byte. That is expected.

-----------------------------------------------------------------------------
 What this reuses (decision 014) and why nothing here is a new dependency
-----------------------------------------------------------------------------
This is a *runner*, not a product dependency: no chapter, the grader, the
playground, or the site imports it. It reuses decision 014's Modal image recipe
(the same `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` base + apt libs proven
by `infra/modal/modal_mjx_probe.py`) and the SAME pins as pyproject's `[mjx]`
extra. The only variant is installing jax's `[cuda12]` plugin instead of the CPU
jaxlib: decision 014 flagged exactly this ("a GPU MJX run needs the jax[cuda12]
variant ... the ci-gpu MJX image must add [cuda12]"). Same jax pin (~=0.10), same
mujoco/mujoco-mjx/flax/optax pins. No re-pin, no new dependency.

L40S is the honest Scale-Lab GPU tier adopted in decision 014 (Modal has no
consumer RTX 4090; the L40S is the same Ada Lovelace architecture in a
datacenter variant, the closest honest analog to the "4090" the curriculum
imagined). Runs are labelled `l40s`, never mislabelled as `4090`.

-----------------------------------------------------------------------------
 Usage (Modal CLI authed via ~/.modal.toml or MODAL_TOKEN_ID/SECRET)
-----------------------------------------------------------------------------
  # the headline: full 4096-robot PPO run on an L40S
  modal run infra/scale_labs/ch2.3_mjx_scale.py

  # the wall-clock cliff on GPU: throughput vs num_envs, past the CPU plateau
  modal run infra/scale_labs/ch2.3_mjx_scale.py --sweep 256,1024,4096,8192

  # a specific env count / seed / GPU tier
  modal run infra/scale_labs/ch2.3_mjx_scale.py --num-envs 4096 --seed 0 --gpu L40S

metrics.json (and the rerun .rrd, if --rerun) are written back under
outputs/ch2.3-mjx-scale/ locally so you can open the recording:
    rerun outputs/ch2.3-mjx-scale/ppo_mjx.rrd

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
# parents[2] may not exist there, so fall back to a harmless placeholder (the
# mount is resolved from the local run, not remotely). Same guard as
# infra/modal/modal_wallclock.py.
_HERE = pathlib.Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent

# Point Z2R_MOUNT_ROOT at a stable rsync snapshot when parallel sessions are
# editing the tree (Modal's mount hasher aborts on a mid-hash change). Same
# escape hatch as the wallclock runner.
MOUNT_ROOT = os.environ.get("Z2R_MOUNT_ROOT", str(REPO_ROOT))

CH_ARTIFACT = "curriculum/phase2_reinforcement/ch2.3_mjx/ppo_mjx.py"

# SAME pins as pyproject's `[mjx]` extra (jax~=0.10, mujoco-mjx==3.10.0,
# flax~=0.12, optax~=0.2) plus the base mujoco==3.10.0 pin. The ONLY change is
# installing jax's `[cuda12]` plugin instead of the CPU jaxlib; decision 014
# flagged this as the required GPU-MJX image variant. rerun-sdk is pinned at the
# decision-014 version so --rerun can record the run. Not a re-pin, not a new dep.
MJX_GPU_PINS = [
    "jax[cuda12]~=0.10",   # same jax pin as [mjx]; [cuda12] is the GPU plugin (decision 014)
    "mujoco==3.10.0",
    "mujoco-mjx==3.10.0",
    "flax~=0.12",
    "optax~=0.2",
    "rerun-sdk==0.26.2",   # optional --rerun recording; pinned per decision 014
]

# Only ignore heavy/irrelevant trees; keep the curriculum/ source the artifact
# needs (device.py, the cartpole.xml env). Mirrors modal_wallclock's ignore list.
MOUNT_IGNORE = [
    ".venv",
    ".git",            # this runner reads no git provenance (unlike the wallclock lane)
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
    # EGL/GL + osmesa: headless MuJoCo/MJX rendering backends (same as the probe
    # and the wallclock image). No git/build tooling needed here: pure jax/mjx.
    .apt_install("libegl1", "libgl1", "libglib2.0-0", "libosmesa6")
    .pip_install(*MJX_GPU_PINS)
    .env(
        {
            "PYTHONPATH": "/repo",   # guarantees `import curriculum...` from any cwd
            "MUJOCO_GL": "egl",      # headless render backend
        }
    )
    .add_local_dir(MOUNT_ROOT, "/repo", ignore=MOUNT_IGNORE, copy=False)
)

app = modal.App("z2r-scale-ch2.3-mjx", image=image)


# One GPU, one run. timeout=3600s covers XLA compile + a full 4096-env train.
# gpu="L40S" is the honest Scale-Lab tier (decision 014); override via the
# entrypoint's --gpu, but keep the label honest (see the warning there).
@app.function(gpu="L40S", timeout=3600)
def run_scale(num_envs: int, seed: int, sweep: str, rerun: bool) -> str:
    """Run the UNCHANGED ch2.3 ppo_mjx.py on this GPU. stdout streams live to the
    `modal run` logs (live training curve / throughput). Returns metrics.json
    text (empty in --sweep mode, which measures the cliff and exits)."""
    import subprocess
    import sys

    print("--- z2r Scale Lab: ch2.3 MJX PPO on GPU ---")
    subprocess.run(["nvidia-smi"], check=False)

    out_dir = "outputs/ch2.3-mjx-scale"
    cmd = [
        sys.executable,
        CH_ARTIFACT,
        "--platform", "gpu",
        "--seed", str(seed),
        "--out", out_dir,
    ]
    if sweep.strip():
        # The chapter's --sweep mode times throughput at each num_envs and exits
        # (no training): the wall-clock cliff, now on a GPU that keeps climbing.
        cmd += ["--sweep", sweep.strip(), "--no-rerun"]
    else:
        cmd += ["--num_envs", str(num_envs)]
        cmd += ["--rerun"] if rerun else ["--no-rerun"]

    print(f"[run] {' '.join(cmd)}")
    # Do NOT capture: let the artifact stream its progress to the Modal logs so
    # the learner watches training live. check=False so we still return whatever
    # metrics landed if the run errors late.
    result = subprocess.run(cmd, cwd="/repo")
    if result.returncode != 0:
        print(f"!! ppo_mjx.py exited {result.returncode}")

    metrics = pathlib.Path("/repo") / out_dir / "metrics.json"
    return metrics.read_text() if metrics.exists() else ""


@app.local_entrypoint()
def main(
    num_envs: int = 4096,
    seed: int = 0,
    sweep: str = "",
    gpu: str = "L40S",
    rerun: bool = False,
):
    """Launch the ch2.3 Scale Lab on a real GPU via Modal.

    num_envs: parallel MJX envs for the full run (default 4096, the headline).
    sweep:    comma list of num_envs to time throughput at, then exit (the GPU
              wall-clock cliff, e.g. "256,1024,4096,8192"). Overrides num_envs.
    gpu:      Modal GPU tier. L40S is the honest Scale-Lab default (decision 014).
    rerun:    record a rerun .rrd of the full run (ignored in --sweep mode).
    """
    if gpu.upper() != "L40S":
        # Honesty guardrail (decision 014): a faster/other GPU is fine to run, but
        # it is NOT a 4090 and its numbers are not the recorded l40s Scale-Lab band.
        print(
            f"[note] gpu={gpu} is not the recorded Scale-Lab tier (l40s). Fine to "
            f"run, but its timings are yours alone, not the published l40s band, "
            f"and never a '4090' (decision 014)."
        )

    print(
        "\n=== zero2robot Scale Lab: ch2.3 MJX PPO (OPTIONAL, costs GPU time) ===\n"
        "Free alternative (no account, no cost):\n"
        "  python curriculum/phase2_reinforcement/ch2.3_mjx/ppo_mjx.py --seed 0\n"
        "  python curriculum/phase2_reinforcement/ch2.3_mjx/ppo_mjx.py --sweep 64,256,1024\n"
    )

    fn = run_scale if gpu.upper() == "L40S" else run_scale.with_options(gpu=gpu)
    mode = f"sweep {sweep}" if sweep.strip() else f"full train, num_envs={num_envs}"
    print(f"Running ch2.3 on a {gpu} ({mode}, seed={seed})...\n")

    metrics_text = fn.remote(num_envs, seed, sweep, rerun)

    if sweep.strip():
        print(
            "\nSweep done. The throughput (env-steps/sec) column is the cliff: on a "
            "GPU it keeps climbing past the CPU plateau (~256 envs) toward 4096.\n"
            "GPU timings are statistical (XLA nondeterminism), not bitwise."
        )
        return

    if not metrics_text.strip():
        print("\nNo metrics.json returned. Check the container logs above.")
        return

    out = REPO_ROOT / "outputs" / "ch2.3-mjx-scale" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(metrics_text)
    print(f"\nWrote metrics -> {out}")
    print(metrics_text)
    print(
        "\nReminder: GPU numbers are STATISTICAL (same seed -> same qualitative "
        "result within a band), not bitwise. The free CPU-jax run is the "
        "bitwise-deterministic one."
    )
