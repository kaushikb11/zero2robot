"""Regenerate the 2D-ring viz data for the ch1.4 diffusion multimodality toy.

The site toy (site/src/components/toys/DiffusionRingToy.tsx) draws the chapter's
ROCK: a multimodal ring target, the DIFFUSION samples that cover all 8 angular
modes and sit ON the ring, versus a same-width MSE REGRESSOR that collapses to the
dead center. Root CLAUDE.md invariant: numbers have provenance. So this file does
NOT invent points — it runs the REAL diffusion.py toy code at seed 0 and dumps the
sample clouds it produces.

WHY WE EXEC THE REAL REGIONS (not import, not re-implement)
----------------------------------------------------------
diffusion.py is a flat teaching script (no __main__ guard): importing it runs the
whole pipeline and its `setup` region pulls in mujoco + lerobot (the PushT policy),
which the 2D toy does not need and which is not installed on the free-tier docs
box. But the DDPM machinery and the toy itself live in two regions — `core`
(schedule / epsilon-denoiser / DDPM sampler) and `toy` (the ring, the two training
loops, ring_stats) — that are pure torch + numpy + math. We extract those two
regions VERBATIM by their `# --- region: … ---` markers and exec them under a tiny
setup shim that replicates ONLY the handful of setup-region constants the toy reads
(TIME_DIM, X0_CLIP, N_TOY, the shared `gen`, `randn`, device) with their exact
diffusion.py values. So the points come from the chapter's own code path, unchanged.

WHAT WE DUMP
------------
For each denoising-step count the toy exposes (2 / 10 / 100) we re-run the toy at
seed 0 with that schedule — exactly what `diffusion.py --denoising_steps S` (and,
at S=2, `--break few_steps`) does — and keep the diffusion sample cloud. We keep
ONE regression cloud (from the canonical S=100 run, matching meta's default) since
the regressor is schedule-independent. Everything is subsampled to ~300 points/set
(deterministic) so vizdata.json stays tiny text (no binary).

PARITY GATE (honesty)
---------------------
Before writing, we verify the regenerated clouds match the MEASURED reference in
meta.yaml (diffusion 8/8 modes, mean_radius ~0.87; regression 0 modes, ~0.06; and
the 2-step Break-It degrades to 7/8, ~0.70). meta was measured on torch 2.10; this
box may run a different torch whose RNG stream differs slightly, so radii are
checked within a band while the mode counts (objective, capacity-robust) are exact.
On mismatch we STOP without writing — never fabricate.

Run:  python site/scripts/vizdata/ch1.4_diffusion.py
      python site/scripts/vizdata/ch1.4_diffusion.py --points 300
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn  # noqa: F401  (referenced by the exec'd core region)
import torch.nn.functional as F  # noqa: F401  (referenced by the exec'd core region)

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))


def _seed_cpu(seed: int) -> None:
    """The CPU-relevant half of curriculum.common.seeding.set_seed, inlined.

    We can't call set_seed directly here: it also calls torch.mps.manual_seed,
    which the pinned-for-measurement torch has but this docs box's older torch
    lacks. The mps/cuda branches seed INDEPENDENT device RNGs the 2D toy never
    draws from on CPU, so replicating only the CPU lines reproduces diffusion.py's
    exact CPU stream (random -> numpy -> torch global, same order)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

_CH = _ROOT / "curriculum" / "phase1_imitation" / "ch1.4_diffusion"
_DIFFUSION_PY = _CH / "diffusion.py"
_OUT = _CH / "demo" / "vizdata.json"

# The steps the toy switches between; 100 is diffusion.py's default schedule, 2 is
# its `--break few_steps` misconception, 10 is an honest midpoint.
STEP_COUNTS = [2, 10, 100]
DEFAULT_STEPS = 100

# --- MEASURED reference (meta.yaml reference_run + Break-It signatures). We match
#     these; mode counts exact, radii within a torch-version band. -----------------
META_DIFF_MODES = 8          # toy_diffusion_modes_covered
META_DIFF_RADIUS = 0.870782  # toy_diffusion_mean_radius
META_REG_MODES = 0           # toy_regress_modes_covered
META_REG_RADIUS = 0.057346   # toy_regress_mean_radius
META_BREAK2_MODES = 7        # break_few_steps_toy_modes
META_BREAK2_RADIUS = 0.70    # break_few_steps: mean_radius 0.87 -> 0.70
RADIUS_BAND = 0.10           # torch-RNG tolerance on mean_radius across versions


def _extract_region(src: str, name: str) -> str:
    """Return the verbatim body of a `# --- region: <name> ---` block."""
    open_tag = f"# --- region: {name} ---"
    close_tag = "# --- endregion ---"
    start = src.index(open_tag) + len(open_tag)
    end = src.index(close_tag, start)
    return src[start:end]


def _run_toy(steps: int) -> dict:
    """Exec diffusion.py's verbatim core+toy regions at seed 0 with an S-step
    schedule; return the sample clouds and the measured ring stats."""
    src = _DIFFUSION_PY.read_text()
    core_src = _extract_region(src, "core")
    toy_src = _extract_region(src, "toy")

    device = torch.device("cpu")
    # Reproduce diffusion.py's setup order EXACTLY: set_seed seeds the torch global
    # RNG (used by the denoiser weight inits), then a private `gen` feeds every
    # stochastic draw (noise, batch indices, the sampler) in a fixed order.
    _seed_cpu(0)
    gen = torch.Generator().manual_seed(0)

    def randn(shape):  # verbatim from diffusion.py setup
        return torch.randn(shape, generator=gen).to(device)

    ns: dict = {
        "torch": torch, "nn": nn, "F": F, "np": np, "math": math,
        "device": device, "gen": gen, "randn": randn,
        # setup-region constants the toy reads (verbatim values from diffusion.py):
        "TIME_DIM": 32, "X0_CLIP": 3.0, "N_TOY": 2000,
        "TOY_ITERS": 1500,  # the not-smoke value from setup
        "BROKEN_SCHEDULE": False,
        "args": SimpleNamespace(
            denoising_steps=steps, model_dim=128, lr=1e-3,
            rerun=False, smoke=False, break_mode=None,
        ),
    }
    exec(compile(core_src, str(_DIFFUSION_PY) + "::core", "exec"), ns)
    exec(compile(toy_src, str(_DIFFUSION_PY) + "::toy", "exec"), ns)
    return {
        "diff": ns["diff_samples"], "reg": ns["reg_samples"], "ring": ns["ring"],
        "diff_r": ns["diff_r"], "diff_modes": ns["diff_modes"],
        "reg_r": ns["reg_r"], "reg_modes": ns["reg_modes"],
        "ring_stats": ns["ring_stats"],
    }


def _subsample(arr, n: int, rng) -> list[list[float]]:
    """Deterministic ~n-point subsample, rounded to 4 dp for a tiny text dump."""
    a = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    m = len(a)
    idx = np.arange(m) if m <= n else np.sort(rng.choice(m, size=n, replace=False))
    # 3 dp: ~1 mm on the unit ring — sub-pixel in the toy's ~460px view, and it
    # keeps the dump tiny.
    return [[round(float(a[i, 0]), 3), round(float(a[i, 1]), 3)] for i in idx]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--points", type=int, default=300, help="points per set in the dump")
    ap.add_argument("--out", type=Path, default=_OUT)
    args = ap.parse_args(argv)

    print(f"regenerating ch1.4 ring viz data from {_DIFFUSION_PY.relative_to(_ROOT)} "
          f"(torch {torch.__version__}, seed 0)")
    runs = {s: _run_toy(s) for s in STEP_COUNTS}

    canon = runs[DEFAULT_STEPS]
    reg_modes, reg_r = canon["reg_modes"], canon["reg_r"]
    # meta measured 8/8 vs 0/8, radius 0.87 vs 0.06 at the default 100-step schedule.
    diff_modes = canon["diff_modes"]
    diff_r = canon["diff_r"]
    print(f"  [S=100] diffusion modes {diff_modes}/8  mean_radius {diff_r:.4f}   "
          f"(meta {META_DIFF_MODES}/8, {META_DIFF_RADIUS:.4f})")
    print(f"  [S=100] regression modes {reg_modes}/8  mean_radius {reg_r:.4f}   "
          f"(meta {META_REG_MODES}/8, {META_REG_RADIUS:.4f})")
    b2 = runs[2]
    print(f"  [S=2]   diffusion modes {b2['diff_modes']}/8  mean_radius {b2['diff_r']:.4f}   "
          f"(meta Break-It {META_BREAK2_MODES}/8, {META_BREAK2_RADIUS:.2f})")
    print(f"  [S=10]  diffusion modes {runs[10]['diff_modes']}/8  mean_radius {runs[10]['diff_r']:.4f}   "
          f"(honest midpoint; not a meta reference)")

    # --- parity gate: match meta or STOP (never fabricate) --------------------
    problems = []
    if diff_modes != META_DIFF_MODES:
        problems.append(f"diffusion modes {diff_modes} != meta {META_DIFF_MODES}")
    if reg_modes != META_REG_MODES:
        problems.append(f"regression modes {reg_modes} != meta {META_REG_MODES}")
    if abs(diff_r - META_DIFF_RADIUS) > RADIUS_BAND:
        problems.append(f"diffusion radius {diff_r:.4f} outside {META_DIFF_RADIUS:.4f}±{RADIUS_BAND}")
    if abs(reg_r - META_REG_RADIUS) > RADIUS_BAND:
        problems.append(f"regression radius {reg_r:.4f} outside {META_REG_RADIUS:.4f}±{RADIUS_BAND}")
    if b2["diff_modes"] >= META_DIFF_MODES:
        problems.append(f"Break-It S=2 did not degrade modes ({b2['diff_modes']}/8)")
    if problems:
        print("\nSTOP — regenerated data does NOT match meta.yaml; not writing:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    # --- assemble the tiny JSON ----------------------------------------------
    rng = np.random.default_rng(0)  # deterministic subsampling
    diffusion = {
        str(s): {
            "modes_covered": int(runs[s]["diff_modes"]),
            "mean_radius": round(float(runs[s]["diff_r"]), 6),
            "points": _subsample(runs[s]["diff"], args.points, rng),
        }
        for s in STEP_COUNTS
    }
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(_ROOT), text=True
        ).strip()
    except Exception:
        git_sha = "unknown"

    out = {
        "schema": "z2r-diffusion-ring-1",
        "chapter": "ch1.4-diffusion",
        "demo": "diffusion_multimodality",
        "target_radius": 1.0,
        "n_sectors": 8,               # angular modes ring_stats bins into
        "default_steps": DEFAULT_STEPS,
        "step_counts": STEP_COUNTS,
        # grey ring target — every direction an equally-good mode, the center the
        # one place no data sits.
        "target": {"points": _subsample(canon["ring"], args.points, rng)},
        # blue — diffusion samples per denoising-step schedule (2 under-denoises).
        "diffusion": diffusion,
        # red — the same-width MSE regressor, collapsed to the dead center.
        "regression": {
            "modes_covered": int(reg_modes),
            "mean_radius": round(float(reg_r), 6),
            "points": _subsample(canon["reg"], args.points, rng),
        },
        "provenance": {
            "derived_from": "the ch1.4 diffusion.py 2D toy (ring target, from-scratch "
                            "DDPM epsilon-denoiser + reverse sampler, and the same-width "
                            "MSE regressor baseline) — the chapter's own code path, unchanged",
            "method": "exec of diffusion.py's verbatim `core` + `toy` regions under a "
                      "setup shim (only the toy's pure torch/numpy path; mujoco/lerobot "
                      "policy regions skipped) at seed 0",
            "generator": "site/scripts/vizdata/ch1.4_diffusion.py",
            "seed": 0,
            "points_per_set": args.points,
            "toy_iters": 1500,
            "matches_meta": {
                "toy_diffusion_modes_covered": int(diff_modes),
                "toy_diffusion_mean_radius": round(float(diff_r), 6),
                "toy_regress_modes_covered": int(reg_modes),
                "toy_regress_mean_radius": round(float(reg_r), 6),
                "break_few_steps_toy_modes": int(b2["diff_modes"]),
                "meta_reference": {
                    "toy_diffusion_modes_covered": META_DIFF_MODES,
                    "toy_diffusion_mean_radius": META_DIFF_RADIUS,
                    "toy_regress_modes_covered": META_REG_MODES,
                    "toy_regress_mean_radius": META_REG_RADIUS,
                    "break_few_steps_toy_modes": META_BREAK2_MODES,
                },
                "note": "mode counts exact; radii within a torch-version RNG band "
                        f"(±{RADIUS_BAND}). meta measured on torch 2.10; "
                        f"regenerated on torch {torch.__version__}.",
            },
            "versions": {"torch": torch.__version__, "numpy": np.__version__},
            "generated": str(date.today()),
            "git": git_sha,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Compact separators: the point clouds are machine data, not hand-edited, and
    # nested-array indenting would bloat the file ~4x. Stays tiny text.
    args.out.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    size_kb = args.out.stat().st_size / 1024
    print(f"\nwrote {args.out.relative_to(_ROOT)}  ({size_kb:.1f} KB, text)")
    print(f"  diffusion covers {diff_modes}/8 modes (r={diff_r:.3f}); "
          f"regression {reg_modes}/8 (r={reg_r:.3f}) — collapsed to center")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
