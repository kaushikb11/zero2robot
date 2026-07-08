#!/usr/bin/env python3
"""Regenerate the ch3.7 "Datasets at Scale" concept-toy vizdata from scale_data.py.

The site's DataScaleToy island renders REAL numbers — never invented shapes. Two
kinds of data feed it, from two honest sources:

  1. COVERAGE (the hero). We REUSE scale_data.py's OWN wrangle + augment pipeline
     to get the actual object start-poses: the 12 sparse SOURCE-demo starts and
     the ~94 re-solved, success-filtered AUGMENTED starts that MimicGen-style
     perturbation tiled the PushT spawn annulus with. Exactly like ch3.3's
     generator, we exec a PREFIX of the chapter file (up to `# --- region:
     measure ---`) in a throwaway namespace so we get scale_data.py's real
     `pusht_obs / pusht_ep / aug_obs_all / kept / aug_yield / embodiments /
     action_mask` at seed 0, cpu, with ZERO edits to the chapter file. This runs
     the wrangle + augment only (no BC training), so it is fast (~5 s).

  2. THE DATA-SCALE EFFECT (the headline bars). The source-vs-augmented BC
     success rates come from FULL default runs (epochs 500, eval 50, aug 8) that
     take minutes on cpu — so we READ them from meta.yaml's `reference_run`
     (seeds 0-2, measured 2026-07-07), the honest recorded numbers. We PARSE them
     out of meta (never hand-copy) and STOP if the parse or the ordering drifts.

Why exec a PREFIX of scale_data.py instead of `import scale_data`
-----------------------------------------------------------------
scale_data.py is a loose script (argparse + everything at module scope, no
`__main__` guard): importing it runs the WHOLE pipeline including two BC training
runs. We must NOT modify it (LOC-capped, chapter-owned). So we read its source
and exec only the prefix up to the measure region — setup + wrangle + augment —
in an isolated namespace, with argv pinned to seed 0 / cpu / --no-rerun and a
CHAPTER-PRIVATE temp dataset dir (NEVER the default outputs/pusht-demos: that is
ch1.1's 500-demo path, and scale_data.py's ensure_dataset would rebuild it down
to 12 demos — silently destroying the learner's dataset).

    Run:  .venv/bin/python site/scripts/vizdata/ch3.7_scale_data.py
    Out:  curriculum/phase3_advanced/ch3.7_scale_data/demo/vizdata.json
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
CHAPTER = REPO / "curriculum" / "phase3_advanced" / "ch3.7_scale_data"
SCALE_PY = CHAPTER / "scale_data.py"
META_YAML = CHAPTER / "meta.yaml"
OUT_JSON = CHAPTER / "demo" / "vizdata.json"

SEED = 0
CUT_MARKER = "# --- region: measure ---"

# The PushT object-spawn geometry, from pusht_env.py (the annulus the demos tile)
# and scale_data.py's augmentation clip. Rendered as the coverage backdrop.
SPAWN_R = (0.10, 0.24)   # PushTEnv._SPAWN_R — block distance from target at reset
CLIP_R = (0.08, 0.26)    # scale_data.py aug clip on the perturbed start radius

# Coverage grid for the honest "cells filled" number: polar bins over the annulus.
COVERAGE_RINGS = 3       # radial rings across the spawn annulus
COVERAGE_SECTORS = 12    # angular sectors — 36 cells total

ABS_TOL = 1e-6


def exec_scale_prefix() -> dict:
    """Exec scale_data.py up to the measure region, in an isolated namespace with
    a CHAPTER-PRIVATE temp dataset dir. Returns the populated globals (the real
    source demos, augmented demos, and cross-embodiment wrangling)."""
    src = SCALE_PY.read_text()
    cut = src.index(CUT_MARKER)
    prefix = src[:cut]

    scratch = Path(tempfile.mkdtemp(prefix="ch3.7-viz-"))
    old_argv = sys.argv
    # NEVER the default --pusht_data/--aloha_data: ensure_dataset rebuilds on an
    # episode-count mismatch and would clobber ch1.1's 500-demo dataset. Pin a
    # private, throwaway path so we get a clean 12-demo coverage-starved regime.
    sys.argv = [str(SCALE_PY), "--seed", str(SEED), "--no-rerun", "--device", "cpu",
                "--out", str(scratch / "out"),
                "--pusht_data", str(scratch / "pusht"),
                "--aloha_data", str(scratch / "aloha")]
    ns: dict = {"__file__": str(SCALE_PY), "__name__": "scale_toy_vizgen"}
    try:
        exec(compile(prefix, str(SCALE_PY), "exec"), ns)  # noqa: S102 — our own trusted source
    finally:
        sys.argv = old_argv
    return ns


def object_start(obs_row: np.ndarray) -> list[float]:
    """The object (tee) start position from a PushT obs frame. Obs layout (see
    pusht_env._obs): [px, py, tx, ty, sin, cos, gx, gy, sin, cos] — idx 2:4 is
    the block xy, the thing the spawn annulus and the augmentation move around."""
    return [round(float(obs_row[2]), 4), round(float(obs_row[3]), 4)]


def coverage_cells(starts: list[list[float]]) -> int:
    """How many of the COVERAGE_RINGS x COVERAGE_SECTORS polar cells over the
    spawn annulus are occupied by at least one start — an honest scalar for
    'how much of the state space this set covers'."""
    r0, r1 = SPAWN_R
    seen: set[tuple[int, int]] = set()
    for x, y in starts:
        r = float(np.hypot(x, y))
        theta = float(np.arctan2(y, x))  # [-pi, pi)
        ring = int(np.clip((r - r0) / (r1 - r0) * COVERAGE_RINGS, 0, COVERAGE_RINGS - 1))
        sec = int(((theta + np.pi) / (2 * np.pi)) * COVERAGE_SECTORS) % COVERAGE_SECTORS
        seen.add((ring, sec))
    return len(seen)


def parse_reference_run() -> dict:
    """Pull the source-vs-augmented success numbers out of meta.yaml's
    reference_run (seeds 0-2). We PARSE, never hand-copy, so the toy cannot drift
    from the chapter's recorded truth."""
    meta = yaml.safe_load(META_YAML.read_text())
    ref = meta["reference_run"]
    # "source 0.02 -> augmented 0.14 (+0.12); yield 0.98, kept 94/96"
    pat = re.compile(
        r"source\s+([\d.]+)\s*->\s*augmented\s+([\d.]+)\s*\(([+\-][\d.]+)\);"
        r"\s*yield\s+([\d.]+),\s*kept\s+(\d+)/(\d+)"
    )
    seeds, source, augmented, delta, yields, kept, attempts = [], [], [], [], [], [], []
    for i in (0, 1, 2):
        m = pat.search(str(ref[f"seed{i}"]))
        if not m:
            raise SystemExit(f"STOP — could not parse reference_run seed{i} in meta.yaml")
        seeds.append(i)
        source.append(float(m.group(1)))
        augmented.append(float(m.group(2)))
        delta.append(float(m.group(3)))
        yields.append(float(m.group(4)))
        kept.append(int(m.group(5)))
        attempts.append(int(m.group(6)))
    return {
        "seeds": seeds, "source": source, "augmented": augmented, "delta": delta,
        "yield": yields, "kept": kept, "attempts": attempts,
        "mean_source": float(ref["mean_source_success"]),
        "mean_augmented": float(ref["mean_augmented_success"]),
        "mean_scale_effect": float(ref["mean_scale_effect"]),
    }


def main() -> int:
    # ---------------------------------------------------- 1) the coverage (real run)
    ns = exec_scale_prefix()
    pusht_obs = ns["pusht_obs"]
    pusht_ep = ns["pusht_ep"]
    aug_obs_all = ns["aug_obs_all"]
    kept = int(ns["kept"])
    attempts = int(ns["attempts"])
    aug_yield = float(ns["aug_yield"])
    embodiments = ns["embodiments"]
    action_mask = ns["action_mask"]
    n_total = int(ns["n_total"])
    PushTEnv = ns["PushTEnv"]

    source_starts = [object_start(pusht_obs[pusht_ep == ep][0]) for ep in np.unique(pusht_ep)]
    augmented_starts = [object_start(o[0]) for o in aug_obs_all]
    source_demos = len(source_starts)

    src_cells = coverage_cells(source_starts)
    aug_cells = coverage_cells(augmented_starts)
    total_cells = COVERAGE_RINGS * COVERAGE_SECTORS

    # ------------------------------------------------ 2) the success bars (from meta)
    ref = parse_reference_run()

    # ------------------------------------------------------- cross-embodiment (real)
    emb = {e["name"]: e for e in embodiments}
    pusht_act_dim = int(emb["pusht"]["act_dim"])
    aloha_act_dim = int(emb["aloha"]["act_dim"])
    padded = int(max(pusht_act_dim, aloha_act_dim))
    pusht_mask_density = round(pusht_act_dim / padded, 6)      # 2/6 = 0.3333
    mixed_mask_density = round(float(action_mask.mean()), 6)   # weighted over the pile

    # -------------------------------------------------------------- honesty gate
    print("regenerated ch3.7 [seed 0, cpu] vs meta.yaml reference_run:")
    print(f"  source demos    : {source_demos}   (coverage-starved regime, meta 12)")
    print(f"  augmentation    : kept {kept}/{attempts} (yield {aug_yield:.3f})   "
          f"(meta seed0 kept {ref['kept'][0]}/{ref['attempts'][0]}, yield {ref['yield'][0]})")
    print(f"  coverage cells  : source {src_cells}/{total_cells} -> augmented {aug_cells}/{total_cells}")
    print(f"  success (meta)  : source mean {ref['mean_source']:.3f} -> "
          f"augmented mean {ref['mean_augmented']:.3f} (delta {ref['mean_scale_effect']:+.3f})")
    print(f"  cross-embodiment: pusht act_dim {pusht_act_dim}, aloha act_dim {aloha_act_dim}, "
          f"pusht mask density {pusht_mask_density}")

    fail: list[str] = []
    if source_demos != 12:
        fail.append(f"source demos {source_demos} != 12 (coverage-starved regime lost)")
    if attempts != source_demos * ns["args"].aug_per_demo:
        fail.append(f"attempts {attempts} != {source_demos} x aug_per_demo")
    # our real seed-0 augmentation must reproduce meta's seed-0 kept/attempts
    if kept != ref["kept"][0] or attempts != ref["attempts"][0]:
        fail.append(f"seed0 augmentation kept {kept}/{attempts} != meta {ref['kept'][0]}/{ref['attempts'][0]}")
    if abs(aug_yield - ref["yield"][0]) > 0.02:
        fail.append(f"seed0 yield {aug_yield:.3f} not ~ meta {ref['yield'][0]}")
    # the coverage story: augmentation must fill strictly more of the annulus
    if not (aug_cells > src_cells):
        fail.append(f"coverage did not grow: source {src_cells} !< augmented {aug_cells}")
    # meta reference_run internal consistency (parse sanity vs recorded means)
    if abs(float(np.mean(ref["source"])) - ref["mean_source"]) > 5e-3:
        fail.append("parsed source seeds disagree with meta mean_source_success")
    if abs(float(np.mean(ref["augmented"])) - ref["mean_augmented"]) > 5e-3:
        fail.append("parsed augmented seeds disagree with meta mean_augmented_success")
    # THE headline ordering — augmented > source on EVERY seed (seed-robust)
    for i, (s, a) in enumerate(zip(ref["source"], ref["augmented"])):
        if not (a > s):
            fail.append(f"seed{i}: augmented {a} not > source {s} (ordering broken)")
    if not (ref["mean_augmented"] > ref["mean_source"]):
        fail.append("mean augmented not > mean source")
    # cross-embodiment heterogeneity — the ch1.7 padding lesson
    if pusht_act_dim != 2 or aloha_act_dim != 6:
        fail.append(f"embodiment act dims {pusht_act_dim}/{aloha_act_dim} != 2/6")
    if abs(pusht_mask_density - 1 / 3) > ABS_TOL:
        fail.append(f"pusht mask density {pusht_mask_density} != 1/3")
    if fail:
        print("\nSTOP — regenerated ch3.7 does NOT match meta.yaml:")
        for f in fail:
            print("  x " + f)
        return 1

    # ------------------------------------------------------------------- pack json
    data = {
        "provenance": {
            "source": "curriculum/phase3_advanced/ch3.7_scale_data/scale_data.py",
            "generator": "site/scripts/vizdata/ch3.7_scale_data.py",
            "seed": SEED,
            "device": "cpu",
            "stack": f"numpy {np.__version__}",
            "coverage_source": "REAL run: scale_data.py wrangle+augment prefix (seed 0, cpu, "
                               "chapter-private 12-demo dataset) — actual source + augmented object starts",
            "success_source": "meta.yaml reference_run (seeds 0-2, full default config, "
                              "epochs 500 / eval 50 / aug 8, measured 2026-07-07) — parsed, not hand-copied",
            "note": "The absolute success is deliberately modest (~2-30%): 12 source demos is a "
                    "coverage-starved regime chosen so the DATA effect is visible. The ORDERING "
                    "(augmented > source on every seed) is the point — the ch1.2 thesis scaled. "
                    "The policy did not change between the two bars, only the data did.",
        },
        # --- panel 1: the data-scale effect (headline bars, error bars from 3 seeds)
        "success": {
            "seeds": ref["seeds"],
            "source": ref["source"],
            "augmented": ref["augmented"],
            "delta": ref["delta"],
            "source_mean": round(ref["mean_source"], 4),
            "augmented_mean": round(ref["mean_augmented"], 4),
            "scale_effect_mean": round(ref["mean_scale_effect"], 4),
            "source_min": round(min(ref["source"]), 4),
            "source_max": round(max(ref["source"]), 4),
            "augmented_min": round(min(ref["augmented"]), 4),
            "augmented_max": round(max(ref["augmented"]), 4),
        },
        # --- panel 2: coverage (real object starts + the honest "cells filled" scalar)
        "coverage": {
            "seed": SEED,
            "spawn_r": list(SPAWN_R),
            "clip_r": list(CLIP_R),
            "source_starts": source_starts,
            "augmented_starts": augmented_starts,
            "source_n": source_demos,
            "augmented_n": len(augmented_starts),
            "kept": kept,
            "attempts": attempts,
            "yield": round(aug_yield, 4),
            "grid": {
                "rings": COVERAGE_RINGS,
                "sectors": COVERAGE_SECTORS,
                "total_cells": total_cells,
                "source_cells": src_cells,
                "augmented_cells": aug_cells,
            },
        },
        # --- panel 3: the cross-embodiment padding note (heterogeneous action dims)
        "cross_embodiment": {
            "embodiments": [
                {"name": "pusht", "act_dim": pusht_act_dim, "frames": int(emb["pusht"]["frames"])},
                {"name": "aloha", "act_dim": aloha_act_dim, "frames": int(emb["aloha"]["frames"])},
            ],
            "padded_action_dim": padded,
            "mixed_frames": n_total,
            "pusht_mask_density": pusht_mask_density,
            "mixed_mask_density": mixed_mask_density,
            "obs_dim": int(PushTEnv.OBS_DIM),
        },
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB)")
    print(f"OK — matches meta.yaml; source {ref['mean_source']:.2f} -> augmented "
          f"{ref['mean_augmented']:.2f} (every seed); coverage {src_cells}->{aug_cells} of "
          f"{total_cells} cells; augmentation kept {kept}/{attempts}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
