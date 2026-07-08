#!/usr/bin/env python3
"""Regenerate the ch2.7 domain-randomization concept-toy vizdata — the HONEST
"narrow vs randomized across the gap" numbers, with ch1.6-style seed-band error
bars. Sibling of site/scripts/vizdata/ch3.3_engine.py.

Why we READ the measured reference_run instead of running dr.py --smoke
----------------------------------------------------------------------
dr.py trains TWO PPO policies from scratch and sweeps both across a shifted-mass
gap (see meta.yaml). The chapter's whole LESSON is a 400k-step, seeds-0-2 result:
the off-nominal DR survival edge is {-0.02, +0.22, -0.09} — it swings with the
seed and its mean sits INSIDE the seed band. A --smoke run (1024 steps, seed 0,
3 sweep points) trains essentially-untrained policies and would NOT reproduce
that measured story — it would invent a different, misleading picture. So, exactly
like ch1.6's EvalBandsToy (which pins its measured reference_run constants rather
than re-running), this generator transcribes dr.py's MEASURED reference_run
survival table from meta.yaml and computes the cross-seed statistics the toy shows.

The honesty gate (STOP-on-drift vs meta)
----------------------------------------
The transcribed survival table below is the SOURCE. We then re-open meta.yaml and
cross-check every quantitative claim meta makes against numbers RECOMPUTED from
that table:
  * the off-nominal survival edge triple {-0.02, +0.22, -0.09} (parsed out of
    exercise_checks.ex1.provenance) must equal our per-seed recompute;
  * nominal survival must clear exercise_checks.ex1.nominal_survival_min (0.9);
  * deepest-gap survival must stay under exercise_checks.ex1.deepgap_survival_max
    (0.35) AND ex2.deepgap_survival_max.
If any drift, we STOP — the toy never renders a number meta does not back.

    Run:  .venv/bin/python site/scripts/vizdata/ch2.7_dr.py
    Out:  curriculum/phase2_reinforcement/ch2.7_dr/demo/vizdata.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
CH = REPO / "curriculum" / "phase2_reinforcement" / "ch2.7_dr"
META_YAML = CH / "meta.yaml"
OUT_JSON = CH / "demo" / "vizdata.json"

# ---------------------------------------------------------------------------
# The MEASURED reference_run, transcribed verbatim from meta.yaml's comment block
# (PROVENANCE: 2026-07-06, cpu, torch 2.10.0 / mujoco 3.10.0 / numpy 2.4.6,
# --total_steps 400000, --sweep_knob mass, eval_episodes 16, seeds 0-2). This is
# dr.py's own survival rate per policy across the mass gap, for seeds 0 / 1 / 2:
#
#                 mass 0.8    1.0     1.2            1.4            1.6
#   narrow        1.00        1.00    0.44/0.94/0.88 0.00/0.38/0.44 0.00/0.00/0.06
#   randomized    1.00        1.00    0.38/1.00/1.00 0.00/1.00/0.00 0.00/0.19/0.00
#
# survival[policy][scale_index] = [seed0, seed1, seed2].
# ---------------------------------------------------------------------------
SWEEP_KNOB = "mass"
SWEEP_GRID = [0.8, 1.0, 1.2, 1.4, 1.6]
NOMINAL_IDX = SWEEP_GRID.index(1.0)
SEEDS = [0, 1, 2]

SURVIVAL: dict[str, list[list[float]]] = {
    "narrow": [
        [1.00, 1.00, 1.00],   # mass 0.8
        [1.00, 1.00, 1.00],   # mass 1.0 (nominal)
        [0.44, 0.94, 0.88],   # mass 1.2
        [0.00, 0.38, 0.44],   # mass 1.4
        [0.00, 0.00, 0.06],   # mass 1.6 (deepest gap)
    ],
    "randomized": [
        [1.00, 1.00, 1.00],   # mass 0.8
        [1.00, 1.00, 1.00],   # mass 1.0 (nominal)
        [0.38, 1.00, 1.00],   # mass 1.2
        [0.00, 1.00, 0.00],   # mass 1.4
        [0.00, 0.19, 0.00],   # mass 1.6 (deepest gap)
    ],
}
POLICIES = ["narrow", "randomized"]

# meta records nominal_return ~203-207 (flat across dr_width {0,1,2}) — i.e. no
# measurable insurance premium at this budget. Recorded for the readout note; not
# a per-seed/per-shift array (meta gives no such table), so the survival curves —
# dr.py's OWN "honest binary signal" — carry the error bars.
NOMINAL_RETURN_LO, NOMINAL_RETURN_HI = 203.0, 207.0

ABS_TOL = 0.011   # meta rounds the edge/survival to 2 dp; guard print noise only.


def off_nominal_indices() -> list[int]:
    return [i for i in range(len(SWEEP_GRID)) if i != NOMINAL_IDX]


def per_seed_offnominal(policy: str, seed_idx: int) -> float:
    """Mean survival over the four off-nominal mass points for one seed."""
    return float(np.mean([SURVIVAL[policy][i][seed_idx] for i in off_nominal_indices()]))


def main() -> int:
    off = off_nominal_indices()

    # ---- cross-seed aggregate curves: mean ± std (the seed band = the error bar)
    aggregate: dict[str, list[dict]] = {}
    for policy in POLICIES:
        curve = []
        for i, scale in enumerate(SWEEP_GRID):
            vals = np.asarray(SURVIVAL[policy][i], dtype=float)
            mean = float(vals.mean())
            std = float(vals.std())  # population std across the 3 seeds
            curve.append({
                "scale": scale,
                "mean": round(mean, 4),
                "std": round(std, 4),
                "lo": round(max(0.0, mean - std), 4),
                "hi": round(min(1.0, mean + std), 4),
                "seeds": [round(float(v), 4) for v in vals],
            })
        aggregate[policy] = curve

    # ---- per-seed curves + per-seed off-nominal DR edge (randomized - narrow)
    per_seed: dict[str, dict] = {}
    edge_per_seed: list[float] = []
    for s_idx, seed in enumerate(SEEDS):
        n_off = per_seed_offnominal("narrow", s_idx)
        r_off = per_seed_offnominal("randomized", s_idx)
        edge = r_off - n_off
        edge_per_seed.append(edge)
        per_seed[str(seed)] = {
            "narrow": [round(SURVIVAL["narrow"][i][s_idx], 4) for i in range(len(SWEEP_GRID))],
            "randomized": [round(SURVIVAL["randomized"][i][s_idx], 4) for i in range(len(SWEEP_GRID))],
            "narrow_offnominal": round(n_off, 4),
            "randomized_offnominal": round(r_off, 4),
            "edge": round(edge, 4),
        }

    edge_arr = np.asarray(edge_per_seed, dtype=float)
    edge_mean = float(edge_arr.mean())
    edge_std = float(edge_arr.std())

    # aggregate nominal + off-nominal survival (mean over seeds)
    def agg_at(policy: str, i: int) -> float:
        return float(np.mean(SURVIVAL[policy][i]))

    nominal = {p: round(agg_at(p, NOMINAL_IDX), 4) for p in POLICIES}
    offnominal = {
        p: round(float(np.mean([agg_at(p, i) for i in off])), 4) for p in POLICIES
    }
    deepgap_idx = len(SWEEP_GRID) - 1
    deepgap = {p: round(agg_at(p, deepgap_idx), 4) for p in POLICIES}

    # ======================================================= honesty gate vs meta
    meta = yaml.safe_load(META_YAML.read_text())
    checks = meta["exercise_checks"]
    ex1, ex2 = checks["ex1"], checks["ex2"]

    # (a) the edge triple {-0.02,+0.22,-0.09}, parsed out of ex1.provenance prose.
    m = re.search(r"edge\s*\{([^}]*)\}", ex1["provenance"])
    if not m:
        print("STOP — could not find the survival-edge triple in meta ex1.provenance")
        return 1
    meta_edge = [float(x) for x in m.group(1).split(",")]

    fail: list[str] = []
    if len(meta_edge) != len(SEEDS):
        fail.append(f"meta edge triple has {len(meta_edge)} entries, expected {len(SEEDS)}")
    else:
        for seed, ours, theirs in zip(SEEDS, edge_per_seed, meta_edge):
            if abs(round(ours, 2) - theirs) > ABS_TOL:
                fail.append(f"seed {seed} edge {ours:+.4f} (→{round(ours,2):+.2f}) != meta {theirs:+.2f}")

    # (b) nominal survival clears meta's floor (both policies stand at nominal).
    nominal_floor = float(ex1["nominal_survival_min"])
    nominal_min = min(min(SURVIVAL[p][NOMINAL_IDX]) for p in POLICIES)
    if nominal_min < nominal_floor - 1e-9:
        fail.append(f"nominal survival min {nominal_min} < meta floor {nominal_floor}")

    # (c) deepest-gap survival stays under meta's ceiling (both collapse at 1.6x) —
    #     checked against BOTH ex1 and ex2 caps.
    deepgap_cap = min(float(ex1["deepgap_survival_max"]), float(ex2["deepgap_survival_max"]))
    deepgap_worst = max(max(SURVIVAL[p][deepgap_idx]) for p in POLICIES)
    if deepgap_worst > deepgap_cap + 1e-9:
        fail.append(f"deepest-gap survival {deepgap_worst} > meta cap {deepgap_cap}")

    print("regenerated ch2.7 DR [measured reference_run, seeds 0-2] vs meta.yaml:")
    print("  off-nominal survival edge (rand - narrow) per seed: "
          + ", ".join(f"{e:+.2f}" for e in edge_per_seed)
          + f"   (meta {', '.join(f'{e:+.2f}' for e in meta_edge)})")
    print(f"  edge mean {edge_mean:+.3f} ± {edge_std:.3f}  → within the seed band "
          f"(|mean| < std: {abs(edge_mean) < edge_std})")
    print(f"  nominal survival  narrow {nominal['narrow']:.2f} / randomized {nominal['randomized']:.2f}  "
          f"(floor {nominal_floor})")
    print(f"  deepest-gap (1.6x) narrow {deepgap['narrow']:.2f} / randomized {deepgap['randomized']:.2f}  "
          f"(cap {deepgap_cap})")

    # the within-band claim IS the lesson: assert the mean edge does not clear the band.
    if abs(edge_mean) >= edge_std:
        fail.append(f"edge mean {edge_mean:+.4f} is NOT within the seed band (std {edge_std:.4f}) — "
                    "the honest 'within-band' story would be broken; refusing to render a clean-win overclaim")

    if fail:
        print("\nSTOP — regenerated DR toy data does NOT match meta.yaml:")
        for f in fail:
            print("  x " + f)
        return 1

    # ================================================================ pack + write
    data = {
        "provenance": {
            "source": "curriculum/phase2_reinforcement/ch2.7_dr/dr.py "
                      "(MEASURED reference_run in meta.yaml — not a live re-run)",
            "generator": "site/scripts/vizdata/ch2.7_dr.py",
            "measured": "2026-07-06, cpu, torch 2.10.0 / mujoco 3.10.0 / numpy 2.4.6",
            "config": "--total_steps 400000, --sweep_knob mass, --eval_episodes 16, seeds 0-2",
            "metric": "survival rate (fraction of held-out episodes that reached the "
                      "time limit without falling) — dr.py's own honest binary signal",
            "note": "Transcribed from meta.yaml reference_run; the survival edge triple, "
                    "the nominal floor, and the deepest-gap cap are cross-checked against "
                    "meta's exercise_checks and the script STOPS on any drift. A --smoke "
                    "run (1024 steps) would NOT reproduce this 400k-step measured story.",
        },
        "sweep_knob": SWEEP_KNOB,
        "sweep_grid": SWEEP_GRID,
        "nominal_idx": NOMINAL_IDX,
        "deepgap_idx": deepgap_idx,
        "seeds": SEEDS,
        "policies": POLICIES,
        # cross-seed aggregate (the default view): mean ± seed-band std per mass point.
        "aggregate": aggregate,
        # per-seed curves (the "single numbers lie" reveal — flip the seed, flip the story).
        "per_seed": per_seed,
        "edge": {
            "per_seed": [round(e, 4) for e in edge_per_seed],
            "per_seed_rounded": [round(e, 2) for e in edge_per_seed],  # {-0.02,+0.22,-0.09}
            "mean": round(edge_mean, 4),
            "std": round(edge_std, 4),
            "within_band": bool(abs(edge_mean) < edge_std),
            "meta": [round(x, 2) for x in meta_edge],
        },
        "nominal": nominal,
        "offnominal": offnominal,
        "deepgap": deepgap,
        "nominal_return_band": [NOMINAL_RETURN_LO, NOMINAL_RETURN_HI],
        # the honest scope note — the toy's panel-3 line.
        "scope_note": "Domain randomization is the promise you TEST, not a guaranteed win. "
                      "At this free-tier budget (400k steps/policy) the off-nominal survival "
                      "edge is +{:.0%} ± {:.0%} across seeds — it sits INSIDE the seed band, "
                      "and past ~1.2x mass the ±12 Nm servos saturate so BOTH policies fall. "
                      "The Scale Lab spends the compute (and/or a walking gait) to make the "
                      "randomized policy converge reliably.".format(edge_mean, edge_std),
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB)")
    print("OK — matches meta.yaml; DR edge is WITHIN the seed band (honest, not a clean win).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
