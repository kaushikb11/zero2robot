#!/usr/bin/env python3
"""Regenerate the ch3.5 contact concept-toy vizdata from contact.py, seed 0.

The site's ContactToy island renders REAL contact trajectories — never invented
shapes. This generator REUSES contact.py's own from-scratch pieces (build_scene,
the PENALTY spring-damper step, the LCP-flavored projected-Gauss-Seidel step, and
the honesty metrics contact_quality / max_penetration / energy) so the browser
replay is bit-faithful to the chapter artifact, then dumps a small JSON the island
loads. Three views come out of one deterministic run:

  1. THE DROP — a ball falling onto the table under PENALTY vs LCP. Penalty drives
     ~25% of the radius INTO the table on impact, rings, and rests SUNK by mg/k;
     LCP catches it in ~one step and holds it exactly on the surface.
  2. THE dt CLIFF — sweep the timestep. Penalty's stiff spring is only stable while
     dt < ~2*sqrt(m/k); past that its energy detonates (267x -> 2600x). LCP has no
     spring to blow up, so it stays bounded across the whole grid.
  3. PENETRATION over time — the honesty metric, per method, from the same drop.

Why we exec a PREFIX of contact.py instead of `import contact`
--------------------------------------------------------------
contact.py is a loose script (no `if __name__ == "__main__"` guard): importing it
runs the WHOLE report — argparse, both solvers, the stability probe, metrics.json,
the .rrd recording. We must NOT modify contact.py (it is LOC-capped at 450). So we
read its source and exec only the prefix up to the `# --- region: report ---`
marker — setup + scenes + contacts + penalty + lcp + simulate — in a throwaway
namespace. That gives us contact.py's OWN build_scene, semi_implicit_step,
penalty_force, lcp_step, simulate, contact_quality at the DEFAULT config (seed 0,
cpu, pure numpy), with zero edits to the file and none of the report side effects.

Everything here is pure-numpy and bitwise deterministic on CPU, so the honesty
gate below is TIGHT — it reproduces meta.yaml's measured numbers exactly and the
script STOPS if any drift.

    Run:  .venv/bin/python site/scripts/vizdata/ch3.5_contact.py
    Out:  curriculum/phase3_advanced/ch3.5_contact/demo/vizdata.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CONTACT_PY = REPO / "curriculum" / "phase3_advanced" / "ch3.5_contact" / "contact.py"
OUT_JSON = REPO / "curriculum" / "phase3_advanced" / "ch3.5_contact" / "demo" / "vizdata.json"

SEED = 0
STIFFNESS = 1.0e4       # k — the penalty spring constant (contact.py default)
BAUMGARTE = 0.2         # lcp position-correction gain (contact.py default)
ITERS = 20              # projected Gauss-Seidel sweeps (contact.py default)
REF_DT = 0.002          # the reference timestep
REF_STEPS = 2000        # the reference horizon (4 s of sim)
CUT_MARKER = "# --- region: report ---"

# The drop-replay window: the whole story (fall -> impact -> ring -> settle) lives
# in the first ~1.6 s; the tail is just the body sitting still. Subsample by STRIDE
# for a small committed file that still resolves penalty's ring (period ~0.063 s).
WIN_STEPS = 800         # 1.6 s at REF_DT — penalty is fully settled by here
STRIDE = 4              # keep every 4th step -> 201 frames at 8 ms spacing

# The dt-cliff sweep: the slider's stops. Spans embed.yaml's dt slider (0.001..0.04),
# straddling dt_crit = 2*sqrt(m/k) = 0.02 so the learner can drag penalty over the edge.
DT_SWEEP = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.016, 0.02, 0.024, 0.03, 0.04]
SWEEP_STEPS = 1500      # enough for penalty to detonate and lcp to settle at each dt
# A run is "stable" if its energy stayed bounded (<10x the drop energy) and finite —
# the SAME criterion contact.py's stability probe uses.
STABLE_ENERGY_MULT = 10.0

# meta.yaml reference_run (seed 0, cpu, numpy 2.4.6) — the honesty gate. If the
# regenerated trajectories drift from these MEASURED numbers, STOP.
META = {
    "penalty_max_penetration_frac": 0.251329183485,
    "penalty_rest_penetration_frac": 0.00981,
    "lcp_max_penetration_frac": 0.056743325071,
    "lcp_rest_penetration_frac": 0.0,
    "dt_crit_penalty": 0.02,
    "penalty_stable_dt": 0.008,   # largest dt on the (1,2,4,8,16,32)x grid penalty survives
    "lcp_stable_dt": 0.064,       # lcp survives the whole grid
    "penalty_blowup_factor_min": 5.0,   # penalty energy_excess at dt=0.03 (>dt_crit) is ~1409x
}
PEN_TOL = 1.0e-6        # cpu determinism is exact; tol only guards float formatting


def exec_contact_prefix():
    """Exec contact.py up to the report region, in an isolated namespace. Returns
    the populated globals dict (contact.py's own scenes + solvers + metrics)."""
    src = CONTACT_PY.read_text()
    prefix = src[: src.index(CUT_MARKER)]

    scratch = Path(tempfile.mkdtemp(prefix="ch3.5-viz-"))
    # argparse runs DURING exec — pin the default config on cpu, no rerun path.
    old_argv = sys.argv
    sys.argv = [str(CONTACT_PY), "--seed", str(SEED), "--scene", "drop",
                "--dt", str(REF_DT), "--steps", str(REF_STEPS),
                "--no-rerun", "--device", "cpu", "--out", str(scratch)]
    ns: dict = {"__file__": str(CONTACT_PY), "__name__": "contact_toy_vizgen"}
    try:
        exec(compile(prefix, str(CONTACT_PY), "exec"), ns)  # noqa: S102 — contact.py is our own trusted source
    finally:
        sys.argv = old_argv
    return ns


def main() -> int:
    ns = exec_contact_prefix()
    build_scene = ns["build_scene"]
    simulate = ns["simulate"]
    semi_implicit_step = ns["semi_implicit_step"]
    penalty_force = ns["penalty_force"]
    lcp_step = ns["lcp_step"]
    contact_quality = ns["contact_quality"]

    # Rebuild the drop scene with a FRESH seed-0 rng, exactly as contact.py's report
    # region does (one rng.uniform draw for the drop-height jitter). This reproduces
    # the reference run's initial condition bit-for-bit.
    rng = np.random.default_rng(SEED)
    scene = build_scene("drop", rng, -1.0, -1.0)   # restitution/damping overrides off -> scene defaults
    radius0 = float(scene["radius"][0])
    mass0 = float(scene["m"][0, 0])

    def run_penalty(dt: float, steps: int):
        def force(q, v):
            return penalty_force(q, v, scene["m"], scene["radius"], STIFFNESS, scene["damping"])
        return simulate(lambda q, v: semi_implicit_step(q, v, scene["m"], force, dt), scene, dt, steps)

    def run_lcp(dt: float, steps: int):
        def step(q, v):
            return lcp_step(q, v, scene["m"], scene["radius"], scene["restitution"], BAUMGARTE, dt, ITERS)
        return simulate(step, scene, dt, steps)

    with np.errstate(all="ignore"):   # a diverging penalty spring overflows on purpose; measure it, don't crash
        # ------------------------------------------------ the reference drop (dt 0.002)
        p_h, p_pen, p_e = run_penalty(REF_DT, REF_STEPS)
        l_h, l_pen, l_e = run_lcp(REF_DT, REF_STEPS)
        pq = contact_quality(p_pen, p_e, radius0)
        lq = contact_quality(l_pen, l_e, radius0)

        # ------------------------------------------------ the dt-cliff sweep
        dt_crit = 2.0 * np.sqrt(mass0 / STIFFNESS)
        sweep = {"penalty": [], "lcp": []}
        for dt in DT_SWEEP:
            for name, runner in (("penalty", run_penalty), ("lcp", run_lcp)):
                _, pen, e = runner(dt, SWEEP_STEPS)
                q = contact_quality(pen, e, radius0)
                finite = not q["blew_up"]
                excess = q["energy_excess"]
                stable = finite and excess < STABLE_ENERGY_MULT
                sweep[name].append({
                    "dt": dt,
                    "max_pen_frac": q["max_penetration_frac"],
                    "energy_excess": excess,
                    "stable": bool(stable),
                })

        # ------------------------------------------------ stability probe (contact.py's grid)
        # Reproduce contact.py's OWN largest-stable-dt measurement so the gate matches
        # meta's penalty_stable_dt / lcp_stable_dt exactly (the coarse 1,2,4,8,16,32 grid).
        stable_dt = {}
        for name, runner in (("penalty", run_penalty), ("lcp", run_lcp)):
            biggest = 0.0
            for mult in (1, 2, 4, 8, 16, 32):
                _, _, e = runner(REF_DT * mult, REF_STEPS)
                if np.isfinite(e).all() and np.max(e - e[0]) / (abs(e[0]) or 1.0) < STABLE_ENERGY_MULT:
                    biggest = REF_DT * mult
            stable_dt[name] = biggest

    # ------------------------------------------------------------------ honesty gate
    pen_excess_at_003 = next(r["energy_excess"] for r in sweep["penalty"] if abs(r["dt"] - 0.03) < 1e-9)
    print("regenerated contact toy [seed 0, cpu, default config] vs meta.yaml:")
    print(f"  penalty  max_pen/r {pq['max_penetration_frac']:.12f}  rest_pen/r {pq['rest_penetration_frac']:.6f}"
          f"   (meta 0.251329183485 / 0.00981)")
    print(f"  lcp      max_pen/r {lq['max_penetration_frac']:.12f}  rest_pen/r {lq['rest_penetration_frac']:.6f}"
          f"   (meta 0.056743325071 / 0.0)")
    print(f"  dt_crit {dt_crit:.4f} (meta 0.02) · penalty_stable_dt {stable_dt['penalty']:.4g} (meta 0.008) · "
          f"lcp_stable_dt {stable_dt['lcp']:.4g} (meta 0.064)")
    print(f"  penalty energy_excess @ dt=0.03 = {pen_excess_at_003:.1f}x  (meta >=5x, ~1409x)")
    print("  dt cliff (penalty energy_excess / stable?):")
    for r in sweep["penalty"]:
        print(f"    dt={r['dt']:.3f} -> excess {r['energy_excess']:>10.4g}  {'stable' if r['stable'] else 'EXPLODES'}")

    fail = []

    def check(label, got, want, tol):
        if abs(got - want) > tol:
            fail.append(f"{label} {got:.12g} != {want} (tol {tol})")

    check("penalty max_pen", pq["max_penetration_frac"], META["penalty_max_penetration_frac"], PEN_TOL)
    check("penalty rest_pen", pq["rest_penetration_frac"], META["penalty_rest_penetration_frac"], 1e-4)
    check("lcp max_pen", lq["max_penetration_frac"], META["lcp_max_penetration_frac"], PEN_TOL)
    check("lcp rest_pen", lq["rest_penetration_frac"], META["lcp_rest_penetration_frac"], 1e-9)
    check("dt_crit", float(dt_crit), META["dt_crit_penalty"], 1e-9)
    check("penalty_stable_dt", stable_dt["penalty"], META["penalty_stable_dt"], 1e-9)
    check("lcp_stable_dt", stable_dt["lcp"], META["lcp_stable_dt"], 1e-9)
    if pen_excess_at_003 < META["penalty_blowup_factor_min"]:
        fail.append(f"penalty blowup @0.03 {pen_excess_at_003:.3g} < {META['penalty_blowup_factor_min']}")
    # the ORDERING that must survive any seed (contact.py's headline)
    if not (lq["max_penetration_frac"] < pq["max_penetration_frac"]
            and lq["rest_penetration_frac"] <= pq["rest_penetration_frac"] + 1e-9):
        fail.append("ordering broken: lcp does not hold shallower than penalty")

    if fail:
        print("\nSTOP — regenerated toy does NOT match meta.yaml:")
        for f in fail:
            print("  x " + f)
        return 1

    # ------------------------------------------------------------------ subsample + dump
    idx = list(range(0, WIN_STEPS + 1, STRIDE))    # the replay window frame indices

    def r4(x: float) -> float:
        return round(float(x), 4)

    def traj(heights: np.ndarray, pens: np.ndarray) -> dict:
        # per-frame ball-CENTER height and penetration-as-fraction-of-radius
        return {
            "height": [r4(heights[t, 0]) for t in idx],
            "pen_frac": [r4(pens[t] / radius0) for t in idx],
        }

    times = [r4(t * REF_DT) for t in idx]

    def sweep_pack(rows: list) -> list:
        out = []
        for r in rows:
            # energy_excess can be inf (a run that overflowed); JSON has no inf, so
            # flag it as blown-up with a null value the toy renders as "off the chart".
            ex = r["energy_excess"]
            finite = np.isfinite(ex)
            out.append({
                "dt": r["dt"],
                "max_pen_frac": r4(r["max_pen_frac"]) if np.isfinite(r["max_pen_frac"]) else None,
                "energy_excess": round(float(ex), 4) if finite else None,
                "stable": r["stable"],
            })
        return out

    data = {
        "provenance": {
            "source": "curriculum/phase3_advanced/ch3.5_contact/contact.py",
            "generator": "site/scripts/vizdata/ch3.5_contact.py",
            "seed": SEED,
            "device": "cpu",
            "config": f"drop scene, stiffness {STIFFNESS:g}, damping {scene['damping']:g}, "
                      f"baumgarte {BAUMGARTE}, iters {ITERS}, dt {REF_DT}, {REF_STEPS} steps",
            "stack": "numpy 2.4.6 (pure numpy, no torch/mujoco in the dynamics)",
            "note": "Real trajectories from contact.py's own penalty + lcp solvers; matches "
                    "meta.yaml reference_run. RECORDED, replayed in the browser — the engine is "
                    "~400 lines of numpy, not a live WASM sim. Frictionless normal contact only; "
                    "the lcp solve is fixed-iteration projected Gauss-Seidel, not a true LCP pivot.",
        },
        "radius": radius0,
        "reference": {
            "dt": REF_DT,
            "steps": REF_STEPS,
            "sim_time_s": round(REF_STEPS * REF_DT, 3),
            "stiffness": STIFFNESS,
            "damping": float(scene["damping"]),
        },
        # ---- the drop replay window (panels 1 + 3) ----
        "drop": {
            "t": times,
            "window_s": round(WIN_STEPS * REF_DT, 3),
            "penalty": {
                **traj(p_h, p_pen),
                "max_pen_frac": round(pq["max_penetration_frac"], 12),
                "rest_pen_frac": round(pq["rest_penetration_frac"], 6),
            },
            "lcp": {
                **traj(l_h, l_pen),
                "max_pen_frac": round(lq["max_penetration_frac"], 12),
                "rest_pen_frac": round(lq["rest_penetration_frac"], 6),
            },
        },
        # ---- the dt cliff (panel 2) ----
        "dt_sweep": {
            "dt": DT_SWEEP,
            "dt_crit": round(float(dt_crit), 6),
            "steps": SWEEP_STEPS,
            "default_dt": REF_DT,       # the slider boots here (both stable — the calm before the cliff)
            "penalty": sweep_pack(sweep["penalty"]),
            "lcp": sweep_pack(sweep["lcp"]),
        },
        "stability": {
            "penalty_stable_dt": stable_dt["penalty"],
            "lcp_stable_dt": stable_dt["lcp"],
            "dt_crit_penalty": round(float(dt_crit), 6),
        },
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB, {len(idx)} replay frames, {len(DT_SWEEP)} dt stops)")
    print("OK — matches meta.yaml; penalty sinks+rings+explodes past dt_crit, lcp holds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
