#!/usr/bin/env python3
"""Regenerate the ch3.3 physics-engine concept-toy vizdata from engine.py, seed 0.

The site's EngineDriftToy island renders REAL integration curves — never invented
shapes. This generator REUSES engine.py's OWN pieces (build_system, the three
integrator steppers euler/semi_implicit/rk4, and energy_drift) so the browser
panels are bit-faithful to the chapter artifact, then dumps a small JSON the
island loads.

Why we exec a PREFIX of engine.py instead of `import engine`
------------------------------------------------------------
engine.py is a loose script (no `if __name__ == "__main__"` guard): importing it
runs the WHOLE report — argparse, banner, a full simulate sweep, metrics.json,
and a rerun recording. We must NOT modify engine.py (it is LOC-capped). So we
read its source and exec only the prefix up to the `# --- region: report ---`
line — i.e. setup + systems + integrators + simulate + energy_drift — in a
throwaway namespace. That hands us engine.py's OWN `build_system`, `INTEGRATORS`
(euler_step / semi_implicit_step / rk4_step), `simulate`, `energy_drift`, and
`COLORS`, at the DEFAULT config (seed 0, cpu, pure numpy), with zero edits.

What we record beyond engine.simulate()
---------------------------------------
engine.simulate() returns (trajectory, energies) but throws away the velocity —
which the spring PHASE PORTRAIT (position vs velocity) needs. So we run a loop
that REUSES engine.py's exact step functions and energy law and additionally
keeps v. We then assert our (trajectory, energies) is BITWISE-equal to
engine.simulate()'s, so the extra velocity capture cannot have perturbed the
faithful physics, and verify the drift signatures against meta.yaml. The script
STOPS if anything drifts.

    Run:  .venv/bin/python site/scripts/vizdata/ch3.3_engine.py
    Out:  curriculum/phase3_advanced/ch3.3_engine/demo/vizdata.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ENGINE_PY = REPO / "curriculum" / "phase3_advanced" / "ch3.3_engine" / "engine.py"
OUT_JSON = REPO / "curriculum" / "phase3_advanced" / "ch3.3_engine" / "demo" / "vizdata.json"

SEED = 0
CUT_MARKER = "# --- region: report ---"

SYSTEMS = ["orbit", "spring", "freefall"]
INTEG = ["euler", "semi_implicit", "rk4"]
# The dt slider stops. Default 0.01 reproduces meta's reference_run exactly
# (2000 steps * 0.01 = 20 s). Every stop integrates the SAME 20 s of sim time,
# so the curves are comparable and the Euler runaway worsening with dt is honest.
DTS = [0.005, 0.01, 0.02, 0.04]
DEFAULT_DT = 0.01
SIM_TIME = 20.0            # seconds of sim time held constant across dt stops
ENERGY_SAMPLES = 90       # points kept per energy-vs-time curve (small text)
PHASE_SAMPLES = 260       # points kept per phase-space trace

# Which two coordinates the phase panel plots for each bounded system. Freefall
# has NO closed portrait (the motion never returns) — that IS the caveat, so it
# carries no phase trace; the toy shows the honest note instead.
PHASE = {
    "orbit": {"kind": "xy", "i": 0, "j": 1, "source_i": "q", "source_j": "q",
              "xlabel": "position x", "ylabel": "position y"},
    "spring": {"kind": "xv", "i": 0, "j": 0, "source_i": "q", "source_j": "v",
               "xlabel": "position x", "ylabel": "velocity vx"},
}

# meta.yaml reference_run (seed 0, cpu, numpy 2.4.6) — the honesty gate. Pure
# numpy is bitwise deterministic on CPU, so these are exact; tol only guards
# float print noise. If the regenerated engine drifts from these, STOP.
META_ORBIT = {
    "e0": -0.486210038749,
    "euler_rel_final": 0.221252393919,
    "euler_rel_max": 0.221252393919,
    "semi_rel_final": 0.000098263658,
    "semi_rel_max": 0.000320278117,
    "rk4_rel_final": -4.5070e-11,
    "rk4_rel_max": 5.1306e-11,
}
# freefall is the honest counter-example: energy drifts under BOTH Euler (+) and
# semi-implicit (-) because the motion never returns — the symplectic advantage
# needs a bounded/oscillatory system. meta records +/-9.3% both.
META_FREEFALL = {"euler_rel_final": 0.093415, "semi_rel_final": -0.093415}
ABS_TOL = 1e-8
RK4_NEGLIGIBLE = 1e-6


def exec_engine_prefix() -> dict:
    """Exec engine.py up to the report region, in an isolated namespace. Returns
    the populated globals dict (engine.py's own steppers + functions)."""
    src = ENGINE_PY.read_text()
    cut = src.index(CUT_MARKER)
    prefix = src[:cut]

    scratch = Path(tempfile.mkdtemp(prefix="ch3.3-viz-"))
    # argparse runs DURING exec — pin the default config on cpu, no rerun path.
    old_argv = sys.argv
    sys.argv = [str(ENGINE_PY), "--seed", str(SEED), "--no-rerun",
                "--device", "cpu", "--out", str(scratch)]
    ns: dict = {"__file__": str(ENGINE_PY), "__name__": "engine_toy_vizgen"}
    try:
        exec(compile(prefix, str(ENGINE_PY), "exec"), ns)  # noqa: S102 — our own trusted source
    finally:
        sys.argv = old_argv
    return ns


def main() -> int:
    ns = exec_engine_prefix()
    build_system = ns["build_system"]
    integrators = ns["INTEGRATORS"]
    simulate = ns["simulate"]
    energy_drift = ns["energy_drift"]
    colors = ns["COLORS"]

    def run(step_fn, system: dict, dt: float, steps: int):
        """Mirror engine.simulate() EXACTLY, additionally keeping velocity v.
        Reuses engine.py's own step_fn + energy law, so the physics is faithful."""
        q = system["q0"].astype(float).copy()
        v = system["v0"].astype(float).copy()
        m, force, energy = system["m"], system["force"], system["energy"]
        traj = np.empty((steps + 1, 3))
        vel = np.empty((steps + 1, 3))
        ener = np.empty(steps + 1)
        traj[0], vel[0], ener[0] = q, v, energy(q, v)
        for i in range(steps):
            q, v = step_fn(q, v, m, force, dt)
            traj[i + 1], vel[i + 1], ener[i + 1] = q, v, energy(q, v)
        return traj, vel, ener

    # ---------------------------------------------------------------- run the sweep
    # For each system rebuild with a FRESH rng(0) exactly as engine.py does
    # (one fresh default_rng(args.seed) per run), so the seed-0 initial condition
    # matches the chapter artifact bit-for-bit.
    result: dict[str, dict] = {}
    for sysname in SYSTEMS:
        system = build_system(sysname, np.random.default_rng(SEED))
        entry: dict = {"e0": None, "energy": {}, "phase": None}

        for dt in DTS:
            steps = int(round(SIM_TIME / dt))
            per_dt: dict = {"drift": {}}
            t_full = np.arange(steps + 1) * dt
            si = np.linspace(0, steps, min(ENERGY_SAMPLES, steps + 1)).round().astype(int)
            per_dt["t"] = [round(float(x), 4) for x in t_full[si]]
            for name in INTEG:
                traj, vel, ener = run(integrators[name], system, dt, steps)
                # Faithfulness guard: our loop must equal engine.simulate() bitwise.
                e_traj, e_ener = simulate(integrators[name], system, dt, steps)
                assert np.array_equal(traj, e_traj) and np.array_equal(ener, e_ener), \
                    f"loop diverged from engine.simulate() on {sysname}/{name}/dt={dt}"
                d = energy_drift(ener)          # engine.py's OWN drift metric
                scale = abs(d["e0"]) if d["e0"] != 0.0 else 1.0
                rel = (ener - d["e0"]) / scale
                per_dt[name] = [round(float(x), 6) for x in rel[si]]
                per_dt["drift"][name] = {
                    "rel_final": round(float(d["rel_final"]), 12),
                    "rel_max": round(float(d["rel_max"]), 12),
                }
                if dt == DEFAULT_DT:
                    entry["e0"] = round(float(d["e0"]), 9)
                    if sysname in PHASE:
                        spec = PHASE[sysname]
                        src_i = traj if spec["source_i"] == "q" else vel
                        src_j = traj if spec["source_j"] == "q" else vel
                        pi = np.linspace(0, steps, min(PHASE_SAMPLES, steps + 1)).round().astype(int)
                        xs = src_i[pi, spec["i"]]
                        ys = src_j[pi, spec["j"]]
                        entry.setdefault("_phase_pts", {})[name] = (xs, ys)
            entry["energy"][f"{dt:g}"] = per_dt
        result[sysname] = entry

    # ------------------------------------------------------------------ honesty gate
    def drift(sysname, dt, name, key):
        return result[sysname]["energy"][f"{dt:g}"]["drift"][name][key]

    print("regenerated engine [seed 0, cpu] vs meta.yaml reference_run:")
    print(f"  orbit e0        : {result['orbit']['e0']:.9f}   (meta {META_ORBIT['e0']})")
    for name in INTEG:
        rf = drift("orbit", DEFAULT_DT, name, "rel_final")
        rm = drift("orbit", DEFAULT_DT, name, "rel_max")
        print(f"  orbit {name:<13}: rel_final {rf:>+.9e}  rel_max {rm:.9e}")
    print(f"  freefall euler  : rel_final {drift('freefall', DEFAULT_DT, 'euler', 'rel_final'):>+.6f}   (meta +0.093415)")
    print(f"  freefall semi   : rel_final {drift('freefall', DEFAULT_DT, 'semi_implicit', 'rel_final'):>+.6f}   (meta -0.093415)")

    fail = []
    if abs(result["orbit"]["e0"] - META_ORBIT["e0"]) > ABS_TOL:
        fail.append(f"orbit e0 {result['orbit']['e0']} != {META_ORBIT['e0']}")
    for name, mf, mm in [
        ("euler", META_ORBIT["euler_rel_final"], META_ORBIT["euler_rel_max"]),
        ("semi_implicit", META_ORBIT["semi_rel_final"], META_ORBIT["semi_rel_max"]),
    ]:
        if abs(drift("orbit", DEFAULT_DT, name, "rel_final") - mf) > ABS_TOL:
            fail.append(f"orbit {name} rel_final {drift('orbit', DEFAULT_DT, name, 'rel_final')} != {mf}")
        if abs(drift("orbit", DEFAULT_DT, name, "rel_max") - mm) > ABS_TOL:
            fail.append(f"orbit {name} rel_max {drift('orbit', DEFAULT_DT, name, 'rel_max')} != {mm}")
    # rk4: negligible drift + the honest one-signed creep (rel_final < 0 here)
    if abs(drift("orbit", DEFAULT_DT, "rk4", "rel_max")) > RK4_NEGLIGIBLE:
        fail.append(f"orbit rk4 rel_max {drift('orbit', DEFAULT_DT, 'rk4', 'rel_max')} not negligible")
    if drift("orbit", DEFAULT_DT, "rk4", "rel_final") >= 0:
        fail.append("orbit rk4 rel_final should be negative (one-signed creep, not bounded)")
    # ordering headline: euler >> semi(bounded) > rk4, on the bounded systems
    for sysname in ("orbit", "spring"):
        e = abs(drift(sysname, DEFAULT_DT, "euler", "rel_final"))
        s = abs(drift(sysname, DEFAULT_DT, "semi_implicit", "rel_max"))
        r = abs(drift(sysname, DEFAULT_DT, "rk4", "rel_max"))
        if not (e > s > r):
            fail.append(f"{sysname} ordering euler>>semi>rk4 broken: {e} {s} {r}")
    # freefall honest counter-example: opposite signs, comparable magnitude
    ff_e = drift("freefall", DEFAULT_DT, "euler", "rel_final")
    ff_s = drift("freefall", DEFAULT_DT, "semi_implicit", "rel_final")
    if not (ff_e > 0 > ff_s):
        fail.append(f"freefall should drift +Euler / -semi: {ff_e} {ff_s}")
    if abs(ff_e - META_FREEFALL["euler_rel_final"]) > 1e-4 or abs(ff_s - META_FREEFALL["semi_rel_final"]) > 1e-4:
        fail.append(f"freefall magnitude off meta +/-9.3%: {ff_e} {ff_s}")
    if fail:
        print("\nSTOP — regenerated engine does NOT match meta.yaml:")
        for f in fail:
            print("  x " + f)
        return 1

    # ------------------------------------------------------------------ pack phase
    # A symmetric world half-extent per system that comfortably holds the Euler
    # spiral-out, so the component maps every trace into one square viewBox.
    for sysname, entry in result.items():
        pts = entry.pop("_phase_pts", None)
        if pts is None:
            continue
        spec = PHASE[sysname]
        extent = 0.0
        for xs, ys in pts.values():
            extent = max(extent, float(np.abs(xs).max()), float(np.abs(ys).max()))
        entry["phase"] = {
            "kind": spec["kind"],
            "xlabel": spec["xlabel"],
            "ylabel": spec["ylabel"],
            "domain": round(extent * 1.06, 4),
            "traces": {
                name: [[round(float(x), 3), round(float(y), 3)] for x, y in zip(xs, ys)]
                for name, (xs, ys) in pts.items()
            },
        }

    data = {
        "provenance": {
            "source": "curriculum/phase3_advanced/ch3.3_engine/engine.py",
            "generator": "site/scripts/vizdata/ch3.3_engine.py",
            "seed": SEED,
            "device": "cpu",
            "config": "default (seed 0, pure numpy, 20 s sim time per dt stop)",
            "stack": "numpy 2.4.6",
            "note": "Real curves from engine.py's own build_system + integrator "
                    "steppers + energy_drift; matches meta.yaml reference_run. dt "
                    "is a SLIDER-only knob — the physics is identical, only the "
                    "step size changes, and every integrator's drift grows with it.",
        },
        "systems": SYSTEMS,
        "integrators": INTEG,
        "colors": {k: colors[k] for k in INTEG},   # engine.py COLORS (rgb 0-255)
        "dts": DTS,
        "default_dt": DEFAULT_DT,
        "default_system": "orbit",
        "sim_time": SIM_TIME,
        # freefall is the honest counter-example: no bounded portrait, both drift.
        "freefall_note": "Freefall never returns, so there is no closed orbit to "
                         "bound: energy drifts under BOTH Euler (+9.3%) and "
                         "semi-implicit (-9.3%). The symplectic advantage needs a "
                         "bounded, oscillatory system.",
        "data": result,
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB)")
    print("OK — matches meta.yaml; ordering euler >> semi(bounded) > rk4; freefall drifts both ways.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
