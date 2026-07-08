#!/usr/bin/env python3
"""Regenerate the ch3.4 constraint / double-pendulum concept-toy vizdata, seed 0.

The site's DoublePendulumToy island renders REAL, pre-computed trajectories — never
an invented swing. This generator REUSES constraints.py's OWN pieces (build_system,
make_force with the J M^-1 J^T Lagrange solve + Baumgarte feedback, the integrator
registry, constraint_violation, energy) so the browser animation is bit-faithful to
the chapter artifact, then dumps a small JSON the island loads.

Why we exec a PREFIX of constraints.py instead of `import constraints`
----------------------------------------------------------------------
constraints.py is a loose script (no `if __name__ == "__main__"` guard): importing
it runs the WHOLE pipeline — argparse, the none-vs-baumgarte comparison, metrics
write, rerun save. We must NOT modify constraints.py (it is LOC-capped and the code
IS the product). So we read its source and exec only the prefix UP TO the
`# --- region: report ---` line — i.e. setup + systems + constraints + integrators +
simulate helpers — in a throwaway namespace, with `--no-rerun` and a scratch --out.
That hands us constraints.py's OWN build_system / make_force / INTEGRATORS /
constraint_violation / energy / simulate, at the DEFAULT config (double pendulum,
semi_implicit, dt 0.005, 4000 steps = 20 s sim), with zero edits to the file.

What we record beyond constraints.py's `simulate`
-------------------------------------------------
constraints.py's `simulate` returns only the TIP path (the last bob). To ANIMATE the
double pendulum we need BOTH bob positions per step, so we run a tiny local rollout
using the SAME (step_fn, force, dt) — identical math — and record the full q. As an
honesty cross-check we assert our rollout's tip column is BITWISE identical to
constraints.py's own `simulate` tip. Every number below is verified against
meta.yaml's reference_run; the script STOPS if it drifts.

The three stories the JSON carries (the demo `constraint_drift`):
  1. runs.A — the double pendulum (seed 0, Baumgarte): the chaotic swing + traced path.
  2. runs.A vs runs.B (seed 0 vs seed 1): almost-identical ICs diverge (chaos), yet
     each seed replays BITWISE (determinism) — the headline. divergence curve + hash.
  3. drift — link-length violation vs time, naive (none) vs Baumgarte, on the SAME
     double pendulum: none stretches without bound, Baumgarte holds it near zero.

    Run:  .venv/bin/python site/scripts/vizdata/ch3.4_constraints.py
    Out:  curriculum/phase3_advanced/ch3.4_constraints/demo/vizdata.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CONSTRAINTS_PY = REPO / "curriculum" / "phase3_advanced" / "ch3.4_constraints" / "constraints.py"
OUT_JSON = REPO / "curriculum" / "phase3_advanced" / "ch3.4_constraints" / "demo" / "vizdata.json"

SEED_A = 0          # the hero double pendulum
SEED_B = 1          # the near-neighbour (a ~0.05-rad-range IC draw): chaos partner
SYSTEM = "double"
INTEGRATOR = "semi_implicit"   # constraints.py's default (the reference_run integrator)
DT = 0.005
STEPS = 4000                   # 20 s of sim time — the full reference_run
BAUMGARTE = 20.0               # the stabilization gain the reference_run uses
CUT_MARKER = "# --- region: report ---"

# animation / curve subsampling (small committed TEXT, no binary)
FRAMES = 150     # animation frames kept per pendulum (both bobs, x-y)
PATH = 260       # points kept for the traced tip path
CURVE = 160      # points kept for the divergence + drift curves

# meta.yaml reference_run (seed 0, cpu, numpy 2.4.6) — the honesty gate. If the
# regenerated toy drifts from these MEASURED numbers, STOP. Pure-numpy CPU, so the
# match is essentially exact; the tol only guards last-bit float noise.
META = {
    "none_max_violation": 0.384014620396,
    "none_energy_rel_max": 0.427091167688,
    "baumgarte_max_violation": 0.023470285993,
    "baumgarte_final_violation": 0.001436710083,
    "baumgarte_energy_rel_max": 0.32484945881,
    "tip_divergence_final": 1.117,   # seed 0 vs seed 1, baumgarte, over 20 s (meta ~1.12)
    "tip_divergence_min": 0.5,       # exercise_checks.ex3 robust floor
}
ABS_TOL = 1.0e-6
DIV_TOL = 0.03


def exec_constraints_prefix(seed: int) -> dict:
    """Exec constraints.py up to the report region, in an isolated namespace.

    argparse runs DURING exec, so we pin the default config on cpu with no rerun and
    a scratch --out. Returns the populated globals (constraints.py's own functions)."""
    src = CONSTRAINTS_PY.read_text()
    prefix = src[: src.index(CUT_MARKER)]
    scratch = Path(tempfile.mkdtemp(prefix="ch3.4-viz-"))
    old_argv = sys.argv
    sys.argv = [
        str(CONSTRAINTS_PY), "--seed", str(seed), "--system", SYSTEM,
        "--integrator", INTEGRATOR, "--no-rerun", "--device", "cpu",
        "--out", str(scratch),
    ]
    ns: dict = {"__file__": str(CONSTRAINTS_PY), "__name__": "ch3.4_constraints_vizgen"}
    try:
        exec(compile(prefix, str(CONSTRAINTS_PY), "exec"), ns)  # noqa: S102 — our own trusted source
    finally:
        sys.argv = old_argv
    return ns


def rollout(ns: dict, system: dict, baumgarte: float):
    """Integrate the chain, recording the FULL q (both bobs) per step.

    Uses constraints.py's OWN make_force / INTEGRATORS / constraint_violation /
    energy — identical math to its `simulate`, just recording every bob instead of
    only the tip. Returns (Q[steps+1, N, 3], violations, energies)."""
    make_force = ns["make_force"]
    step_fn = ns["INTEGRATORS"][INTEGRATOR]
    constraint_violation = ns["constraint_violation"]
    energy = ns["energy"]
    force = make_force(system, baumgarte)

    q = system["q0"].astype(float).copy()
    v = system["v0"].astype(float).copy()
    m = system["m"]
    n = q.shape[0]
    Q = np.empty((STEPS + 1, n, 3))
    viol = np.empty(STEPS + 1)
    en = np.empty(STEPS + 1)
    Q[0], viol[0], en[0] = q, constraint_violation(system, q), energy(system, q, v)
    for i in range(STEPS):
        q, v = step_fn(q, v, m, force, DT)
        Q[i + 1], viol[i + 1], en[i + 1] = q, constraint_violation(system, q), energy(system, q, v)
    return Q, viol, en


def build_and_roll(seed: int, baumgarte: float):
    """Reproduce `constraints.py --seed S`: fresh PCG64(seed) -> build_system -> roll."""
    ns = exec_constraints_prefix(seed)
    rng = np.random.default_rng(seed)          # exactly constraints.py's single RNG source
    system = ns["build_system"](SYSTEM, rng)   # consumes the one tilt draw
    tilt = float(np.arctan2(-system["q0"][0][1], system["q0"][0][0]))  # recover the launch tilt
    Q, viol, en = rollout(ns, system, baumgarte)
    return ns, system, tilt, Q, viol, en


def subsample(n_plus1: int, k: int) -> np.ndarray:
    """Evenly spaced indices over [0, n], always including both endpoints."""
    return np.unique(np.linspace(0, n_plus1 - 1, min(k, n_plus1)).round().astype(int))


def main() -> int:
    escale = 2.0 * 9.81 * 2.0  # sum(m)*|g|*sum(L) — constraints.py's energy-drift scale

    # ---- run A: seed 0, Baumgarte (the hero pendulum: holds together, chaotic swing)
    nsA, sysA, tiltA, QA, violA_b, enA_b = build_and_roll(SEED_A, BAUMGARTE)
    # ---- run A naive: seed 0, none (the drift story, SAME pendulum)
    _, _, _, QA_n, violA_none, enA_none = build_and_roll(SEED_A, 0.0)
    # ---- run B: seed 1, Baumgarte (the near-neighbour: chaos partner)
    _, _, tiltB, QB, _, _ = build_and_roll(SEED_B, BAUMGARTE)

    # ---- honesty cross-check: our rollout tip == constraints.py's own simulate tip
    tipA_ref, _viol_ref, _en_ref = nsA["simulate"](
        nsA["INTEGRATORS"][INTEGRATOR], sysA, nsA["make_force"](sysA, BAUMGARTE), DT, STEPS
    )
    if not np.array_equal(QA[:, -1], tipA_ref):
        print("STOP — local rollout tip diverges from constraints.py simulate tip")
        return 1

    # ---- determinism: re-run seed 0 Baumgarte and demand a BITWISE-identical trajectory
    _, _, _, QA2, _, _ = build_and_roll(SEED_A, BAUMGARTE)
    determinism_ok = np.array_equal(QA, QA2)
    traj_hash = hashlib.sha256(QA.tobytes()).hexdigest()

    # ---- the measured headline numbers
    none_max = float(np.max(violA_none))
    none_erel = float(np.max(np.abs(enA_none - enA_none[0])) / escale)
    baum_max = float(np.max(violA_b))
    baum_final = float(violA_b[-1])
    baum_erel = float(np.max(np.abs(enA_b - enA_b[0])) / escale)

    tipA, tipB = QA[:, -1], QB[:, -1]
    div = np.linalg.norm(tipA - tipB, axis=1)
    div_final = float(div[-1])
    init_delta = abs(tiltA - tiltB)

    # ------------------------------------------------------------------ honesty gate
    print("regenerated toy [seed 0/1, cpu, default config] vs meta.yaml reference_run:")
    print(f"  none        : max|len err| {none_max:.12f}   (meta 0.384014620396)")
    print(f"                energy_rel_max {none_erel:.12f} (meta 0.427091167688)")
    print(f"  baumgarte   : max|len err| {baum_max:.12f}   (meta 0.023470285993)")
    print(f"                final|len err| {baum_final:.12f}(meta 0.001436710083)")
    print(f"                energy_rel_max {baum_erel:.12f} (meta 0.32484945881)")
    print(f"  chaos       : seed0 tilt {tiltA:+.6f} rad, seed1 tilt {tiltB:+.6f} rad, "
          f"delta {init_delta:.6f} rad")
    print(f"                tip divergence over 20 s {div_final:.6f}  (meta ~1.12, floor 0.5)")
    print(f"  determinism : seed-0 rerun bitwise identical = {determinism_ok}  "
          f"(traj sha256 {traj_hash[:12]})")

    checks = [
        ("none_max_violation", none_max, META["none_max_violation"], ABS_TOL),
        ("none_energy_rel_max", none_erel, META["none_energy_rel_max"], ABS_TOL),
        ("baumgarte_max_violation", baum_max, META["baumgarte_max_violation"], ABS_TOL),
        ("baumgarte_final_violation", baum_final, META["baumgarte_final_violation"], ABS_TOL),
        ("baumgarte_energy_rel_max", baum_erel, META["baumgarte_energy_rel_max"], ABS_TOL),
        ("tip_divergence_final", div_final, META["tip_divergence_final"], DIV_TOL),
    ]
    fail = [f"{n} {v:.9f} != {ref} (tol {tol})" for n, v, ref, tol in checks if abs(v - ref) > tol]
    if div_final < META["tip_divergence_min"]:
        fail.append(f"tip divergence {div_final:.4f} below floor {META['tip_divergence_min']}")
    if not determinism_ok:
        fail.append("seed-0 rerun NOT bitwise identical — determinism claim broken")
    if not (none_max > 3.0 * baum_max):
        fail.append(f"drift ordering broken: none {none_max:.4f} !>> baumgarte {baum_max:.4f}")
    if fail:
        print("\nSTOP — regenerated toy does NOT match meta.yaml:")
        for f in fail:
            print("  x " + f)
        return 1

    # ------------------------------------------------------------------ subsample + pack
    fidx = subsample(STEPS + 1, FRAMES)     # shared animation indices (A and B time-synced)
    pidx = subsample(STEPS + 1, PATH)       # traced-path indices
    cidx = subsample(STEPS + 1, CURVE)      # curve indices

    def frames(Q: np.ndarray) -> list:
        # [bob0_x, bob0_y, bob1_x, bob1_y] per frame (x-y plane; z is ~0)
        return [[round(float(Q[i, 0, 0]), 3), round(float(Q[i, 0, 1]), 3),
                 round(float(Q[i, 1, 0]), 3), round(float(Q[i, 1, 1]), 3)] for i in fidx]

    def path_xy(Q: np.ndarray) -> list:
        return [[round(float(Q[i, 1, 0]), 3), round(float(Q[i, 1, 1]), 3)] for i in pidx]

    # arena bounds across both pendulums (pivot at origin), with a small margin
    allxy = np.concatenate([QA[:, :, :2].reshape(-1, 2), QB[:, :, :2].reshape(-1, 2)])
    xmin, ymin = allxy.min(axis=0)
    xmax, ymax = allxy.max(axis=0)
    mx, my = 0.12 * (xmax - xmin), 0.12 * (ymax - ymin)
    bounds = {
        "xmin": round(float(xmin - mx), 3), "xmax": round(float(xmax + mx), 3),
        "ymin": round(float(ymin - my), 3), "ymax": round(float(ymax + my), 3),
    }

    times = [round(float(i) * DT, 3) for i in cidx]
    data = {
        "provenance": {
            "source": "curriculum/phase3_advanced/ch3.4_constraints/constraints.py",
            "generator": "site/scripts/vizdata/ch3.4_constraints.py",
            "system": SYSTEM,
            "integrator": INTEGRATOR,
            "dt": DT,
            "steps": STEPS,
            "baumgarte_omega": BAUMGARTE,
            "device": "cpu",
            "stack": "numpy 2.4.6",
            "note": "Real double-pendulum trajectories from constraints.py's own "
                    "make_force (J M^-1 J^T Lagrange solve + Baumgarte) + integrator; "
                    "matches meta.yaml reference_run. The chain is CHAOTIC yet BITWISE "
                    "deterministic per seed (pure numpy).",
        },
        "dt": DT,
        "steps": STEPS,
        "sim_seconds": round(STEPS * DT, 3),
        "link_length": 1.0,
        "pivot": [0.0, 0.0],
        "bounds": bounds,
        "frame_count": int(len(fidx)),
        # ---- panels 1 & 2: the two pendulums (both bobs per frame), Baumgarte-stabilised
        "runs": {
            "A": {  # the hero: seed 0 (panel 1 animation + panel 2 left)
                "seed": SEED_A,
                "tilt_rad": round(tiltA, 6),
                "frames": frames(QA),
                "path": path_xy(QA),
                "tip_final": [round(float(tipA[-1, 0]), 4), round(float(tipA[-1, 1]), 4)],
            },
            "B": {  # the near-neighbour: seed 1 (panel 2 right)
                "seed": SEED_B,
                "tilt_rad": round(tiltB, 6),
                "frames": frames(QB),
                "path": path_xy(QB),
                "tip_final": [round(float(tipB[-1, 0]), 4), round(float(tipB[-1, 1]), 4)],
            },
        },
        # ---- panel 2 headline: chaos (divergence) yet determinism (bitwise replay)
        "chaos": {
            "init_delta_rad": round(init_delta, 6),
            "divergence": {
                "t": times,
                "dist": [round(float(div[i]), 4) for i in cidx],
                "final": round(div_final, 4),
                "floor": META["tip_divergence_min"],
            },
            "determinism": {
                "bitwise_reproducible": bool(determinism_ok),
                "traj_sha256_12": traj_hash[:12],
                "note": "Two seed-0 runs produce a byte-identical trajectory (pure "
                        "numpy on CPU). Chaos is not nondeterminism: nearby seeds "
                        "diverge, but each seed replays exactly.",
            },
        },
        # ---- panel 3: constraint drift, naive vs Baumgarte, on the SAME pendulum
        "drift": {
            "baumgarte_omega": BAUMGARTE,
            "t": times,
            "none": [round(float(violA_none[i]), 5) for i in cidx],
            "baumgarte": [round(float(violA_b[i]), 5) for i in cidx],
            "none_max": round(none_max, 6),
            "baumgarte_max": round(baum_max, 6),
            "baumgarte_final": round(baum_final, 6),
            "none_energy_rel_max": round(none_erel, 6),
            "baumgarte_energy_rel_max": round(baum_erel, 6),
            "drift_ratio": round(none_max / baum_max, 2),
        },
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB, {len(fidx)} frames)")
    print("OK — matches meta.yaml; chaotic-yet-deterministic (bitwise) + drift ordering holds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
