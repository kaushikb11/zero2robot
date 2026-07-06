#!/usr/bin/env python3
"""Regenerate the ch1.5 flow-matching concept-toy vizdata from flow.py, seed 0.

The site's FlowRingToy island renders REAL points — never invented shapes. This
generator REUSES flow.py's own toy pieces (the ring target, the regression net,
the VelocityNet, and the forward-Euler `ode_sample_loop`) so the browser scatter
is bit-faithful to the chapter artifact, then dumps a small JSON the island loads.

Why we exec a PREFIX of flow.py instead of `import flow`
--------------------------------------------------------
flow.py is a loose script (no `if __name__ == "__main__"` guard): importing it
runs the WHOLE pipeline — argparse, demo generation, LeRobot load, 300-epoch
policy training, eval, ONNX export. We must NOT modify flow.py (it is LOC-capped
at 438). So we read its source and exec only the prefix up to the `# Sample from
both` line — i.e. setup + core + the toy region's ring build and net training —
in a throwaway namespace. That gives us flow.py's OWN trained `toy` net,
`regress` net, `ode_sample_loop`, `ring_stats`, `interpolate`, and the `ring`
target, at the DEFAULT config (model_dim 128, seed 0, cpu), with zero edits to
the file and none of the heavy policy path.

Reproducing the step-efficiency sweep faithfully
------------------------------------------------
In flow.py the sampler's start noise `x = randn(shape)` is drawn from the shared
`gen` RNG right after the toy+regress training finishes — and `flow_steps` only
changes how many Euler steps integrate that same start noise (training never sees
it: the decoupling). So a fresh `flow.py --flow_steps K` run would enter the
`flow_samples` call with an IDENTICAL RNG state for every K. We reproduce that
exactly: snapshot the gen state S0 at the cut point, and for each K in the sweep
restore S0 before sampling. K=100 reproduces flow.py's default `flow_samples`;
the regression draw reproduces its default `reg_samples`. Verified against
meta.yaml's measured numbers below; the script STOPS if they drift.

    Run:  .venv/bin/python site/scripts/vizdata/ch1.5_flow.py
    Out:  curriculum/phase1_imitation/ch1.5_flow/demo/vizdata.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
FLOW_PY = REPO / "curriculum" / "phase1_imitation" / "ch1.5_flow" / "flow.py"
OUT_JSON = REPO / "curriculum" / "phase1_imitation" / "ch1.5_flow" / "demo" / "vizdata.json"

SEED = 0
STEPS = [2, 3, 5, 8, 100]   # the slider stops; parallels the meta.yaml sweep
SUBSAMPLE = 300             # points kept per scatter set (small committed text)
CUT_MARKER = "# Sample from both"

# meta.yaml reference_run (seed 0, cpu, torch 2.10.0 / numpy 2.4.6) — the honesty
# gate. If the regenerated toy drifts from these MEASURED numbers, STOP.
META = {
    "flow_modes": 8,
    "flow_mean_radius": 0.935944,
    "flow_ring_hit": 0.6685,
    "regress_modes": 0,
    "regress_mean_radius": 0.058228,
    "twostep_modes": 3,          # --break few_steps: 8/8 -> 3/8 at flow_steps=2
}
RADIUS_TOL = 0.03   # cpu determinism should make this exact; tol guards float noise
HIT_TOL = 0.02


def exec_flow_prefix():
    """Exec flow.py up to the sampling cut, in an isolated namespace. Returns the
    populated globals dict (flow.py's own trained nets + functions)."""
    src = FLOW_PY.read_text()
    cut = src.index(CUT_MARKER)
    # Prefix = setup + core + ring build + toy/regress training (stops before the
    # first sampling draw, so we can snapshot the RNG cleanly). We then append ONLY
    # flow.py's `ring_stats` def (it lives just past the cut and draws no RNG) so
    # the measurement is flow.py's own, verbatim.
    stats_def = src[src.index("def ring_stats"):src.index("flow_r, flow_hit")]
    prefix = src[:cut] + "\n" + stats_def

    scratch = Path(tempfile.mkdtemp(prefix="ch1.5-viz-"))
    # argparse runs DURING exec — pin the default config on cpu, no rerun/onnx path.
    old_argv = sys.argv
    sys.argv = [str(FLOW_PY), "--seed", str(SEED), "--no-rerun",
                "--device", "cpu", "--out", str(scratch)]
    ns: dict = {"__file__": str(FLOW_PY), "__name__": "flow_toy_vizgen"}
    try:
        exec(compile(prefix, str(FLOW_PY), "exec"), ns)  # noqa: S102 — flow.py is our own trusted source
    finally:
        sys.argv = old_argv
    return ns


def main() -> int:
    ns = exec_flow_prefix()
    torch = ns["torch"]
    ode_sample_loop = ns["ode_sample_loop"]
    ring_stats = ns["ring_stats"]
    toy = ns["toy"]
    regress = ns["regress"]
    randn = ns["randn"]
    gen = ns["gen"]
    N_TOY = ns["N_TOY"]
    ring = ns["ring"].detach().cpu().numpy()

    # Snapshot the RNG exactly where flow.py stands entering its `flow_samples`
    # call (right after toy + regress training). Every K restores from here, so
    # each reproduces a fresh `flow.py --flow_steps K` run's start noise.
    s0 = gen.get_state()

    def sample(steps: int) -> np.ndarray:
        gen.set_state(s0)
        return ode_sample_loop(toy, (N_TOY, 2), None, steps).cpu().numpy()

    # Regression samples: reproduce flow.py's default order — the flow_samples
    # call draws one start-noise randn, THEN reg_samples draws the next randn.
    gen.set_state(s0)
    _ = ode_sample_loop(toy, (N_TOY, 2), None, 100)      # consume the start-noise draw
    with torch.no_grad():
        reg_samples = regress(randn((N_TOY, 2))).cpu().numpy()
    reg_r, reg_hit, reg_modes = ring_stats(reg_samples)

    per_step = {}
    for k in STEPS:
        pts = sample(k)
        r, hit, modes = ring_stats(pts)
        per_step[k] = {"pts": pts, "mean_radius": r, "ring_hit": hit, "modes": modes}

    f100 = per_step[100]
    f2 = per_step[2]

    # ------------------------------------------------------------------ honesty gate
    print("regenerated toy [seed 0, cpu, default config] vs meta.yaml:")
    print(f"  flow @100 steps : modes {f100['modes']}/8  mean_radius {f100['mean_radius']:.6f}  "
          f"ring_hit {f100['ring_hit']:.4f}   (meta 8/8, 0.935944, 0.6685)")
    print(f"  flow @2   steps : modes {f2['modes']}/8  mean_radius {f2['mean_radius']:.6f}  "
          f"ring_hit {f2['ring_hit']:.4f}   (meta 3/8)")
    print(f"  regression      : modes {reg_modes}/8  mean_radius {reg_r:.6f}  "
          f"ring_hit {reg_hit:.4f}   (meta 0/8, 0.058228)")
    print("  step sweep (modes/8, ring_hit, mean_radius):")
    for k in STEPS:
        s = per_step[k]
        print(f"    {k:>3} -> {s['modes']}/8  hit {s['ring_hit']:.3f}  r {s['mean_radius']:.3f}")

    fail = []
    if f100["modes"] != META["flow_modes"]:
        fail.append(f"flow modes {f100['modes']} != {META['flow_modes']}")
    if abs(f100["mean_radius"] - META["flow_mean_radius"]) > RADIUS_TOL:
        fail.append(f"flow radius {f100['mean_radius']:.4f} != {META['flow_mean_radius']}")
    if abs(f100["ring_hit"] - META["flow_ring_hit"]) > HIT_TOL:
        fail.append(f"flow ring_hit {f100['ring_hit']:.4f} != {META['flow_ring_hit']}")
    if reg_modes != META["regress_modes"]:
        fail.append(f"regress modes {reg_modes} != {META['regress_modes']}")
    if abs(reg_r - META["regress_mean_radius"]) > RADIUS_TOL:
        fail.append(f"regress radius {reg_r:.4f} != {META['regress_mean_radius']}")
    if f2["modes"] != META["twostep_modes"]:
        fail.append(f"2-step modes {f2['modes']} != {META['twostep_modes']}")
    if fail:
        print("\nSTOP — regenerated toy does NOT match meta.yaml:")
        for f in fail:
            print("  x " + f)
        return 1

    # ------------------------------------------------------------------ subsample + dump
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(N_TOY, size=min(SUBSAMPLE, N_TOY), replace=False))

    def pack(a: np.ndarray) -> list:
        return [[round(float(x), 3), round(float(y), 3)] for x, y in a[idx]]

    # SVG domain: symmetric half-extent that comfortably holds the 2-step overshoot.
    extent = 0.0
    for k in STEPS:
        extent = max(extent, float(np.abs(per_step[k]["pts"][idx]).max()))
    extent = max(extent, float(np.abs(ring[idx]).max()))
    domain = round(extent * 1.06, 2)

    data = {
        "provenance": {
            "source": "curriculum/phase1_imitation/ch1.5_flow/flow.py",
            "generator": "site/scripts/vizdata/ch1.5_flow.py",
            "seed": SEED,
            "device": "cpu",
            "config": "default (model_dim 128, TOY_ITERS 1500, seed 0)",
            "stack": "torch 2.10.0 / numpy 2.4.6",
            "note": "Real points from flow.py's own toy net + ode_sample_loop; "
                    "matches meta.yaml reference_run. flow_steps is a SAMPLER-only "
                    "knob (training never sees it — the decoupling).",
        },
        "subsample": len(idx),
        "n_total": int(N_TOY),
        "domain": domain,
        "target": pack(ring),
        "regression": pack(reg_samples),
        "regression_stats": {
            "modes": int(reg_modes),
            "ring_hit": round(float(reg_hit), 4),
            "mean_radius": round(float(reg_r), 4),
        },
        "steps": STEPS,
        # full mode coverage arrives at ~3 steps; 2 under-resolves (Euler overshoot)
        "full_coverage_from": 3,
        "flow": {
            str(k): {
                "pts": pack(per_step[k]["pts"]),
                "modes": int(per_step[k]["modes"]),
                "ring_hit": round(float(per_step[k]["ring_hit"]), 4),
                "mean_radius": round(float(per_step[k]["mean_radius"]), 4),
            }
            for k in STEPS
        },
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB, {len(idx)} pts/set)")
    print("OK — matches meta.yaml; flow_steps decoupled from training (sampler-only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
