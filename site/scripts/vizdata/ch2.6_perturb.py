#!/usr/bin/env python3
"""Regenerate the ch2.6 sim-to-real concept-toy vizdata from perturb.py, seed 0.

The site's LatencyDegradeToy island renders REAL degradation curves + REAL
cartpole rollouts — never invented shapes. This generator REUSES perturb.py's OWN
pieces (build_policy, the ObsDelay ring buffer, perceive, evaluate, run_sweep,
worst_family) so the browser panels are bit-faithful to the chapter artifact,
then dumps a small JSON the island loads.

Why we exec a PREFIX of perturb.py instead of `import perturb`
-------------------------------------------------------------
perturb.py is a loose script (no `if __name__ == "__main__"` guard): importing it
runs the WHOLE report — argparse, a full three-family sweep, metrics.json, and a
rerun recording. We must NOT modify perturb.py (it is LOC-capped). So we read its
source and exec only the prefix up to the `# --- region: report ---` line — i.e.
setup + policy + perturb + eval + sweep — in a throwaway namespace. That hands us
perturb.py's OWN build_policy, ObsDelay, perceive, evaluate, run_sweep,
worst_family, CartpoleEnv, TASKS and grids at the DEFAULT config (seed 0, cpu),
with zero edits.

Why the SCRIPTED baseline, not the learned PPO
----------------------------------------------
No policy binary ships in this repo (.pt is gitignored, root invariant 5), so the
site build must not depend on one. perturb.py is designed around exactly this: with
no checkpoint it falls back to the chapter's scripted baseline controller, which
was ALSO tuned for the clean sim and — per meta.yaml — "degrades under the same
noise/latency," so the lesson still lands. We therefore build the scripted policy
(build_policy(task, None)) and sweep the FULL grids on cpu, seed 0. This is the
same hermetic path CI runs, reproducible on any machine with no artifact.

The scripted baseline reproduces meta.yaml's LEARNED reference_run STRUCTURE
exactly, and that structure is the honesty gate below:
  - clean baseline holds (success 1.0, return == the 500-step horizon)
  - latency is the WORST perturbation (the only family that breaks it)
  - 8-step latency (160 ms) collapses it (success 0.0)
  - gravity mismatch is a NO-OP on the balancer (success 1.0 everywhere)
  - sensor noise only DENTS it (degrades, never to zero) — worse than gravity,
    far milder than latency
The learned PPO breaks even earlier (meta: latency@4 already gone); the scripted
baseline's cliff sits at 8. WHICH perturbation hurts most — latency, a thing DR
alone does not fix — is the seam into ch2.7, and it is identical for both.

    Run:  .venv/bin/python site/scripts/vizdata/ch2.6_perturb.py
    Out:  curriculum/phase2_reinforcement/ch2.6_perturb/demo/vizdata.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PERTURB_PY = REPO / "curriculum" / "phase2_reinforcement" / "ch2.6_perturb" / "perturb.py"
OUT_JSON = REPO / "curriculum" / "phase2_reinforcement" / "ch2.6_perturb" / "demo" / "vizdata.json"

SEED = 0
CUT_MARKER = "# --- region: report ---"

# Rollout capture: one held-out episode at low vs high latency, subsampled small.
ROLLOUT_LATENCY_LOW = 0    # clean: the balancer survives the full horizon
ROLLOUT_LATENCY_HIGH = 8   # 160 ms stale: it over-corrects on stale state and falls
ROLLOUT_SAMPLES = 90       # max frames kept per rollout trace (small text)

# meta.yaml reference_run (the LEARNED PPO, seeds 0-2) — the honesty gate. The
# scripted baseline is a different controller, so we gate on the STRUCTURE both
# share, not the learned policy's exact per-seed numbers.
META_BASELINE_SUCCESS = 1.0
META_WORST = "latency"
FULL_RETURN = 500.0        # cartpole horizon (MAX_STEPS) == a perfect balancing return
ABS_TOL = 1e-6


def exec_perturb_prefix() -> dict:
    """Exec perturb.py up to the report region, in an isolated namespace. Returns
    the populated globals dict (perturb.py's own policy + perturbation + eval fns).
    argparse runs DURING exec — pin seed 0, cpu, no rerun, and do NOT pass --smoke
    so the FULL sweep grids stand (latency 0..8, noise 0..0.2, gravity 0.5..2.0)."""
    src = PERTURB_PY.read_text()
    prefix = src[: src.index(CUT_MARKER)]

    scratch = Path(tempfile.mkdtemp(prefix="ch2.6-viz-"))
    old_argv = sys.argv
    sys.argv = [str(PERTURB_PY), "--seed", str(SEED), "--no-rerun",
                "--device", "cpu", "--out", str(scratch)]
    ns: dict = {"__file__": str(PERTURB_PY), "__name__": "perturb_toy_vizgen"}
    try:
        exec(compile(prefix, str(PERTURB_PY), "exec"), ns)  # noqa: S102 — our own trusted source
    finally:
        sys.argv = old_argv
    return ns


def record_rollout(ns: dict, policy, latency_steps: int) -> dict:
    """Run ONE held-out episode under `latency_steps` of observation delay,
    reusing perturb.py's exact ObsDelay + perceive loop (so the physics the policy
    fights is bit-faithful to evaluate()), and keep the pole-angle / cart-position
    trace. This is the same episode index the sweep's success rate is built from
    (seed EVAL_SEED0 + args.seed + 0)."""
    CartpoleEnv = ns["CartpoleEnv"]
    ObsDelay = ns["ObsDelay"]
    perceive = ns["perceive"]
    EVAL_SEED0 = ns["EVAL_SEED0"]

    env = CartpoleEnv()
    raw = env.reset(seed=EVAL_SEED0 + SEED + 0)
    delay = ObsDelay(latency_steps, raw)
    angles = [float(env.pole_angle)]
    carts = [float(env.cart_pos)]
    done, steps, info = False, 0, {"terminated": False, "truncated": False}
    while not done:
        action = policy(perceive(raw, delay, 0.0))
        raw, _reward, done, info = env.step(action)
        steps += 1
        angles.append(float(env.pole_angle))
        carts.append(float(env.cart_pos))

    n = len(angles)
    idx = np.linspace(0, n - 1, min(ROLLOUT_SAMPLES, n)).round().astype(int)
    hz = float(CartpoleEnv.CONTROL_HZ)
    return {
        "latency_steps": latency_steps,
        "latency_ms": int(round(latency_steps * 1000 / hz)),
        "steps": steps,                                  # == return (alive bonus +1/step)
        "fell": bool(info["terminated"]),                # pole fell / cart off rail
        "survived": bool(info["truncated"]),             # reached the horizon upright
        "outcome": "falls" if info["terminated"] else "balances",
        "t": [round(float(i) / hz, 3) for i in idx],     # seconds
        "pole_angle": [round(angles[i], 4) for i in idx],  # rad, 0 = upright
        "cart_pos": [round(carts[i], 4) for i in idx],     # m, 0 = rail center
    }


def main() -> int:
    ns = exec_perturb_prefix()
    build_policy = ns["build_policy"]
    run_sweep = ns["run_sweep"]
    worst_family = ns["worst_family"]
    args = ns["args"]
    TASK = ns["TASK"]
    CartpoleEnv = ns["CartpoleEnv"]

    assert args.task == "cartpole", f"expected cartpole default, got {args.task}"
    assert not args.smoke, "generator must run FULL grids (no --smoke)"

    # Force the SCRIPTED baseline (no binary): the hermetic path perturb.py uses in
    # CI, reproducible anywhere, and — per meta.yaml — degrading the same way.
    policy, policy_label = build_policy(args.task, None)
    assert policy_label == "scripted-baseline", policy_label

    # ---------------------------------------------------------------- the sweeps
    swept = run_sweep(policy)                       # perturb.py's OWN three-family sweep
    worst, drop = worst_family(swept["baseline"], swept["sweeps"])
    baseline = swept["baseline"]
    metric_name = TASK["metric_name"]               # "mean_return"

    # human labels/axes for each family (the physics is perturb.py's; this is UI)
    hz = float(CartpoleEnv.CONTROL_HZ)
    FAMILY_META = {
        "latency": {"label": "observation latency", "unit": "steps",
                    "axis": "latency (control steps)", "ms_per_step": int(round(1000 / hz))},
        "sensor_noise": {"label": "sensor noise", "unit": "std",
                         "axis": "gaussian obs-noise σ", "ms_per_step": None},
        "model_mismatch": {"label": "gravity mismatch", "unit": "×",
                           "axis": "gravity scale (× nominal)", "ms_per_step": None},
    }
    sweeps_out: dict = {}
    for name, fam in swept["sweeps"].items():
        fm = FAMILY_META[name]
        sweeps_out[name] = {
            "knob": fam["knob"],
            "label": fm["label"],
            "unit": fm["unit"],
            "axis": fm["axis"],
            "ms_per_step": fm["ms_per_step"],
            # each point: [magnitude, success_rate, mean_return]
            "points": [[round(float(m), 6), round(float(sr), 6), round(float(mt), 4)]
                       for m, sr, mt in fam["points"]],
        }

    # ------------------------------------------------------------- the rollouts
    rollouts = {
        "clean": record_rollout(ns, policy, ROLLOUT_LATENCY_LOW),
        "degraded": record_rollout(ns, policy, ROLLOUT_LATENCY_HIGH),
    }

    # ------------------------------------------------------------------ honesty gate
    def pt(name, i):
        return sweeps_out[name]["points"][i]

    base_succ = round(float(baseline["success_rate"]), 6)
    base_ret = round(float(baseline[metric_name]), 4)
    lat_last = sweeps_out["latency"]["points"][-1]        # latency @ 8 steps
    grav = sweeps_out["model_mismatch"]["points"]
    noise_last = sweeps_out["sensor_noise"]["points"][-1]  # noise @ 0.2

    print("regenerated perturb [seed 0, cpu, scripted-baseline] vs meta.yaml reference_run:")
    print(f"  baseline        : success {base_succ:.2f}  return {base_ret:.1f}   (meta success 1.00)")
    print(f"  worst family    : {worst}  (success drop {drop:+.2f})   (meta latency)")
    print(f"  latency @ 8 step : success {lat_last[1]:.2f}  return {lat_last[2]:.1f}   (meta [0,0,0])")
    print(f"  sensor_noise@0.2 : success {noise_last[1]:.2f}  return {noise_last[2]:.1f}")
    print(f"  gravity mismatch : success {[p[1] for p in grav]}   (meta 1.00 everywhere)")
    print(f"  rollout clean    : {rollouts['clean']['steps']} steps, {rollouts['clean']['outcome']}")
    print(f"  rollout degraded : {rollouts['degraded']['steps']} steps, {rollouts['degraded']['outcome']}")

    fail = []
    if abs(base_succ - META_BASELINE_SUCCESS) > ABS_TOL:
        fail.append(f"baseline success {base_succ} != meta {META_BASELINE_SUCCESS}")
    if abs(base_ret - FULL_RETURN) > ABS_TOL:
        fail.append(f"baseline return {base_ret} != full horizon {FULL_RETURN}")
    if worst != META_WORST:
        fail.append(f"worst perturbation {worst} != meta {META_WORST}")
    # latency is the breaker: its extreme point collapses success to 0
    if lat_last[1] > ABS_TOL:
        fail.append(f"latency@8 success {lat_last[1]} should be 0 (meta [0,0,0])")
    # gravity mismatch is a NO-OP on the balancer (success 1.0 at every grid point)
    if any(abs(p[1] - 1.0) > ABS_TOL for p in grav):
        fail.append(f"gravity mismatch not a no-op: {[p[1] for p in grav]}")
    # ordering headline — WHICH perturbation hurts most: latency >> noise > gravity
    def drop_of(name):
        return base_succ - sweeps_out[name]["points"][-1][1]
    d_lat, d_noise, d_grav = drop_of("latency"), drop_of("sensor_noise"), drop_of("model_mismatch")
    if not (d_lat > d_noise > d_grav):
        fail.append(f"ordering latency>>noise>gravity broken: {d_lat} {d_noise} {d_grav}")
    if not (d_grav == 0.0):
        fail.append(f"gravity drop should be exactly 0: {d_grav}")
    if not (0.0 < d_noise < d_lat):
        fail.append(f"noise should DENT (0 < drop < latency): {d_noise}")
    # rollout structural facts: clean survives upright, degraded falls past the limit
    limit = float(CartpoleEnv.ANGLE_LIMIT)
    rc, rd = rollouts["clean"], rollouts["degraded"]
    if not (rc["survived"] and not rc["fell"]):
        fail.append(f"clean rollout should survive to the horizon: {rc['outcome']} @ {rc['steps']}")
    if not (rc["steps"] == CartpoleEnv.MAX_STEPS):
        fail.append(f"clean rollout should reach MAX_STEPS: {rc['steps']}")
    if max(abs(a) for a in rc["pole_angle"]) > limit:
        fail.append("clean rollout pole should never exceed the fall limit")
    if not (rd["fell"] and not rd["survived"]):
        fail.append(f"degraded rollout should fall: {rd['outcome']} @ {rd['steps']}")
    if max(abs(a) for a in rd["pole_angle"]) < limit:
        fail.append("degraded rollout pole should exceed the fall limit before it ends")
    if not (rd["steps"] < rc["steps"]):
        fail.append(f"degraded rollout should end sooner than clean: {rd['steps']} vs {rc['steps']}")

    if fail:
        print("\nSTOP — regenerated perturb does NOT match meta.yaml structure:")
        for f in fail:
            print("  x " + f)
        return 1

    # ---------------------------------------------------------------------- pack
    data = {
        "provenance": {
            "source": "curriculum/phase2_reinforcement/ch2.6_perturb/perturb.py",
            "generator": "site/scripts/vizdata/ch2.6_perturb.py",
            "seed": SEED,
            "device": "cpu",
            "policy": policy_label,
            "eval_episodes": int(args.eval_episodes),
            "config": "default full grids (seed 0, cpu), scripted-baseline (hermetic, no binary)",
            "stack": "torch 2.10.0 / mujoco 3.10.0 / numpy 2.4.6",
            "note": "Real degradation curves + rollouts from perturb.py's OWN "
                    "build_policy + ObsDelay + perceive + evaluate + run_sweep. The "
                    "SCRIPTED baseline is the hermetic controller perturb.py runs "
                    "when no .pt is present (root invariant 5: no binaries in git); "
                    "per meta.yaml it degrades under the same noise/latency as the "
                    "learned PPO. Matches meta.yaml reference_run STRUCTURE: baseline "
                    "holds, latency is the worst family and 8-step (160 ms) latency "
                    "breaks it, gravity mismatch is a no-op, sensor noise only dents. "
                    "The learned PPO breaks EARLIER (meta: latency@4 already gone); "
                    "the scripted baseline's cliff sits at 8. WHICH perturbation hurts "
                    "most — latency, which DR alone does not fix — is identical for "
                    "both, and is the seam into ch2.7.",
        },
        "task": "cartpole",
        "policy": policy_label,
        "seed": SEED,
        "device": "cpu",
        "eval_episodes": int(args.eval_episodes),
        "control_hz": int(CartpoleEnv.CONTROL_HZ),
        "ms_per_step": int(round(1000 / hz)),
        "angle_limit_rad": round(float(CartpoleEnv.ANGLE_LIMIT), 4),
        "max_steps": int(CartpoleEnv.MAX_STEPS),
        "metric_name": metric_name,
        "families": ["latency", "sensor_noise", "model_mismatch"],
        "default_family": "latency",
        "baseline": {"success_rate": base_succ, "mean_return": base_ret},
        "sweeps": sweeps_out,
        "worst_perturbation": worst,
        "rollouts": rollouts,
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB)")
    print("OK — matches meta.yaml structure; latency >> noise > gravity(no-op); "
          "clean balances, +160 ms latency falls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
