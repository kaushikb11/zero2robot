#!/usr/bin/env python3
"""Regenerate the ch2.3 "mjx_parallel_training" concept-toy vizdata from ppo_mjx.py.

The site's MjxParallelToy island renders a RECORDED-DATA concept toy (like ch1.4's
diffusion ring or ch1.6's eval bands) — NOT a live MJX sim. MJX runs on jax/XLA and
CANNOT run in the browser (no jax-WASM path; the playground's contract runtimes are
for torch->ONNX policies only), so the toy replays REAL numbers this generator
measures from the chapter artifact at seed 0.

What we measure (all seed 0, cpu-jax, DEFAULT config — bitwise-deterministic)
---------------------------------------------------------------------------
  1. THE WALL-CLOCK CLIFF: ppo_mjx.py's own `sweep` over num_envs in {16,64,256,
     1024}. Throughput (env-steps/sec) CLIMBS with parallel envs then PLATEAUS and
     REVERSES on CPU around 256 envs (cores saturate). Timing is representative, not
     bitwise, so we dump meta.yaml's canonical reference numbers as the curve and
     assert this run REPRODUCES the qualitative shape (16 < 64 < 256 > 1024).
  2. THE RETURN CURVE: a full training run at num_envs 64 (36 gradient updates),
     capturing the mean episodic return per iteration, then the deterministic eval.
     Must reproduce meta.yaml's eval_return_seed0 = 407.2 (the pole is solved).
  3. THE THROUGHPUT-vs-GRADIENT-QUALITY tradeoff: a second training run at num_envs
     256 (only 9 updates at the same 300k env-step budget). Must reproduce
     meta.yaml's 90.4 eval — runs FASTER per step yet learns WORSE (fewer updates).

Why we exec ppo_mjx.py instead of `import` + main()
---------------------------------------------------
ppo_mjx.py parses argv and sets JAX_PLATFORMS at MODULE load, and only runs main()
under `if __name__ == "__main__"`. So (exactly like site/scripts/vizdata/ch1.5_flow.py)
we exec the file with a pinned argv and a non-"__main__" name in a throwaway
namespace — that gives us its OWN build / update_step / evaluate functions, the real
MJX env, and the real PPO, with ZERO edits to the file (it is LOC-capped). We then
drive its training loop and sweep ourselves, mirroring main()'s key splits so the
eval reproduces bit-for-bit.

    Run:  .venv/bin/python site/scripts/vizdata/ch2.3_mjx.py
    Out:  curriculum/phase2_reinforcement/ch2.3_mjx/demo/vizdata.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PPO_PY = REPO / "curriculum" / "phase2_reinforcement" / "ch2.3_mjx" / "ppo_mjx.py"
OUT_JSON = REPO / "curriculum" / "phase2_reinforcement" / "ch2.3_mjx" / "demo" / "vizdata.json"

SEED = 0

# meta.yaml reference_run (seed 0, cpu-jax, DEFAULT config; measured 2026-07-06) —
# the honesty gate + the canonical cliff curve the chapter/exercises cite. If a
# regenerated number drifts from these MEASURED values, STOP.
META_CLIFF_ENVS = [16, 64, 256, 1024]
META_CLIFF_THROUGHPUT = [50154, 83638, 104544, 76387]  # env-steps/sec; 256 = the CPU peak
META_PLATEAU_ENVS = 256
META_EVAL_SEED0 = 407.2   # num_envs 64, 36 updates — solves (cap 500, random ~10-30)
META_EVAL_SEED1 = 499.7   # seed band upper end (meta reference; we run seed 0 only)
META_EVAL_256 = 90.4      # num_envs 256, 9 updates — runs faster, learns worse
EVAL_TOL = 1.0            # cpu determinism is bitwise; tol only guards float print noise

# Projected GPU continuation for the cliff panel. NOT measured here — an illustration
# of the Scale-Lab regime (jax[cuda], 4090 / L40S), where throughput keeps climbing
# past the CPU plateau to 4096 envs. The toy renders this DASHED and clearly labeled
# "projected — not measured", per demo/embed.yaml's tier_honesty.
PROJECTED_GPU_ENVS = [256, 1024, 4096]
PROJECTED_GPU_THROUGHPUT = [900000, 3200000, 9000000]  # order-of-magnitude schematic only


def exec_ppo_module():
    """Exec ppo_mjx.py with a pinned cpu/seed-0/no-rerun argv in an isolated,
    non-'__main__' namespace so main() does NOT run. Returns the populated globals
    (its build / update_step / evaluate functions, args, jax, etc.), zero edits.

    We register a real module object in sys.modules first: flax's Module dataclass
    transform resolves type hints via sys.modules[cls.__module__], so a bare exec
    dict (as ch1.5_flow.py uses for torch) would crash. The name is unique + removed
    in the finally so we never shadow a real import."""
    import types

    scratch = Path(tempfile.mkdtemp(prefix="ch2.3-viz-"))
    mod_name = "ppo_mjx_vizgen"
    mod = types.ModuleType(mod_name)
    mod.__file__ = str(PPO_PY)
    old_argv = sys.argv
    sys.argv = [str(PPO_PY), "--seed", str(SEED), "--no-rerun",
                "--platform", "cpu", "--out", str(scratch)]
    sys.modules[mod_name] = mod
    try:
        exec(compile(PPO_PY.read_text(), str(PPO_PY), "exec"), mod.__dict__)  # noqa: S102 — our own trusted source
    finally:
        sys.argv = old_argv
        sys.modules.pop(mod_name, None)
    return mod.__dict__


def train_capture(ns, num_envs):
    """Run ppo_mjx's training loop at `num_envs`, mirroring main()'s key splits so
    the eval is bit-identical. Returns (per-iteration returns, num_updates, eval)."""
    jax = ns["jax"]
    args = ns["args"]
    args.num_envs = num_envs

    batch_size = args.num_envs * args.num_steps
    num_iterations = args.total_steps // batch_size
    key = jax.random.PRNGKey(SEED)
    key, build_key, eval_key = jax.random.split(key, 3)  # == main()
    runner, update_step = ns["build"](args, build_key)

    returns, global_steps = [], []
    global_step = 0
    for _ in range(1, num_iterations + 1):
        runner, (ret, _stats) = update_step(runner)
        jax.block_until_ready(runner)
        global_step += batch_size
        r = float(ret)
        returns.append(None if (r != r) else round(r, 2))  # NaN (no episode ended yet) -> null
        global_steps.append(global_step)

    ts = runner[0]
    mean_eval = float(ns["evaluate"](ts.apply_fn, ts.params, eval_key, args.eval_envs))
    return returns, global_steps, num_iterations, round(mean_eval, 1)


def measure_cliff(ns):
    """Re-run ppo_mjx's sweep logic (build + warmup + time 3 steps) at each num_envs
    to VERIFY the qualitative cliff shape reproduces. Returns measured env-steps/sec."""
    import time

    jax = ns["jax"]
    args = ns["args"]
    key = jax.random.PRNGKey(SEED)
    measured = []
    for n in META_CLIFF_ENVS:
        args.num_envs = n
        runner, update_step = ns["build"](args, key)
        for _ in range(2):  # warmup: pay the XLA compile (twice, as ppo_mjx does)
            runner, _ = update_step(runner)
        jax.block_until_ready(runner)
        t0, reps = time.time(), 3
        for _ in range(reps):
            runner, _ = update_step(runner)
        jax.block_until_ready(runner)
        measured.append(reps * n * args.num_steps / (time.time() - t0))
    return measured


def main() -> int:
    ns = exec_ppo_module()

    # (1) return curve + eval at num_envs 64 (the solving run) --------------------
    returns64, steps64, updates64, eval64 = train_capture(ns, 64)
    # (2) the tradeoff: num_envs 256, same 300k budget, only 9 updates ------------
    _returns256, _steps256, updates256, eval256 = train_capture(ns, 256)
    # (3) the wall-clock cliff shape (timing; verify, don't commit the raw numbers)
    measured_cliff = measure_cliff(ns)

    # ------------------------------------------------------------------ honesty gate
    print("regenerated ch2.3 toy [seed 0, cpu-jax, default config] vs meta.yaml:")
    print(f"  return curve  : num_envs 64, {updates64} updates, eval {eval64}  (meta 407.2)")
    print(f"  tradeoff      : num_envs 256, {updates256} updates, eval {eval256}  (meta 90.4)")
    print("  wall-clock cliff (measured now vs meta canonical env-steps/sec):")
    for n, m, ref in zip(META_CLIFF_ENVS, measured_cliff, META_CLIFF_THROUGHPUT):
        print(f"    {n:>5} envs -> {m:>10,.0f}   (meta {ref:>7,})")

    fail = []
    if abs(eval64 - META_EVAL_SEED0) > EVAL_TOL:
        fail.append(f"eval64 {eval64} != meta {META_EVAL_SEED0}")
    if abs(eval256 - META_EVAL_256) > EVAL_TOL:
        fail.append(f"eval256 {eval256} != meta {META_EVAL_256}")
    if updates64 != 36:
        fail.append(f"updates64 {updates64} != 36")
    if updates256 != 9:
        fail.append(f"updates256 {updates256} != 9")
    # qualitative cliff shape: climbs 16<64<256, then reverses 1024<256 (the plateau)
    m16, m64, m256, m1024 = measured_cliff
    if not (m16 < m64 < m256):
        fail.append(f"cliff does not climb 16<64<256: {measured_cliff}")
    if not (m1024 < m256):
        fail.append(f"cliff does not reverse past plateau (1024<256): {measured_cliff}")
    if fail:
        print("\nSTOP — regenerated toy does NOT match meta.yaml:")
        for f in fail:
            print("  x " + f)
        return 1

    # ------------------------------------------------------------------ assemble + dump
    def updates_at(n: int) -> int:
        return 300_000 // (n * 128)  # fixed 300k env-step budget / (num_envs * num_steps)

    # tradeoff rows: updates for ALL four env counts (computed); eval only where
    # actually MEASURED (64 & 256). throughput = meta canonical cliff.
    eval_by_env = {64: eval64, 256: eval256}
    tradeoff = [
        {
            "num_envs": n,
            "throughput": t,
            "updates": updates_at(n),
            "eval": eval_by_env.get(n),          # null where not measured
            "solves": (eval_by_env.get(n) or 0) >= 350,
        }
        for n, t in zip(META_CLIFF_ENVS, META_CLIFF_THROUGHPUT)
    ]

    data = {
        "provenance": {
            "source": "curriculum/phase2_reinforcement/ch2.3_mjx/ppo_mjx.py",
            "generator": "site/scripts/vizdata/ch2.3_mjx.py",
            "seed": SEED,
            "platform": "cpu-jax (JAX_PLATFORMS=cpu)",
            "config": "default (num_envs 64, num_steps 128, total_steps 300000)",
            "stack": "jax 0.10.2, mujoco-mjx 3.10.0",
            "recorded": True,
            "live_mjx": False,
            "note": "RECORDED data replayed in-browser — MJX runs on jax/XLA and "
                    "CANNOT run in the browser. Cliff throughput is meta.yaml's "
                    "canonical reference (timing, not bitwise); this run reproduced "
                    "its shape (16<64<256>1024). eval_return is bitwise (cpu-jax).",
        },
        # ---- Panel 1: the wall-clock cliff ------------------------------------
        "cliff": {
            "num_envs": META_CLIFF_ENVS,
            "throughput": META_CLIFF_THROUGHPUT,
            "plateau_at": META_PLATEAU_ENVS,
            "measured_now": [round(x) for x in measured_cliff],  # this run, for the record
            "projected_gpu": {
                "num_envs": PROJECTED_GPU_ENVS,
                "throughput": PROJECTED_GPU_THROUGHPUT,
                "measured": False,
                "note": "PROJECTED (Scale Lab: jax[cuda], 4090 / L40S) — NOT measured "
                        "here. A GPU keeps climbing past the CPU plateau to 4096 envs.",
            },
        },
        # ---- Panel 2: parallel training — the recorded return climb -----------
        "return_curve": {
            "num_envs": 64,
            "updates": updates64,
            "cap": 500,
            "solve_threshold": 350,
            "random_baseline": 30,
            "global_step": steps64,
            "returns": returns64,               # per-iteration mean episodic return
            "eval_return_seed0": eval64,        # measured now
            "eval_return_seed1": META_EVAL_SEED1,  # meta reference (seed band upper end)
        },
        "grid": {"rows": 8, "cols": 8},         # 64 envs, a viewable 8x8 schematic
        # ---- Panel 3: throughput vs gradient quality --------------------------
        "tradeoff": {
            "budget_env_steps": 300_000,
            "rows": tradeoff,
            "note": "At a FIXED 300k env-step budget, MORE envs = FEWER gradient "
                    "updates. 256 envs runs faster (105k env-steps/s) yet learns "
                    "worse (9 updates -> 90) than 64 envs (36 updates -> 407).",
        },
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB)")
    print("OK — matches meta.yaml; RECORDED data (MJX cannot run in-browser).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
