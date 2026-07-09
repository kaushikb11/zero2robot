#!/usr/bin/env python3
"""Regenerate the ch2.3 "swarm" hero-reel vizdata from ppo_mjx.py.

The landing hero's FIRST reel slide is a field of many cartpole robots training
at once — the ch2.3 "4096 Robots at Once" headline, made visible. MJX runs on
jax/XLA and CANNOT run in the browser, so (exactly like site/scripts/vizdata/
ch2.3_mjx.py) this generator RECORDS real per-env spatial state from the chapter
artifact and the reel replays those recorded numbers on a 2D canvas. Nothing is
scripted, faked, or a video: every cart position + pole angle below is a real
MJX rollout of ppo_mjx.py's own policy at seed 0.

What we record (all seed 0, cpu-jax, DEFAULT config — bitwise-deterministic)
---------------------------------------------------------------------------
We train the chapter's PPO exactly as main() does (num_envs 64, 36 updates ->
the run that SOLVES, meta.yaml eval 407.2), snapshotting the policy params at
TWO checkpoints:

  * EARLY  — after 2 gradient updates. The policy barely moves; poles topple.
  * LATE   — the final, solved policy. The field balances.

At each checkpoint we roll a SAMPLED GRID of N=48 fresh MJX cartpoles (a 6x8
field — a representative subset of the many parallel envs; the GPU Scale Lab
runs 4096) forward W control-steps under the DETERMINISTIC mean action (same as
evaluate()), recording each env's cart position qpos[0] and pole angle qpos[1].
The rollout is one jitted lax.scan (like evaluate) so it compiles once and runs
fast — a naive Python step loop is ~100x slower and blows the free-tier budget.

Honesty gate: the LATE policy must actually solve (eval >= 350) and clearly beat
EARLY, or we STOP — the "flailing -> coordinated" story must be real.

    Run:  .venv/bin/python site/scripts/vizdata/ch2.3_swarm.py
    Out:  curriculum/phase2_reinforcement/ch2.3_mjx/demo/swarm.vizdata.json
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PPO_PY = REPO / "curriculum" / "phase2_reinforcement" / "ch2.3_mjx" / "ppo_mjx.py"
OUT_JSON = REPO / "curriculum" / "phase2_reinforcement" / "ch2.3_mjx" / "demo" / "swarm.vizdata.json"

SEED = 0
GRID_ROWS, GRID_COLS = 6, 8      # N = 48 sampled envs — a viewable field
N_ENVS = GRID_ROWS * GRID_COLS
CAPTURE_STEPS = 120              # control-steps rolled per checkpoint (~2.4 s of sim)
KEEP_FRAMES = 36                 # downsample the window to this many frames for a compact JSON
EARLY_AFTER = 2                  # snapshot the "flailing" policy after this many updates
CAPTURE_SEED = 20260709          # dedicated PRNGKey for the recorded field (fixed -> reproducible)

META_EVAL_SEED0 = 407.2          # meta.yaml: the solved num_envs-64 run (the LATE policy target)
SOLVE_THRESHOLD = 350            # LATE must clear this; EARLY must be well below it


def exec_ppo_module():
    """Exec ppo_mjx.py with a pinned cpu/seed-0/no-rerun argv in an isolated,
    non-'__main__' namespace so main() does NOT run. Returns the populated globals
    (its build / evaluate / reset_batch / step_batch / obs_batch functions, args,
    jax, etc.), with ZERO edits to the LOC-capped chapter file. Mirrors the exec
    pattern in site/scripts/vizdata/ch2.3_mjx.py (flax needs a real module object
    in sys.modules to resolve its dataclass type hints)."""
    scratch = Path(tempfile.mkdtemp(prefix="ch2.3-swarm-"))
    mod_name = "ppo_mjx_swarmgen"
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


def make_capture(ns):
    """A jitted rollout that records the SPATIAL state of N_ENVS cartpoles: reset a
    fresh field, then step CAPTURE_STEPS times under the deterministic policy MEAN
    (no sampling — exactly how evaluate() acts), recording each env's cart position
    (qpos[0]) and pole angle (qpos[1], 0 = upright) at every step. One lax.scan, so
    it compiles once. Returns fn(params) -> (cart_x, angle) each (CAPTURE_STEPS, N)."""
    jax = ns["jax"]
    jnp = ns["jnp"]
    reset_batch, step_batch, obs_batch = ns["reset_batch"], ns["step_batch"], ns["obs_batch"]
    model = ns["ActorCritic"](ns["args"].hidden_dim)
    key = jax.random.PRNGKey(CAPTURE_SEED)

    def capture(params):
        datas = reset_batch(jax.random.split(key, N_ENVS))

        def body(carry, _):
            datas = carry
            obs = obs_batch(datas)
            mean, _, _ = model.apply({"params": params}, obs)  # deterministic mean action
            ndatas = step_batch(datas, mean)
            # obs = [cart_x, cart_v, cos(theta), sin(theta), theta_dot]; recover the
            # two spatial dofs: cart slider position and pole angle (atan2 seam-free).
            cart_x = obs[:, 0]
            angle = jnp.arctan2(obs[:, 3], obs[:, 2])
            return ndatas, (cart_x, angle)

        _, (cart_x, angle) = jax.lax.scan(body, datas, None, length=CAPTURE_STEPS)
        return cart_x, angle  # (CAPTURE_STEPS, N_ENVS)

    return jax.jit(capture)


def downsample_rows(arr, keep):
    """Even-stride subsample the leading (time) axis of a python list-of-lists to
    `keep` rows, always including first and last."""
    n = len(arr)
    if n <= keep:
        return arr
    return [arr[round(i * (n - 1) / (keep - 1))] for i in range(keep)]


def pack(cart_x, angle, ns):
    """(CAPTURE_STEPS, N) cart_x + angle -> a compact per-frame flat list. Each
    frame is [dx0, a0, dx1, a1, ...] over the N envs, where dx = cart_x clamped to
    +-CART_LIMIT and mapped to [-1, 1] (cell-local), and a = pole angle (rad)."""
    import numpy as np

    cart_limit = ns["CART_LIMIT"]
    cx = np.asarray(cart_x)
    ang = np.asarray(angle)
    dx = np.clip(cx / cart_limit, -1.0, 1.0)
    frames = []
    for t in range(cx.shape[0]):
        row = []
        for e in range(cx.shape[1]):
            row.append(round(float(dx[t, e]), 3))
            row.append(round(float(ang[t, e]), 3))
        frames.append(row)
    return downsample_rows(frames, KEEP_FRAMES)


def main() -> int:
    ns = exec_ppo_module()
    jax = ns["jax"]
    args = ns["args"]
    args.num_envs = 64  # the canonical solving config (matches meta.yaml eval 407.2)

    # --- train exactly as main() does, snapshotting EARLY + LATE params ----------
    batch_size = args.num_envs * args.num_steps
    num_iterations = args.total_steps // batch_size
    key = jax.random.PRNGKey(SEED)
    key, build_key, eval_key = jax.random.split(key, 3)  # == main()
    runner, update_step = ns["build"](args, build_key)

    early_params = None
    for it in range(1, num_iterations + 1):
        runner, _ = update_step(runner)
        jax.block_until_ready(runner)
        if it == EARLY_AFTER:
            early_params = runner[0].params  # the "flailing" snapshot
    late_params = runner[0].params           # the solved policy

    # --- honesty gate: LATE solves and clearly beats EARLY -----------------------
    evaluate = ns["evaluate"]
    apply_fn = runner[0].apply_fn
    eval_late = round(float(evaluate(apply_fn, late_params, eval_key, args.eval_envs)), 1)
    eval_early = round(float(evaluate(apply_fn, early_params, eval_key, args.eval_envs)), 1)
    print("ch2.3 swarm [seed 0, cpu-jax, default config]:")
    print(f"  iterations         : {num_iterations}  (meta 36)")
    print(f"  EARLY eval (after {EARLY_AFTER}): {eval_early}")
    print(f"  LATE  eval          : {eval_late}  (meta {META_EVAL_SEED0})")
    fail = []
    if eval_late < SOLVE_THRESHOLD:
        fail.append(f"LATE eval {eval_late} < solve threshold {SOLVE_THRESHOLD}")
    if not (eval_late > eval_early + 100):
        fail.append(f"LATE {eval_late} does not clearly beat EARLY {eval_early}")
    if num_iterations != 36:
        fail.append(f"num_iterations {num_iterations} != 36 (config drift)")
    if fail:
        print("\nSTOP — swarm story not real:")
        for f in fail:
            print("  x " + f)
        return 1

    # --- record the two fields ---------------------------------------------------
    capture = make_capture(ns)
    early_cx, early_ang = capture(early_params)
    late_cx, late_ang = capture(late_params)
    jax.block_until_ready(late_ang)

    data = {
        "provenance": {
            "source": "curriculum/phase2_reinforcement/ch2.3_mjx/ppo_mjx.py",
            "generator": "site/scripts/vizdata/ch2.3_swarm.py",
            "seed": SEED,
            "platform": "cpu-jax (JAX_PLATFORMS=cpu)",
            "config": "default (num_envs 64, num_steps 128, total_steps 300000)",
            "stack": "jax 0.10.2, mujoco-mjx 3.10.0",
            "recorded": True,
            "live_mjx": False,
            "sample": f"{N_ENVS} of the parallel MJX envs (a {GRID_ROWS}x{GRID_COLS} "
                      "field); the GPU Scale Lab runs 4096",
            "action": "deterministic policy mean (same as evaluate)",
            "note": "RECORDED per-env cart position + pole angle replayed in-browser "
                    "(MJX runs on jax/XLA and CANNOT run in the browser). EARLY = the "
                    "policy after 2 updates (flailing); LATE = the solved policy. "
                    f"eval EARLY {eval_early} -> LATE {eval_late} (bitwise, cpu-jax).",
        },
        "rows": GRID_ROWS,
        "cols": GRID_COLS,
        "n": N_ENVS,
        "eval": {"early": eval_early, "late": eval_late, "cap": ns["MAX_STEPS"]},
        # each checkpoint: KEEP_FRAMES frames; each frame flat [dx,angle]*N over envs
        "early": pack(early_cx, early_ang, ns),
        "late": pack(late_cx, late_ang, ns),
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB)")
    print(f"  {N_ENVS} envs x {KEEP_FRAMES} frames x 2 checkpoints (RECORDED, not live)")
    print("OK — real swarm; EARLY flails, LATE solves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
