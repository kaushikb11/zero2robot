#!/usr/bin/env python3
"""Regenerate the ch2.2 SAC "off-policy bargain" concept-toy vizdata, seed 0.

THE OFF-POLICY BARGAIN, made into a browser toy: the SAC-trained pusher-reach arm
driving its fingertip onto a held-out target (a recorded rollout replay), and the
sample-efficiency headline — SAC vs a compact on-policy PPO, environment-steps to
solve the SAME task. The site's SacReachToy island renders REAL curves + a REAL
recorded rollout from sac.py + compare_ppo_sac.py, never invented shapes.

Why we exec a PREFIX of sac.py instead of `import sac`
------------------------------------------------------
sac.py is a loose script (same pattern as the ch3.3-3.6 engines): its train region
runs the WHOLE 30k-step training loop at module scope, then the report region saves
a checkpoint + metrics.json + an optional .rrd. We must NOT modify sac.py (it is
LOC-capped and voice/human-owned). So we read its source and exec only the PREFIX
up to the `# --- region: report ---` marker in a throwaway namespace, at the DEFAULT
config (seed 0, cpu, --no-rerun). That trains SAC EXACTLY as the chapter does and
hands us sac.py's OWN trained `actor`, its `evaluate`, its measured `curve` and
`env_steps_to_solve`, plus `PusherReachEnv` — with zero edits and none of the
report/metrics/rerun side effects.

What we record
--------------
  1. THE SAMPLE-EFFICIENCY CURVE — sac.py's OWN `curve` (env_step, eval_return,
     eval_dist), which reproduces meta.yaml's reference_run bit-for-bit on CPU
     (final 0.0434 m, solved <0.05 m at 18000 env steps). Plus the compact PPO
     reference from compare_ppo_sac.py (train_ppo, verbatim) over its 200k budget:
     it learns from ~0.176 m down to a ~0.13 m plateau but never clears the 0.05 m
     bar — the off-policy bargain, MEASURED.
  2. THE RECORDED ROLLOUT — the trained deterministic actor (tanh of the mean, no
     sampling, exactly sac.py's eval action) rolled out on the FIRST held-out eval
     seed that succeeds (deterministic, not cherry-picked), recording the arm
     geometry (elbow + fingertip) so the toy replays the arm reaching the target.

We STOP if the reproduced numbers drift from meta.yaml's reference_run.

    Run:  .venv/bin/python site/scripts/vizdata/ch2.2_sac.py
    Out:  curriculum/phase2_reinforcement/ch2.2_sac/demo/vizdata.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
SAC_PY = REPO / "curriculum" / "phase2_reinforcement" / "ch2.2_sac" / "sac.py"
COMPARE_PY = REPO / "curriculum" / "phase2_reinforcement" / "ch2.2_sac" / "compare_ppo_sac.py"
OUT_JSON = REPO / "curriculum" / "phase2_reinforcement" / "ch2.2_sac" / "demo" / "vizdata.json"

SEED = 0
CUT_MARKER = "# --- region: report ---"
PPO_BUDGET = 200_000      # compare_ppo_sac.py's on-policy budget (its default; the finding)
PPO_POINTS = 40           # PPO curve points kept (small committed text)
ROLLOUT_FRAMES = 48       # frames kept along the recorded reaching rollout

# meta.yaml reference_run (seed 0, cpu; torch 2.10.0 / mujoco 3.10.0 / numpy 2.4.6)
# — the honesty gate. sac.py's seed-0 CPU run is bitwise reproducible, so these are
# exact; the tolerances only guard float print noise. If the regenerated SAC drifts
# from these MEASURED numbers, STOP.
META = {
    "env_steps_to_solve": 18000,
    "final_dist": 0.04342,
    "final_return": -5.8472,
    "final_success": 0.4,
    "scripted_dist": 9e-05,
}
# sac.py's OWN measured curve (env_step, eval_return, eval_dist), seed 0, full 30k.
# Embedded so the gate also catches a mid-curve drift, not only the endpoints.
META_CURVE = [
    (2000, -17.076, 0.15239), (4000, -16.295, 0.14825), (6000, -15.527, 0.1418),
    (8000, -15.676, 0.14032), (10000, -12.751, 0.07985), (12000, -10.935, 0.06264),
    (14000, -11.356, 0.0796), (16000, -10.357, 0.05904), (18000, -8.587, 0.03584),
    (20000, -8.836, 0.06257), (22000, -7.268, 0.04404), (24000, -7.292, 0.04753),
    (26000, -6.126, 0.03801), (28000, -6.08, 0.03929), (30000, -5.847, 0.04342),
]
DIST_TOL = 5e-4     # eval distances reproduce to ~1e-5 on CPU; this guards print noise
RET_TOL = 5e-2      # returns reproduce to ~1e-3; loose guard

# Honest, non-artifact baselines quoted in the chapter (prose + env docstring).
RANDOM_DIST = 0.176     # random policy leaves the fingertip ~0.176 m from target
SOLVE_DIST = 0.05       # eval mean final dist below this counts as "solved" (sac.py)


def exec_sac_prefix() -> dict:
    """Exec sac.py up to the report region, in an isolated namespace. This TRAINS
    SAC exactly as the chapter does (seed 0, cpu, 30k steps) and returns the
    populated globals — sac.py's own trained `actor`, `evaluate`, `curve`, etc."""
    src = SAC_PY.read_text()
    prefix = src[: src.index(CUT_MARKER)]
    scratch = Path(tempfile.mkdtemp(prefix="ch2.2-viz-"))
    # argparse runs DURING exec — pin the DEFAULT config on cpu, no rerun path.
    old_argv = sys.argv
    sys.argv = [str(SAC_PY), "--seed", str(SEED), "--no-rerun",
                "--device", "cpu", "--out", str(scratch)]
    ns: dict = {"__file__": str(SAC_PY), "__name__": "ch22_sac_vizgen"}
    try:
        exec(compile(prefix, str(SAC_PY), "exec"), ns)  # noqa: S102 — our own trusted source
    finally:
        sys.argv = old_argv
    return ns


def subsample_idx(n: int, keep: int) -> np.ndarray:
    """Evenly spaced indices into range(n), always including first + last."""
    if n <= keep:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, keep).round().astype(int))


def record_reach_rollout(ns: dict) -> dict:
    """Roll out the trained DETERMINISTIC actor (tanh of the mean, sac.py's eval
    action) on held-out eval seeds and record the FIRST one that succeeds — the arm
    reaching the target. Records arm geometry (elbow + fingertip) per frame so the
    toy can replay the reach. First-of-kind pick: deterministic, not cherry-picked."""
    actor = ns["actor"]
    PusherReachEnv = ns["PusherReachEnv"]
    device = ns["device"]
    eval_env = PusherReachEnv()
    elbow_id = eval_env.model.body("link2").id  # link2 body origin sits at the elbow joint

    def det_action(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            _, _, mean_action = actor.sample(
                torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
        return mean_action[0].cpu().numpy()

    def one_episode(seed: int) -> dict:
        obs = eval_env.reset(seed=seed)
        tx, ty = eval_env.target_pos
        frames, dists = [], []
        reached_frame = None

        def snap(i: int) -> None:
            nonlocal reached_frame
            ex, ey = eval_env.data.xpos[elbow_id][:2]
            fx, fy = eval_env.fingertip_pos
            d = eval_env._dist()
            frames.append((float(ex), float(ey), float(fx), float(fy)))
            dists.append(float(d))
            if reached_frame is None and d < eval_env.SUCCESS_TOL:
                reached_frame = i

        snap(0)
        done, i = False, 0
        info = {"success": False, "dist": eval_env._dist()}
        while not done:
            obs, _, done, info = eval_env.step(det_action(obs))
            i += 1
            snap(i)
        return {"seed": seed, "target": [float(tx), float(ty)], "frames": frames,
                "dists": dists, "reached_frame": reached_frame,
                "success": bool(info["success"]), "final_dist": float(info["dist"])}

    # sac.py evals on seeds 500_000 + args.seed + ep; walk the same held-out block
    # and take the first success. Fall back to the closest final-dist if (very
    # unlikely) none succeed in the window — flagged so nothing silent.
    best = None
    for ep in range(30):
        r = one_episode(500_000 + SEED + ep)
        if r["success"]:
            return r
        if best is None or r["final_dist"] < best["final_dist"]:
            best = r
    best["_no_success_in_window"] = True
    return best


def pack_rollout(r: dict) -> dict:
    idx = subsample_idx(len(r["frames"]), ROLLOUT_FRAMES)
    frames = [r["frames"][i] for i in idx]
    dists = [r["dists"][i] for i in idx]
    # remap reached_frame onto the subsampled index space (first kept frame at/after it)
    reached = None
    if r["reached_frame"] is not None:
        for j, i in enumerate(idx):
            if i >= r["reached_frame"]:
                reached = j
                break
    return {
        "seed": r["seed"],
        "target": [round(r["target"][0], 4), round(r["target"][1], 4)],
        "success": r["success"],
        "final_dist": round(r["final_dist"], 4),
        "reached_frame": reached,
        # [elbow_x, elbow_y, fingertip_x, fingertip_y, dist] per frame (base at origin)
        "frames": [[round(ex, 4), round(ey, 4), round(fx, 4), round(fy, 4), round(d, 4)]
                   for (ex, ey, fx, fy), d in zip(frames, dists)],
    }


def run_ppo() -> tuple[int | None, list]:
    """The compact on-policy PPO reference (compare_ppo_sac.train_ppo, VERBATIM) over
    its 200k budget. Returns (steps_to_solve or None, curve[(env_step, eval_dist)])."""
    sys.path.insert(0, str(COMPARE_PY.parent))
    import compare_ppo_sac  # the chapter's companion tooling, imported unmodified
    return compare_ppo_sac.train_ppo(SEED, PPO_BUDGET, "cpu")


def main() -> int:
    ns = exec_sac_prefix()
    curve = ns["curve"]                     # [(env_step, eval_return, eval_dist)], seed 0
    steps_to_solve = ns["steps_to_solve"]
    evaluate = ns["evaluate"]

    final_return, final_dist, final_success = evaluate(ns["args"].eval_episodes)

    # ------------------------------------------------------------------ honesty gate
    print("regenerated SAC [seed 0, cpu, 30k] vs meta.yaml reference_run:")
    print(f"  env_steps_to_solve : {steps_to_solve}   (meta {META['env_steps_to_solve']})")
    print(f"  final eval dist    : {final_dist:.5f} m  (meta {META['final_dist']})")
    print(f"  final eval return  : {final_return:.4f}   (meta {META['final_return']})")
    print(f"  final success rate : {final_success:.2f}     (meta {META['final_success']})")

    fail: list[str] = []
    if steps_to_solve != META["env_steps_to_solve"]:
        fail.append(f"env_steps_to_solve {steps_to_solve} != {META['env_steps_to_solve']}")
    if abs(final_dist - META["final_dist"]) > DIST_TOL:
        fail.append(f"final_dist {final_dist:.5f} != {META['final_dist']}")
    if abs(final_return - META["final_return"]) > RET_TOL:
        fail.append(f"final_return {final_return:.4f} != {META['final_return']}")
    if abs(final_success - META["final_success"]) > 1e-9:
        fail.append(f"final_success {final_success} != {META['final_success']}")
    if len(curve) != len(META_CURVE):
        fail.append(f"curve length {len(curve)} != {len(META_CURVE)}")
    else:
        for (s, ret, d), (ms, mret, md) in zip(curve, META_CURVE):
            if s != ms or abs(d - md) > DIST_TOL or abs(ret - mret) > RET_TOL:
                fail.append(f"curve point {s}: ({ret:.3f},{d:.5f}) drifted from ({mret},{md})")
    # the sample-efficiency headline: SAC crosses the solve bar, and it does so where
    # meta says (18000); the last eval is noisy RL (dist bounces around the bar).
    if not any(d < SOLVE_DIST for _, _, d in curve):
        fail.append("SAC never crossed the solve bar in the curve")
    if fail:
        print("\nSTOP — regenerated SAC does NOT match meta.yaml:")
        for f in fail:
            print("  x " + f)
        return 1

    # --------------------------------------------------------------- the rollout
    roll = record_reach_rollout(ns)
    if roll.get("_no_success_in_window"):
        print("\nSTOP — no successful held-out rollout found in the eval-seed window "
              "(the reaching panel needs a real success).")
        return 1
    packed = pack_rollout(roll)
    print(f"\nrecorded reaching rollout: held-out seed {packed['seed']}, "
          f"final dist {packed['final_dist']} m, reached at frame {packed['reached_frame']}")

    # ------------------------------------------------------------------ PPO side
    print(f"\nPPO (on-policy, discards each rollout) on pusher-reach, budget {PPO_BUDGET:,}...")
    ppo_solve, ppo_curve = run_ppo()  # [(env_step, eval_dist)]
    ppo_plateau = float(np.mean([d for _, d in ppo_curve[-10:]]))
    print(f"  PPO steps_to_solve : {ppo_solve}   (meta: NOT solved in {PPO_BUDGET:,})")
    print(f"  PPO plateau dist   : {ppo_plateau:.4f} m (last-10 eval mean; prose ~0.13)")

    if ppo_solve is not None:
        fail.append(f"PPO unexpectedly solved at {ppo_solve} (meta: not solved in budget)")
    if not (SOLVE_DIST < ppo_plateau < RANDOM_DIST):
        fail.append(f"PPO plateau {ppo_plateau:.4f} not between solve {SOLVE_DIST} and random {RANDOM_DIST}")
    if fail:
        print("\nSTOP — PPO comparison does NOT match the chapter's measured finding:")
        for f in fail:
            print("  x " + f)
        return 1

    # env-steps-to-solve headline: SAC solved at 18000; PPO did not solve within its
    # 200k budget, so SAC is at least budget/solve = 11.1x more sample-efficient.
    speedup = round(PPO_BUDGET / steps_to_solve, 1)

    # ------------------------------------------------------------------- pack JSON
    ppo_idx = subsample_idx(len(ppo_curve), PPO_POINTS)
    data = {
        "provenance": {
            "source": "curriculum/phase2_reinforcement/ch2.2_sac/sac.py + compare_ppo_sac.py",
            "generator": "site/scripts/vizdata/ch2.2_sac.py",
            "seed": SEED,
            "device": "cpu",
            "config": "sac.py defaults (total_steps 30000, buffer 100000, autotune on); "
                      "compare_ppo_sac.py train_ppo budget 200000",
            "stack": "torch 2.10.0 / mujoco 3.10.0 / numpy 2.4.6",
            "note": "Real curves + a real recorded rollout from sac.py's OWN trained actor "
                    "and compare_ppo_sac.py's PPO reference; matches meta.yaml reference_run "
                    "(final 0.0434 m, solved <0.05 m at 18000 env steps). The comparison metric "
                    "is eval mean fingertip-to-target distance vs environment steps — the SAME "
                    "signal env_steps_to_solve is defined on. HONEST caveats: the PPO reference "
                    "is UNTUNED (a tuned PPO with reward norm / entropy bonus would do better — "
                    "the point is off-policy needs far less tuning to be sample-efficient HERE); "
                    "SAC pays MORE compute per env step (a gradient step every step); and the win "
                    "is THIS regime (dense reward + cheap env + heavy replay reuse), not always. "
                    "The rollout is the first-of-kind held-out success (deterministic, not "
                    "cherry-picked). RL evals are noisy — the last eval dist bounces around the bar.",
        },
        "solve_dist": SOLVE_DIST,
        "success_tol": float(ns["PusherReachEnv"].SUCCESS_TOL),
        "link_len": float(ns["PusherReachEnv"].LINK_LEN),
        "reach": round(2 * float(ns["PusherReachEnv"].LINK_LEN), 4),
        "baselines": {"random_dist": RANDOM_DIST, "scripted_dist": META["scripted_dist"]},
        "headline": {"speedup": speedup, "sac_solve": steps_to_solve, "ppo_budget": PPO_BUDGET},
        # THE RECORDED ROLLOUT — the trained arm reaching a held-out target
        "rollout": packed,
        # THE BARGAIN — eval distance-to-target vs env steps, SAC vs PPO
        "curves": {
            "sac": {
                "budget": int(ns["args"].total_steps),
                "solve_step": steps_to_solve,
                "final_dist": round(final_dist, 5),
                "final_return": round(final_return, 4),
                "final_success": round(final_success, 4),
                # [env_step, eval_dist, eval_return]
                "points": [[int(s), round(d, 5), round(ret, 3)] for s, ret, d in curve],
            },
            "ppo": {
                "budget": PPO_BUDGET,
                "solve_step": ppo_solve,
                "plateau_dist": round(ppo_plateau, 5),
                # [env_step, eval_dist]
                "points": [[int(ppo_curve[i][0]), round(float(ppo_curve[i][1]), 5)] for i in ppo_idx],
            },
        },
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB)")
    print(f"OK — matches meta.yaml; SAC solved at {steps_to_solve} env steps, "
          f"PPO not solved in {PPO_BUDGET:,} → SAC ~{speedup}x more sample-efficient.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
