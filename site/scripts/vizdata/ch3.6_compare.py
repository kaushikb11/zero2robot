#!/usr/bin/env python3
"""Regenerate the ch3.6 sim-to-sim concept-toy vizdata from compare.py, seed 0.

THE FULL CIRCLE, made into a browser toy: the SAME ch1.1 behavior-cloning policy
(trained in MuJoCo, reloaded not retrained) rolled out in BOTH MuJoCo (the ground
truth) and the learner's from-scratch numpy engine (ch3.3-3.5) — and the sim-to-sim
gap that opens between them. The site's SimGapToy island renders REAL rollouts and
REAL success rates from compare.py, never invented shapes.

Why we exec a PREFIX of compare.py instead of `import compare`
--------------------------------------------------------------
compare.py is a loose script (same pattern as ch3.3-3.5): importing it runs the
WHOLE pipeline — argparse, banner, the 50-episode report loop, metrics.json, the
optional .rrd. We must NOT modify compare.py. So we read its source and exec only
the PREFIX up to the `# --- region: report ---` marker in a throwaway namespace.
That gives us compare.py's OWN loaded ch1.1 policy plus its rollout functions
(`run_mujoco`, `run_engine_closed`, `replay_engine`, `pose_divergence`) and the
engine primitives (`reset_engine`, `engine_obs`, `step_engine`, `block_yaw`,
`policy_action`) VERBATIM — at the reference config (--episodes 50, seed 0, cpu),
with zero edits to the file and none of the report/metrics/rerun side effects.

The policy checkpoint
---------------------
meta.yaml's reference_run was measured with ch1.1's OWN canonical policy trained by
bc.py at 600 epochs / seed 0 on 500 scripted-expert demos. On disk that is
outputs/ch1.1-bc/bc_policy.ts.pt (its OWN bc.py eval == 31/50 == 0.62, and this
chapter's MuJoCo-side rollout reproduces that 0.62 exactly — the honesty cross-check).
compare.py's default --policy path points at a smoke checkpoint; we point at the
real one and STOP if the reproduced numbers drift from meta.yaml.

    Run:  .venv/bin/python site/scripts/vizdata/ch3.6_compare.py
    Out:  curriculum/phase3_advanced/ch3.6_compare/demo/vizdata.json
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
COMPARE_PY = REPO / "curriculum" / "phase3_advanced" / "ch3.6_compare" / "compare.py"
OUT_JSON = REPO / "curriculum" / "phase3_advanced" / "ch3.6_compare" / "demo" / "vizdata.json"
# ch1.1's OWN canonical policy behind meta.yaml's reference_run (bc.py 600ep/seed0 on
# 500 scripted demos; its own eval == 31/50 == 0.62). NOT compare.py's smoke-default path.
POLICY = REPO / "outputs" / "ch1.1-bc" / "bc_policy.ts.pt"

SEED = 0
EPISODES = 50
CUT_MARKER = "# --- region: report ---"
N_FRAMES = 48   # frames kept per rollout trace (small committed text; the toy plays them in lockstep)

# meta.yaml reference_run (seed 0, cpu; numpy 2.4.6 / torch 2.10.0 / mujoco 3.10.0)
# — the honesty gate. If the reproduced comparison drifts from these MEASURED
# numbers, STOP. These are exact on CPU (numpy engine + torch.no_grad eval + CPU
# MuJoCo are all bitwise deterministic), so the tolerances only guard float noise.
META = {
    "mj_success_rate": 0.62,
    "engine_success_rate": 0.20,
    "transfer_retained": 0.32,
    "mean_pos_divergence_m": 0.082256,
    "mean_ang_divergence_rad": 0.944621,
}
RATE_TOL = 1e-9      # the 0/1 success counts are exact
DIV_TOL = 1e-5       # the divergence means reproduce to ~1e-6

# PushT geometry (from curriculum/common/envs/pusht/pusht.xml, mirrored in
# compare.py + pusht_env.py) — the two boxes that make the T, so the toy draws a
# faithful block. Body frame: bar centered at origin, stem 0.06 m below (-y).
TEE = {
    "bar_half": [0.06, 0.015],     # bar box half-extents (x, y), m
    "stem_half": [0.015, 0.045],   # stem box half-extents (x, y), m
    "stem_offset": 0.06,           # bar-center -> stem-center along body -y, m
}
POS_TOL = 0.03   # success position tolerance (m) — the target ring the toy draws
ANG_TOL = 0.20   # success angle tolerance (rad)

Z95 = 1.959963985  # 0.975 standard-normal quantile — same constant as ch1.6 harness.py


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials — the ch1.6 idiom
    (harness.py wilson_ci), so the two success rates ship with honest error bars."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def exec_compare_prefix() -> dict:
    """Exec compare.py up to the report region, in an isolated namespace. Returns
    the populated globals (compare.py's own loaded policy + rollout/engine funcs)."""
    src = COMPARE_PY.read_text()
    prefix = src[: src.index(CUT_MARKER)]
    scratch = Path(tempfile.mkdtemp(prefix="ch3.6-viz-"))
    # argparse runs DURING exec — pin the reference config on cpu, no rerun path,
    # and the REAL trained ch1.1 checkpoint (so load_policy takes the trained path).
    old_argv = sys.argv
    sys.argv = [str(COMPARE_PY), "--policy", str(POLICY), "--seed", str(SEED),
                "--episodes", str(EPISODES), "--no-rerun", "--device", "cpu",
                "--out", str(scratch)]
    sys.path.insert(0, str(REPO))  # so curriculum.common resolves (compare.py does this too)
    ns: dict = {"__file__": str(COMPARE_PY), "__name__": "ch36_compare_vizgen"}
    try:
        exec(compile(prefix, str(COMPARE_PY), "exec"), ns)  # noqa: S102 — our own trusted source
    finally:
        sys.argv = old_argv
    return ns


def subsample(poses: np.ndarray, n_frames: int) -> np.ndarray:
    """Keep at most n_frames poses, evenly spaced, always including first + last."""
    m = len(poses)
    if m <= n_frames:
        return poses
    idx = np.unique(np.linspace(0, m - 1, n_frames).round().astype(int))
    return poses[idx]


def main() -> int:
    ns = exec_compare_prefix()
    run_mujoco = ns["run_mujoco"]
    run_engine_closed = ns["run_engine_closed"]
    replay_engine = ns["replay_engine"]
    pose_divergence = ns["pose_divergence"]
    reset_engine = ns["reset_engine"]
    engine_obs = ns["engine_obs"]
    step_engine = ns["step_engine"]
    block_yaw = ns["block_yaw"]
    policy_action = ns["policy_action"]
    horizon = ns["horizon"]
    num_episodes = ns["num_episodes"]

    def engine_closed_poses(seed: int) -> tuple[np.ndarray, bool]:
        """run_engine_closed VERBATIM, but recording the block-pose trace. Same
        step sequence -> the returned success bool is identical to run_engine_closed."""
        state = reset_engine(seed)
        poses = [np.array([*state["q"][1], block_yaw(state["q"][1], state["q"][2])])]
        success = False
        for _ in range(horizon):
            step_engine(state, policy_action(engine_obs(state)))
            poses.append(np.array([*state["q"][1], block_yaw(state["q"][1], state["q"][2])]))
            if state["success"]:
                success = True
                break
        return np.array(poses), success

    # ---- reproduce compare.py's full 50-episode report (the aggregate is the honesty gate)
    mj_successes = eng_successes = 0
    pos_divs: list[float] = []
    ang_divs: list[float] = []
    per_ep: list[dict] = []
    for ep in range(num_episodes):
        seed = 10_000 + SEED + ep          # ch1.1's held-out eval seed block (compare.py's exact rule)
        mj = run_mujoco(seed)
        eng_ok = run_engine_closed(seed)
        eng_open = replay_engine(seed, mj["actions"])
        pos_div, ang_div = pose_divergence(mj["poses"], eng_open)
        mj_successes += int(mj["success"])
        eng_successes += int(eng_ok)
        pos_divs.append(pos_div)
        ang_divs.append(ang_div)
        per_ep.append({"ep": ep, "seed": seed, "mj": bool(mj["success"]),
                       "eng": bool(eng_ok), "poses": mj["poses"]})

    mj_rate = mj_successes / num_episodes
    eng_rate = eng_successes / num_episodes
    mean_pos = float(np.mean(pos_divs))
    mean_ang = float(np.mean(ang_divs))
    transfer = eng_rate / mj_rate if mj_rate > 0 else float("nan")

    print("reproduced comparison [ch1.1 policy in both sims, seed 0, cpu, 50 ep] vs meta.yaml:")
    print(f"  MuJoCo  BC success : {mj_rate:.3f}  ({mj_successes}/{num_episodes})   (meta 0.62; == bc.py's own ch1.1 eval)")
    print(f"  engine  BC success : {eng_rate:.3f}  ({eng_successes}/{num_episodes})   (meta 0.20 — the SAME policy, physics changed)")
    print(f"  transfer retained  : {transfer:.3f}                (meta 0.32)")
    print(f"  mean pos divergence: {mean_pos:.6f} m       (meta 0.082256)")
    print(f"  mean ang divergence: {mean_ang:.6f} rad     (meta 0.944621 — the ANGLE gap dominates)")

    fail = []
    if abs(mj_rate - META["mj_success_rate"]) > RATE_TOL:
        fail.append(f"mj_rate {mj_rate} != {META['mj_success_rate']}")
    if abs(eng_rate - META["engine_success_rate"]) > RATE_TOL:
        fail.append(f"engine_rate {eng_rate} != {META['engine_success_rate']}")
    if abs(round(transfer, 2) - META["transfer_retained"]) > 0.011:
        fail.append(f"transfer {transfer:.3f} != {META['transfer_retained']}")
    if abs(mean_pos - META["mean_pos_divergence_m"]) > DIV_TOL:
        fail.append(f"mean_pos {mean_pos:.6f} != {META['mean_pos_divergence_m']}")
    if abs(mean_ang - META["mean_ang_divergence_rad"]) > DIV_TOL:
        fail.append(f"mean_ang {mean_ang:.6f} != {META['mean_ang_divergence_rad']}")
    if fail:
        print("\nSTOP — reproduced comparison does NOT match meta.yaml:")
        for f in fail:
            print("  x " + f)
        return 1

    # ---- pick the featured side-by-side rollouts, DETERMINISTICALLY (first-of-kind),
    #      so nothing is cherry-picked: the first mj-success/engine-fail episode (the
    #      headline: same policy, only the physics changed), the first both-succeed
    #      (the engine PARTLY works — 18%), and the first both-fail (the policy is
    #      imperfect in both — honest). Each carries its seed + why in provenance.
    def first(pred) -> dict:
        return next(e for e in per_ep if pred(e))

    picks = [
        ("headline", first(lambda e: e["mj"] and not e["eng"]),
         "MuJoCo succeeds, your engine does not — the SAME policy, only the physics changed"),
        ("both", first(lambda e: e["mj"] and e["eng"]),
         "both sims succeed — your engine PARTLY works (~1 in 5 episodes transfer)"),
        ("neither", first(lambda e: not e["mj"] and not e["eng"]),
         "neither sim succeeds — the BC policy is imperfect in both, not just yours"),
    ]

    rollouts = []
    for key, e, why in picks:
        eng_poses, eng_ok = engine_closed_poses(e["seed"])
        assert eng_ok == e["eng"], f"engine-closed success mismatch at seed {e['seed']}"
        mj_sub = subsample(np.asarray(e["poses"]), N_FRAMES)
        eng_sub = subsample(eng_poses, N_FRAMES)

        def pack(a: np.ndarray) -> list:
            return [[round(float(x), 4), round(float(y), 4), round(float(yaw), 4)] for x, y, yaw in a]

        rollouts.append({
            "key": key,
            "seed": e["seed"],
            "why": why,
            "mj_success": e["mj"],
            "engine_success": e["eng"],
            "mj_steps": int(len(e["poses"]) - 1),
            "engine_steps": int(len(eng_poses) - 1),
            "mj": pack(mj_sub),
            "engine": pack(eng_sub),
        })

    # ---- SVG world half-extent: hold every stored block position (+ block reach) with margin
    extent = POS_TOL
    for r in rollouts:
        for series in ("mj", "engine"):
            for x, y, _ in r[series]:
                extent = max(extent, abs(x), abs(y))
    world_half = round((extent + TEE["stem_offset"] + TEE["bar_half"][0]) * 1.06, 3)

    mj_ci = wilson_ci(mj_successes, num_episodes)
    eng_ci = wilson_ci(eng_successes, num_episodes)

    data = {
        "provenance": {
            "source": "curriculum/phase3_advanced/ch3.6_compare/compare.py",
            "generator": "site/scripts/vizdata/ch3.6_compare.py",
            "policy": "outputs/ch1.1-bc/bc_policy.ts.pt (ch1.1's OWN canonical policy, bc.py 600 epochs, seed 0, 500 scripted demos; its own eval == 31/50 == 0.62)",
            "seed": SEED,
            "device": "cpu",
            "config": f"--episodes {EPISODES} --pusher_mass 4.0 --block_damp 6.0 --baumgarte 40.0 (horizon {horizon})",
            "stack": "numpy 2.4.6 / torch 2.10.0 / mujoco 3.10.0",
            "note": "The SAME ch1.1 BC policy in BOTH sims, closed-loop. MuJoCo is the "
                    "ground truth (its success reproduces ch1.1's own bc.py eval, 31/50 == 0.62); the "
                    "learner's from-scratch two-point-mass engine keeps only ~1/3 of that "
                    "(0.20). Matches meta.yaml reference_run. Rollouts are first-of-kind picks "
                    "(deterministic, not cherry-picked).",
        },
        "episodes": num_episodes,
        "world_half": world_half,
        "pos_tol": POS_TOL,
        "ang_tol": ANG_TOL,
        "tee": TEE,
        # --- TRANSFER: the two success rates with ch1.6-style Wilson error bars (the headline) ---
        "transfer": {
            "mujoco": {"k": mj_successes, "n": num_episodes, "rate": round(mj_rate, 4),
                       "ci": [round(mj_ci[0], 4), round(mj_ci[1], 4)]},
            "engine": {"k": eng_successes, "n": num_episodes, "rate": round(eng_rate, 4),
                       "ci": [round(eng_ci[0], 4), round(eng_ci[1], 4)]},
            "retained": round(transfer, 4),
        },
        # --- DIVERGENCE: the open-loop dynamics gap (the WHY: the ANGLE gap dominates) ---
        "divergence": {
            "mean_pos_m": round(mean_pos, 6),
            "mean_ang_rad": round(mean_ang, 6),
        },
        # --- the side-by-side rollouts (recorded block-pose traces, subsampled) ---
        "rollouts": rollouts,
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB, {len(rollouts)} rollouts, <= {N_FRAMES} frames/trace)")
    print(f"  MuJoCo {mj_rate:.2f} [{mj_ci[0]:.2f},{mj_ci[1]:.2f}]  vs  engine {eng_rate:.2f} "
          f"[{eng_ci[0]:.2f},{eng_ci[1]:.2f}]  (Wilson 95%, n={num_episodes})")
    print("OK — matches meta.yaml; the MuJoCo side reproduces ch1.1's 0.62, the sim-to-sim gap is real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
