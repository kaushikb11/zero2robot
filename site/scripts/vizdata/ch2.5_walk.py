#!/usr/bin/env python3
"""Regenerate the ch2.5 locomotion concept-toy vizdata from walk.py, seed 0.

The site's QuadrupedWalkToy island renders a REAL emergent gait — never an
invented shape. This generator REUSES walk.py's OWN training loop (the SAC actor,
`evaluate`, `rollout`, `trot_action`) so the browser panels are faithful to the
chapter artifact, then dumps a small JSON the island loads:

  1. a recorded side-view GAIT rollout (the deterministic hero episode, seed
     500000 = walk.py's own `rollout(gait_env, 500_000 + seed, use_actor=True)`)
     — torso pose + leg joints + foot contacts per frame, so the toy replays the
     quadruped WALKING and the torso travelling forward,
  2. the FORWARD-PROGRESS curve over training (walk.py's `curve`: eval forward
     distance climbing off the "just stand" floor toward / past the scripted-trot
     line — the walk taking shape),
  3. the honest FREE-TIER CEILING numbers (mean eval forward dist / return /
     length vs the scripted trot) so the toy tells the truth: the emergent gait
     travels FURTHER than the trot but falls before the horizon, so its RETURN
     stays below the trot's — emergent != robust at a CPU-laptop budget.

Why we exec a PREFIX of walk.py instead of `import walk`
--------------------------------------------------------
walk.py is a loose script (no `if __name__ == "__main__"` guard): importing it
runs the WHOLE thing — argparse, the full 60k-step SAC training, the report,
metrics.json, a rerun recording. We must NOT modify walk.py (it is LOC-capped).
So we read its source and exec only the prefix up to the `# --- region: report ---`
line — setup + env + model + replay + update + eval + TRAIN — in a throwaway
namespace. That runs the real seed-0 training and hands us walk.py's OWN trained
`actor`, `evaluate`, `rollout`, `trot_action`, `curve`, at the default config
(seed 0, cpu, torch/mujoco/numpy pinned in pyproject), with zero edits.

Determinism: torch/mujoco/numpy versions are pinned and match meta.yaml's
reference_run provenance (torch 2.10.0 / mujoco 3.10.0 / numpy 2.4.6); CPU MuJoCo
+ seeded torch reproduce the reference metrics. The script STOPS if the
regenerated run drifts from meta.yaml's reference_run.

    Run:    .venv/bin/python site/scripts/vizdata/ch2.5_walk.py         (full 60k)
    Dev:    .venv/bin/python site/scripts/vizdata/ch2.5_walk.py --smoke (plumbing only, NO meta gate)
    Out:    curriculum/phase2_reinforcement/ch2.5_walk/demo/vizdata.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
WALK_PY = REPO / "curriculum" / "phase2_reinforcement" / "ch2.5_walk" / "walk.py"
OUT_JSON = REPO / "curriculum" / "phase2_reinforcement" / "ch2.5_walk" / "demo" / "vizdata.json"

SEED = 0
TOTAL_STEPS = 60_000       # meta.yaml reference_run: seed 0, 60k steps
CUT_MARKER = "# --- region: report ---"
LEGS = ("FL", "FR", "HL", "HR")     # FL/HL are +y (far side), FR/HR are -y (near side)
GAIT_FRAMES = 96           # subsampled frames kept for the replay (small text)
FOOT_CONTACT_Z = 0.030     # foot sphere (r=0.022) centre this low => in stance

# meta.yaml reference_run (seed 0, cpu; torch 2.10.0 / mujoco 3.10.0 / numpy 2.4.6)
# — the honesty gate. If the regenerated run drifts from these, STOP.
META = {
    "mean_eval_forward_dist": 3.0246,
    "mean_eval_return": 247.22,
    "mean_eval_length": 249.1,
    "scripted_trot_forward_dist": 2.1536,
}
# Non-learned reference distances from the env (README / walk.py header): the
# "just stand" floor and random flailing — the two rungs the emergent gait clears.
STAND_DIST = -0.01
RANDOM_DIST = -0.30
# tolerances: versions match meta, so CPU reproduction is near-exact; these guard
# only last-digit / thread-order float noise, not a real regression.
TOL = {"dist": 0.06, "return": 6.0, "length": 6.0, "trot": 0.06}


def exec_walk_prefix(smoke: bool) -> dict:
    """Exec walk.py up to the report region, in an isolated namespace. Runs the
    REAL seed-0 SAC training and returns the populated globals (walk.py's own
    trained actor + eval/rollout functions + the training curve)."""
    src = WALK_PY.read_text()
    prefix = src[: src.index(CUT_MARKER)]

    scratch = Path(tempfile.mkdtemp(prefix="ch2.5-viz-"))
    old_argv = sys.argv
    if smoke:  # plumbing check only: walk.py --smoke pins tiny steps (NO meta gate)
        sys.argv = [str(WALK_PY), "--smoke", "--seed", str(SEED), "--no-rerun", "--out", str(scratch)]
    else:      # the reference run: default config, cpu, 60k steps
        sys.argv = [str(WALK_PY), "--seed", str(SEED), "--total_steps", str(TOTAL_STEPS),
                    "--no-rerun", "--device", "cpu", "--out", str(scratch)]
    ns: dict = {"__file__": str(WALK_PY), "__name__": "walk_toy_vizgen"}
    try:
        exec(compile(prefix, str(WALK_PY), "exec"), ns)  # noqa: S102 — our own trusted source
    finally:
        sys.argv = old_argv
    return ns


def capture_gait(ns: dict, seed: int) -> dict:
    """Roll out ONE deterministic hero episode (the actor's mean action), mirroring
    walk.py's `rollout(gait_env, 500_000 + seed, use_actor=True)` EXACTLY, and
    additionally record per-frame side-view geometry (torso pose, leg joints, foot
    contacts) that walk.py's scalar rollout throws away. The eval policy is
    deterministic (tanh of the mean), so the recorded motion is faithful."""
    import torch

    QuadrupedEnv = ns["QuadrupedEnv"]
    actor = ns["actor"]
    device = ns["device"]
    blind = ns["_blind"]
    default_torso_mass = ns["default_torso_mass"]

    env = QuadrupedEnv()
    model, data = env.model, env.data
    torso_bid = model.body("torso").id
    model.body_mass[torso_bid] = default_torso_mass  # nominal body (eval), as walk.py does

    obs = blind(env.reset(seed=seed))
    x0 = float(data.qpos[env._root_qadr])

    def snapshot() -> dict:
        # side view = sagittal x-z plane. torso forward axis (world) for pitch.
        xmat = data.xmat[torso_bid].reshape(3, 3)
        fx, _, fz = xmat[:, 0]                          # torso body x-axis in world
        frame = {
            "t": round(float(data.time), 3),
            "torso": [round(float(data.qpos[env._root_qadr]), 3),      # world x
                      round(float(data.qpos[env._root_qadr + 2]), 3),  # world z (height)
                      round(float(np.arctan2(fz, fx)), 4)],            # pitch (rad)
            "vx": round(float(env.forward_vel), 3),
            "legs": {}, "contact": [],
        }
        for leg in LEGS:
            hip = data.body(f"{leg}_thigh").xpos
            knee = data.body(f"{leg}_shin").xpos
            foot = data.geom(f"{leg}_foot").xpos
            frame["legs"][leg] = [
                [round(float(hip[0]), 3), round(float(hip[2]), 3)],
                [round(float(knee[0]), 3), round(float(knee[2]), 3)],
                [round(float(foot[0]), 3), round(float(foot[2]), 3)],
            ]
            frame["contact"].append(bool(float(foot[2]) < FOOT_CONTACT_Z))
        return frame

    frames = [snapshot()]
    done, ret, vxs, steps = False, 0.0, [], 0
    while not done:
        with torch.no_grad():
            _, _, mean_action = actor.sample(
                torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
        action = mean_action[0].cpu().numpy()
        obs, reward, done, info = env.step(action)
        obs = blind(obs)
        ret += reward
        vxs.append(info["forward_vel"])
        steps += 1
        frames.append(snapshot())

    forward = float(data.qpos[env._root_qadr]) - x0

    # Faithfulness guard: our capture loop's scalars must equal walk.py's own
    # deterministic rollout() for the same seed (both use the actor's mean action).
    ref = ns["rollout"](QuadrupedEnv(), seed, True)  # (return, forward, vel, length)
    assert abs(ret - ref[0]) < 1e-4 and abs(forward - ref[1]) < 1e-5 and int(steps) == int(ref[3]), \
        f"capture diverged from walk.py rollout(): {(ret, forward, steps)} vs {ref}"

    return {"return": ret, "forward": forward, "length": steps,
            "mean_vx": float(np.mean(vxs)), "frames": frames}


def subsample(frames: list, keep: int) -> list:
    if len(frames) <= keep:
        return frames
    idx = np.linspace(0, len(frames) - 1, keep).round().astype(int)
    return [frames[i] for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="plumbing check: run walk.py --smoke (tiny), SKIP the meta gate")
    a = ap.parse_args()

    ns = exec_walk_prefix(a.smoke)
    args = ns["args"]
    curve = ns["curve"]                # [(env_step, eval_return, eval_forward_dist), ...]
    walks_at = ns["walks_at"]
    QuadrupedEnv = ns["QuadrupedEnv"]

    # ---- reproduce walk.py's report metrics (means over the held-out eval seeds)
    final_return, final_dist, final_vx, final_len = ns["evaluate"](args.eval_episodes)
    bar_env = QuadrupedEnv()
    trot_rows = [ns["rollout"](bar_env, 500_000 + args.seed + ep, False)
                 for ep in range(args.eval_episodes)]
    trot_dist = float(np.mean([r[1] for r in trot_rows]))
    trot_return = float(np.mean([r[0] for r in trot_rows]))

    # ---- the deterministic hero gait for the replay panel (seed 500000)
    gait = capture_gait(ns, 500_000 + args.seed)

    print(f"regenerated walk [seed {SEED}, cpu, {args.total_steps} steps] vs meta reference_run:")
    print(f"  mean_eval_forward_dist   : {final_dist:+.4f}   (meta {META['mean_eval_forward_dist']})")
    print(f"  mean_eval_return         : {final_return:.4f}   (meta {META['mean_eval_return']})")
    print(f"  mean_eval_length         : {final_len:.2f}     (meta {META['mean_eval_length']})")
    print(f"  scripted_trot_forward_dist: {trot_dist:+.4f}   (meta {META['scripted_trot_forward_dist']})")
    print(f"  scripted_trot_return     : {trot_return:.2f}")
    print(f"  hero gait (seed 500000)  : forward {gait['forward']:+.3f} m  len {gait['length']}  return {gait['return']:.1f}")
    print(f"  walks_at_env_steps       : {walks_at}")

    # ---------------------------------------------------------------- honesty gate
    if a.smoke:
        print("\n[smoke] plumbing OK — meta gate SKIPPED (smoke run does not emerge a gait).")
    else:
        fail = []
        if abs(final_dist - META["mean_eval_forward_dist"]) > TOL["dist"]:
            fail.append(f"mean_eval_forward_dist {final_dist} != {META['mean_eval_forward_dist']}")
        if abs(final_return - META["mean_eval_return"]) > TOL["return"]:
            fail.append(f"mean_eval_return {final_return} != {META['mean_eval_return']}")
        if abs(final_len - META["mean_eval_length"]) > TOL["length"]:
            fail.append(f"mean_eval_length {final_len} != {META['mean_eval_length']}")
        if abs(trot_dist - META["scripted_trot_forward_dist"]) > TOL["trot"]:
            fail.append(f"scripted_trot_forward_dist {trot_dist} != {META['scripted_trot_forward_dist']}")
        # the chapter's honest headline: emergent gait BEATS the trot on distance,
        # but its RETURN sits BELOW the trot's (it falls before the horizon).
        if not (final_dist > trot_dist):
            fail.append(f"emergent gait should beat the trot on distance: {final_dist} vs {trot_dist}")
        if not (final_return < trot_return):
            fail.append(f"emergent gait return should sit below the trot's: {final_return} vs {trot_return}")
        if not (final_len < QuadrupedEnv.MAX_STEPS):
            fail.append(f"emergent gait should fall before the {QuadrupedEnv.MAX_STEPS}-step horizon: len {final_len}")
        if fail:
            print("\nSTOP — regenerated walk does NOT match meta.yaml:")
            for f in fail:
                print("  x " + f)
            return 1

    # ------------------------------------------------------------------ pack json
    gait_frames = subsample(gait["frames"], GAIT_FRAMES)
    data = {
        "provenance": {
            "source": "curriculum/phase2_reinforcement/ch2.5_walk/walk.py",
            "generator": "site/scripts/vizdata/ch2.5_walk.py",
            "seed": SEED,
            "device": "cpu",
            "total_steps": int(args.total_steps),
            "config": f"default (seed {SEED}, domain_rand on, cpu, {args.total_steps} steps)",
            "stack": "torch 2.10.0 / mujoco 3.10.0 / numpy 2.4.6",
            "smoke": bool(a.smoke),
            "note": "Real emergent gait from walk.py's own SAC actor (nobody scripts "
                    "the gait). The hero rollout is the deterministic eval episode "
                    "seed 500000 = walk.py's own gait recording. Matches meta.yaml "
                    "reference_run. HONEST: the emergent gait travels further than the "
                    "scripted trot but falls before the horizon, so its return stays "
                    "below the trot's — a full robust trot is the Scale Lab.",
        },
        # ---- geometry the side-view renderer needs (from quadruped.xml / env) ----
        "geometry": {
            "torso_half": [0.18, 0.035],   # x, z half-sizes of the torso box (side view)
            "foot_radius": 0.022,
            "floor_z": 0.0,
            "stand_height": QuadrupedEnv.STAND_HEIGHT,
            "target_height": QuadrupedEnv.TARGET_HEIGHT,
            "fall_height": QuadrupedEnv.FALL_HEIGHT,
            "legs": list(LEGS),
            "near_legs": ["FR", "HR"],     # -y side: drawn solid
            "far_legs": ["FL", "HL"],      # +y side: drawn faded (depth cue)
        },
        # ---- panel 1: the recorded hero gait (side view, camera follows torso x) --
        "gait": {
            "seed": 500_000 + args.seed,
            "forward": round(gait["forward"], 4),
            "length": int(gait["length"]),
            "return": round(gait["return"], 4),
            "mean_vx": round(gait["mean_vx"], 4),
            "frames": gait_frames,
        },
        # ---- panel 2: the forward-progress-over-training curve (the walk emerging)-
        "curve": [[int(s), round(float(r), 3), round(float(d), 4)] for (s, r, d) in curve],
        "walks_at_env_steps": walks_at,
        # ---- panel 3: the honest free-tier ceiling numbers -----------------------
        "summary": {
            "mean_eval_forward_dist": round(final_dist, 4),
            "mean_eval_forward_vel": round(final_vx, 4),
            "mean_eval_return": round(final_return, 4),
            "mean_eval_length": round(final_len, 2),
            "scripted_trot_forward_dist": round(trot_dist, 4),
            "scripted_trot_return": round(trot_return, 4),
            "stand_forward_dist": STAND_DIST,
            "random_forward_dist": RANDOM_DIST,
            "horizon_steps": QuadrupedEnv.MAX_STEPS,
        },
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB)")
    if a.smoke:
        print("OK — [smoke] plumbing verified (data is a non-emergent tiny run; DO NOT commit).")
    else:
        print("OK — matches meta.yaml; emergent gait beats the trot on distance, falls early (return below).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
