#!/usr/bin/env python3
"""Regenerate the ch2.4 reward-hacking concept-toy vizdata from rewards.py, seed 0.

The site's RewardHackToy island is the MAP'S NAMED HERO for ch2.4 (specification
gaming). It renders REAL, recorded quadruped rollouts + the REAL hack self-reward
rise-curve, never an invented shape. This generator REUSES rewards.py's OWN pieces
(its `train` / `evaluate` PPO, the four reward programs, the QuadrupedEnv) so the
panels are faithful to the chapter artifact, then dumps a small JSON the island loads.

Why we exec a PREFIX of rewards.py instead of `import rewards`
-------------------------------------------------------------
rewards.py is a loose script (no `if __name__ == "__main__"` guard): importing it
runs the WHOLE comparison — argparse, four full trainings, metrics.json, rerun. We
must NOT modify rewards.py (it is LOC-capped, human-owned). So we read its source
and exec only the prefix up to the `# --- region: demos ---` line — i.e. setup +
the reward programs + envs + model + ppo + train + eval — in a throwaway namespace.
That hands us rewards.py's OWN `train`, `evaluate`, `REWARD_DESIGNS`, `QuadrupedEnv`,
`Agent`, at the DEFAULT config (seed 0, cpu), with zero edits.

THE HONESTY MODEL (why we EMBED meta numbers, not recompute magnitudes)
-----------------------------------------------------------------------
Root CLAUDE.md invariant 2 + this env's README: env resets are BITWISE-reproducible
on CPU, but PPO *training* is only STATISTICALLY reproducible (torch runs under
use_deterministic_algorithms(warn_only=True); non-deterministic ops fall back, so
absolute forward distance / return drift machine-to-machine). meta.yaml says so in
as many words: "The RELIABLE, seed-robust signals ... are ORDERINGS ... never
absolute magnitudes." A fresh seed-0 shaped run here walks ~+6 m, not meta's +4.61 —
same qualitative walk, different magnitude. So:

  * Every DISPLAYED number in the toy is meta.yaml's cited reference_run (shaped
    +4.61 m, hack -0.181 m / height 0.277 / design_return 105 -> 1048, sparse +0.79,
    curriculum +7.84) — the chapter's recorded, provenance-carrying measured values.
  * The toy's GEOMETRY (side-by-side rollouts) + the reward RISE-CURVE SHAPE come
    from a REAL seed-0 run in THIS process (real physics, real PPO history).
  * The STOP gate asserts only the SEED-ROBUST qualitative signatures reproduce —
    exactly the checks meta.yaml's exercise_checks define (ex1: hack |forward| small,
    hack design_return rises >=2x; ordering shaped forward >> sparse forward; the
    hack rears TALLER than the shaped walker). If those break, STOP.

    Run:  .venv/bin/python site/scripts/vizdata/ch2.4_rewards.py
    Out:  curriculum/phase2_reinforcement/ch2.4_rewards/demo/vizdata.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
REWARDS_PY = REPO / "curriculum" / "phase2_reinforcement" / "ch2.4_rewards" / "rewards.py"
OUT_JSON = REPO / "curriculum" / "phase2_reinforcement" / "ch2.4_rewards" / "demo" / "vizdata.json"

SEED = 0
CUT_MARKER = "# --- region: demos ---"

# meta.yaml reference_run (seed 0, cpu, torch 2.10.0 / mujoco 3.10.0 / numpy 2.4.6;
# measured 2026-07-06) — the recorded, cited numbers the toy DISPLAYS. Training is
# not bitwise-reproducible (see the module docstring), so these are embedded, not
# recomputed; a fresh seed-0 run reproduces the SIGNATURES, not the magnitudes.
META = {
    "shaped_forward_m": 4.61,
    "sparse_forward_m": 0.79,
    "hack_forward_m": -0.181,
    "hack_height_m": 0.277,
    "hack_return_first": 105.0,
    "hack_return_last": 1048.0,
    "curriculum_forward_m": 7.84,
    # ex1's affordable 150k budget (meta.yaml exercise_checks.ex1.provenance): the
    # honest note that even a small budget shows the hack — +0.41 m, a fraction of
    # the shaped +4.61 m walk, while its self-reward rose 128 -> 910 (7x).
    "ex1_hack_forward_m": 0.41,
    "ex1_hack_return_first": 128.0,
    "ex1_hack_return_last": 910.0,
}
# The seed-ROBUST gate thresholds, taken verbatim from meta.yaml exercise_checks.
HACK_FORWARD_ABS_MAX = 1.0   # ex1.hack_forward_abs_max — the hack does NOT walk
HACK_RETURN_RISE_MIN = 2.0   # ex1.hack_return_rise_min — its own reward IS optimized

# Rollout capture: how many held-out eval episodes to roll, and how many frames to
# keep per shown rollout (subsampled — small text, no binary).
EVAL_EPISODES = 10
ROLLOUT_FRAMES = 56
RETURN_CURVE_POINTS = 80

# Side-view legs we draw (the sagittal x-z silhouette): one FRONT leg + one HIND
# leg. FL/FR overlap in side view, as do HL/HR, so a front+hind pair reads as a
# clean 2-legged profile driven by the real recorded joint geometry.
FRONT_LEG, HIND_LEG = "FL", "HL"


def exec_rewards_prefix(total_steps: int, design: str) -> dict:
    """Exec rewards.py up to the demos region, in an isolated namespace, pinned to
    the given budget/design on cpu with no rerun. Returns the populated globals."""
    src = REWARDS_PY.read_text()
    cut = src.index(CUT_MARKER)
    prefix = src[:cut]

    scratch = Path(tempfile.mkdtemp(prefix="ch2.4-viz-"))
    old_argv = sys.argv
    sys.argv = [str(REWARDS_PY), "--seed", str(SEED), "--no-rerun", "--device", "cpu",
                "--design", design, "--total_steps", str(total_steps),
                "--eval_episodes", str(EVAL_EPISODES), "--out", str(scratch)]
    ns: dict = {"__file__": str(REWARDS_PY), "__name__": "rewards_toy_vizgen"}
    try:
        exec(compile(prefix, str(REWARDS_PY), "exec"), ns)  # noqa: S102 — our own trusted source
    finally:
        sys.argv = old_argv
    return ns


def record_rollout(ns: dict, agent, episode: int) -> dict:
    """Roll out the trained policy MEAN (no sampling) on a held-out seed, EXACTLY as
    rewards.py's evaluate() does (seed 500000 + SEED + episode), additionally
    recording the side-view body geometry each step. Real physics, real policy."""
    QuadrupedEnv = ns["QuadrupedEnv"]
    device = ns["device"]
    env = QuadrupedEnv()
    obs = env.reset(seed=500_000 + SEED + episode)
    root = env._root_qadr
    x0 = float(env.data.qpos[root])

    def xz(p):
        return [float(p[0]), float(p[2])]

    frames, heights = [], []
    done = False
    while not done:
        with torch.no_grad():
            mean = agent.actor_mean(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
        action = mean[0].cpu().numpy()
        # capture BEFORE stepping so frame 0 is the shared start pose
        torso = env.data.body("torso")
        m = np.asarray(torso.xmat).reshape(3, 3)
        pitch = float(np.arctan2(m[2, 0], m[0, 0]))  # body-x tilt in the x-z plane
        cx, cz = xz(torso.xpos)
        leg = {}
        for tag, name in (("front", FRONT_LEG), ("hind", HIND_LEG)):
            leg[tag] = [
                xz(env.data.body(f"{name}_thigh").xpos),  # hip pivot
                xz(env.data.body(f"{name}_shin").xpos),   # knee pivot
                xz(env.data.geom(f"{name}_foot").xpos),   # foot
            ]
        frames.append({"cx": cx, "cz": cz, "pitch": pitch,
                       "front": leg["front"], "hind": leg["hind"]})
        heights.append(env.torso_height)
        obs, _r, done, _info = env.step(action)
    forward = float(env.data.qpos[root]) - x0
    return {"frames": frames, "forward": forward, "x0": x0,
            "mean_height": float(np.mean(heights)), "steps": len(frames)}


def subsample(seq: list, n: int) -> list:
    if len(seq) <= n:
        return seq
    idx = np.linspace(0, len(seq) - 1, n).round().astype(int)
    return [seq[i] for i in idx]


def round_frame(f: dict) -> dict:
    r = lambda p: [round(p[0], 3), round(p[1], 3)]  # noqa: E731
    return {
        "c": [round(f["cx"], 3), round(f["cz"], 3), round(f["pitch"], 3)],
        "f": [r(f["front"][0]), r(f["front"][1]), r(f["front"][2])],
        "h": [r(f["hind"][0]), r(f["hind"][1]), r(f["hind"][2])],
    }


def train_design(design: str, total_steps: int = 400_000):
    """Train one reward program at seed 0 and return (agent, history, ns, iters)."""
    ns = exec_rewards_prefix(total_steps, design)
    reward_fn = ns["REWARD_DESIGNS"][design]
    print(f"  training '{design}' ({ns['num_iterations']} iters x {ns['batch_size']} steps)...")
    agent, history = ns["train"](reward_fn)
    return agent, history, ns


def main() -> int:
    torch.set_num_threads(1)  # steadier, faster tiny-MLP training

    # -------------------------------------------------------- HACK: the hero design
    # Train once at the full 400k budget. Keep its REAL design_return history (the
    # rise-curve shape) + a representative rollout (rears tall, goes nowhere).
    agent_hack, hist_hack, ns_hack = train_design("hack")
    hack_return = [v for v in hist_hack["design_return"] if v == v]  # drop early nan
    hack_rise = (hack_return[-1] / hack_return[0]) if hack_return and hack_return[0] > 0 else float("nan")
    hack_rolls = [record_rollout(ns_hack, agent_hack, e) for e in range(EVAL_EPISODES)]
    hack_fwd_mean = float(np.mean([r["forward"] for r in hack_rolls]))
    hack_h_mean = float(np.mean([r["mean_height"] for r in hack_rolls]))
    # representative episode: forward closest to meta's cited mean (near zero)
    hack_pick = min(hack_rolls, key=lambda r: abs(r["forward"] - META["hack_forward_m"]))

    # ----------------------------------------------------- SHAPED: the forward walk
    agent_shaped, _hist_shaped, ns_shaped = train_design("shaped")
    shaped_rolls = [record_rollout(ns_shaped, agent_shaped, e) for e in range(EVAL_EPISODES)]
    shaped_fwd_mean = float(np.mean([r["forward"] for r in shaped_rolls]))
    shaped_h_mean = float(np.mean([r["mean_height"] for r in shaped_rolls]))
    shaped_pick = min(shaped_rolls, key=lambda r: abs(r["forward"] - META["shaped_forward_m"]))

    # ------------------------------------------- SPARSE: barely moves (the ordering)
    agent_sparse, _h, ns_sparse = train_design("sparse")
    sparse_fwd_mean = float(np.mean([record_rollout(ns_sparse, agent_sparse, e)["forward"]
                                     for e in range(EVAL_EPISODES)]))

    # ================================================================= honesty gate
    print("\nregenerated seed-0 run vs meta.yaml (SIGNATURES, not magnitudes):")
    print(f"  hack   forward {hack_fwd_mean:+.3f} m  height {hack_h_mean:.3f}  "
          f"design_return {hack_return[0]:.1f} -> {hack_return[-1]:.1f}  (rise {hack_rise:.1f}x)")
    print(f"  shaped forward {shaped_fwd_mean:+.3f} m  height {shaped_h_mean:.3f}")
    print(f"  sparse forward {sparse_fwd_mean:+.3f} m")
    print("  meta cites: shaped +4.61 / sparse +0.79 / hack -0.181 (height 0.277, "
          "return 105->1048)")

    fail = []
    if abs(hack_fwd_mean) > HACK_FORWARD_ABS_MAX:
        fail.append(f"hack forward {hack_fwd_mean:+.3f} not small (|.|<= {HACK_FORWARD_ABS_MAX}): it should NOT walk")
    if not (hack_rise >= HACK_RETURN_RISE_MIN):
        fail.append(f"hack design_return rise {hack_rise:.2f}x < {HACK_RETURN_RISE_MIN}x: the reward is not being optimized")
    if not (shaped_fwd_mean > sparse_fwd_mean):
        fail.append(f"ordering broken: shaped forward {shaped_fwd_mean:+.3f} not > sparse {sparse_fwd_mean:+.3f}")
    if not (shaped_fwd_mean > 1.0):
        fail.append(f"shaped did not learn a walk: forward {shaped_fwd_mean:+.3f} <= 1 m")
    if not (abs(hack_fwd_mean) < 0.2 * shaped_fwd_mean):
        fail.append(f"hack forward {hack_fwd_mean:+.3f} not a small fraction of the shaped walk {shaped_fwd_mean:+.3f}")
    if not (hack_h_mean > shaped_h_mean):
        fail.append(f"hack should rear TALLER than the shaped walker: hack h {hack_h_mean:.3f} !> shaped {shaped_h_mean:.3f}")
    if fail:
        print("\nSTOP — regenerated run does NOT reproduce meta.yaml's signatures:")
        for f in fail:
            print("  x " + f)
        return 1

    # ------------------------------------------------------------------- pack curves
    # The hack self-reward rise-curve (REAL per-iteration history, subsampled). The
    # x-axis is training progress in [0,1]; the DISPLAYED endpoints are meta's cited
    # 105 -> 1048, so we present the real SHAPE renormalized onto the cited band.
    n = len(hack_return)
    ci = np.linspace(0, n - 1, min(RETURN_CURVE_POINTS, n)).round().astype(int)
    lo, hi = hack_return[0], hack_return[-1]
    span = (hi - lo) if hi > lo else 1.0
    curve = []
    for i in ci:
        frac = i / max(1, n - 1)
        # map the real curve's value onto meta's [first,last] band (shape preserved)
        norm = (hack_return[i] - lo) / span
        val = META["hack_return_first"] + norm * (META["hack_return_last"] - META["hack_return_first"])
        curve.append([round(float(frac), 4), round(float(val), 2)])

    def pack_roll(pick: dict) -> dict:
        frames = subsample(pick["frames"], ROLLOUT_FRAMES)
        return {
            "x0": round(pick["x0"], 4),
            "forward": round(pick["forward"], 4),   # THIS episode's own distance (provenance)
            "steps": pick["steps"],
            "frames": [round_frame(f) for f in frames],
        }

    data = {
        "provenance": {
            "source": "curriculum/phase2_reinforcement/ch2.4_rewards/rewards.py",
            "env": "curriculum/common/envs/quadruped/quadruped_env.py",
            "generator": "site/scripts/vizdata/ch2.4_rewards.py",
            "seed": SEED,
            "device": "cpu",
            "stack": "torch 2.10.0, mujoco 3.10.0, numpy 2.4.6",
            "displayed_numbers": "meta.yaml reference_run (measured 2026-07-06). Env "
                "resets are bitwise-reproducible on CPU; PPO TRAINING is only "
                "statistically reproducible (torch use_deterministic_algorithms "
                "warn_only), so absolute forward/return drift run-to-run. The toy "
                "DISPLAYS meta's cited magnitudes; the rollout GEOMETRY + reward "
                "rise-curve SHAPE are a real seed-0 run in this process, verified to "
                "reproduce meta's seed-robust SIGNATURES (hack |forward| small, hack "
                "design_return rises >=2x, shaped walk >> sparse, hack rears taller).",
            "this_run": {
                "shaped_forward_mean_m": round(shaped_fwd_mean, 3),
                "hack_forward_mean_m": round(hack_fwd_mean, 3),
                "hack_height_mean_m": round(hack_h_mean, 3),
                "sparse_forward_mean_m": round(sparse_fwd_mean, 3),
                "hack_return_first": round(hack_return[0], 2),
                "hack_return_last": round(hack_return[-1], 2),
                "hack_return_rise_x": round(hack_rise, 2),
                "shaped_shown_episode_forward_m": round(shaped_pick["forward"], 3),
                "hack_shown_episode_forward_m": round(hack_pick["forward"], 3),
            },
        },
        # geometry constants the component needs to draw the side view
        "geom": {
            "torso_half": [0.18, 0.035],   # box half-extents (x, z) from quadruped.xml
            "leg_radius": 0.018,           # capsule radius
            "foot_radius": 0.022,
            "floor_z": 0.0,
            "target_height": ns_hack["QuadrupedEnv"].TARGET_HEIGHT,   # 0.25 ride height
            "fall_height": ns_hack["QuadrupedEnv"].FALL_HEIGHT,       # 0.14
        },
        # the DISPLAYED reference numbers (meta.yaml reference_run)
        "meta": META,
        "gate": {"hack_forward_abs_max": HACK_FORWARD_ABS_MAX,
                 "hack_return_rise_min": HACK_RETURN_RISE_MIN},
        # PANEL 1 — the two recorded rollouts (real seed-0 geometry)
        "rollouts": {
            "shaped": pack_roll(shaped_pick),
            "hack": pack_roll(hack_pick),
        },
        # PANEL 2 — the hack self-reward rise (real shape, cited band). Points are
        # [training_progress 0..1, design_return].
        "hack_return_curve": curve,
        # PANEL 3 — the shaping / curriculum context (cited forward distances)
        "shaping": [
            {"design": "sparse", "forward_m": META["sparse_forward_m"],
             "note": "sparse “did it move?” — no gradient out of the crouch, barely trains"},
            {"design": "shaped", "forward_m": META["shaped_forward_m"],
             "note": "dense shaped reward — the graded signal a walk emerges from"},
            {"design": "hack", "forward_m": META["hack_forward_m"],
             "note": "reward raw height — rears tall, walks nowhere (the hack)"},
            {"design": "curriculum", "forward_m": META["curriculum_forward_m"],
             "note": "stage the reward in time: stand first, then add forward"},
        ],
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB)")
    print("OK — signatures reproduce meta.yaml; hack reward rises while forward stays tiny.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
