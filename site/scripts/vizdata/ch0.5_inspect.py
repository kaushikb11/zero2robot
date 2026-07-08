#!/usr/bin/env python3
"""Regenerate the ch0.5 "Inspecting Your Dataset" concept-toy vizdata, seed 0.

The site's DatasetInspectToy island is a RECORDED-DATA dataset browser: it scrubs
a demonstration episode frame-by-frame (the pusher worrying the T-block, with the
recorded action at each step) and shows the honest shape of the whole dataset
(episode count, lengths, the reconstructed success rate, action/state ranges). It
teaches what a demonstration dataset IS — the (state, action) pairs BC will learn
from. So it needs REAL rows from a REAL LeRobot v3 dataset, never invented shapes.

Why we exec a PREFIX of inspect.py instead of `import inspect`
--------------------------------------------------------------
Two reasons. First, inspect.py's basename collides with the stdlib `inspect`
module. Second, it is a loose script (no `if __name__ == "__main__"` guard): the
`# --- region: run ---` tail provisions + loads + writes metrics + logs rerun the
moment it is imported. We must NOT modify inspect.py (it is LOC-capped and its
tests are human-owned). So — exactly like site/scripts/vizdata/ch3.3_engine.py —
we read its source and exec only the prefix up to `# --- region: run ---` in a
throwaway namespace. That hands us inspect.py's OWN `provision_dataset`,
`load_episodes`, `inspect_episodes`, `frame_errors`, `decode_tee_yaw`, and the
shared `PushTEnv`, at the default config (seed 0, cpu), with zero edits — so the
browser rows are bit-faithful to the chapter artifact.

The honesty gate (STOP-on-drift vs meta.yaml)
---------------------------------------------
meta.yaml records a reference run under exercise_checks.ex1: `inspect.py --seed 0
--episodes 3 --no-rerun`, natural episode ends, gives n_episodes 3 and
success_rate 1.0 with --break none, and 0.0 under --break yaw-swap (reading the
sin/cos yaw backwards reflects every angle). We regenerate that exact config and
assert BOTH before writing. If the reconstructed reading drifts, we STOP.

    Run:  .venv/bin/python site/scripts/vizdata/ch0.5_inspect.py
    Out:  curriculum/phase0_foundations/ch0.5_inspect/demo/vizdata.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
INSPECT_PY = REPO / "curriculum" / "phase0_foundations" / "ch0.5_inspect" / "inspect.py"
OUT_JSON = REPO / "curriculum" / "phase0_foundations" / "ch0.5_inspect" / "demo" / "vizdata.json"

SEED = 0
EPISODES = 3               # matches meta.yaml exercise_checks.ex1 reference run
CUT_MARKER = "# --- region: run ---"

# Keep the committed JSON small + text-only: subsample each episode to at most
# this many frames (linspace, endpoints kept) so scrubbing stays smooth but the
# file stays a few KB. The recorded terminal frame (where "reached" is decided)
# is always the last sample, so the reconstructed success rate is faithful.
MAX_FRAMES_PER_EP = 60

# meta.yaml exercise_checks.ex1 reference_run (seed 0, cpu, --episodes 3,
# natural episode ends). The honesty gate: if the regenerated reading drifts from
# these, STOP rather than ship an invented dataset shape.
META_EX1 = {
    "n_episodes": 3,
    "success_rate_none": 1.0,
    "success_rate_yaw_swap": 0.0,
}


def exec_inspect_prefix(scratch: Path):
    """Exec inspect.py up to the run region, in an isolated namespace with argv
    pinned to the reference config. Returns the populated globals dict (the
    artifact's own provision/load/inspect functions + the shared PushTEnv)."""
    src = INSPECT_PY.read_text()
    cut = src.index(CUT_MARKER)
    prefix = src[:cut]

    old_argv = sys.argv
    # argparse.parse_args() runs DURING exec — pin the meta ex1 config, no rerun.
    sys.argv = [str(INSPECT_PY), "--seed", str(SEED), "--episodes", str(EPISODES),
                "--no-rerun", "--out", str(scratch)]
    ns: dict = {"__file__": str(INSPECT_PY), "__name__": "inspect_toy_vizgen"}
    try:
        exec(compile(prefix, str(INSPECT_PY), "exec"), ns)  # noqa: S102 — our own trusted source
    finally:
        sys.argv = old_argv
    return ns


def build(scratch: Path) -> int:
    ns = exec_inspect_prefix(scratch)
    provision_dataset = ns["provision_dataset"]
    load_episodes = ns["load_episodes"]
    inspect_episodes = ns["inspect_episodes"]
    frame_errors = ns["frame_errors"]
    decode_tee_yaw = ns["decode_tee_yaw"]
    PushTEnv = ns["PushTEnv"]
    STATE_NAMES = ns["STATE_NAMES"]
    TASK = ns["TASK"]

    # ---- provision a REAL LeRobot v3 dataset exactly as ch0.4 / inspect.py does,
    #      then read it back through the artifact's OWN loader (seed 0, cpu). The
    #      parquet lives ONLY in this temp dir (deleted on exit); nothing binary
    #      is written into the repo. -----
    dataset_root = scratch / "dataset"
    print(f"[ch0.5 vizdata] provisioning {EPISODES}-episode stand-in (seed {SEED}) …")
    provision_dataset(dataset_root, EPISODES, SEED, smoke=False)

    info, episodes = load_episodes(dataset_root, "none")           # schema + per-episode arrays
    summary = inspect_episodes(episodes, "none")                    # lengths + reconstructed reading
    # the same episodes read WRONG on purpose — the "seeing like a robot" caveat.
    summary_yaw = inspect_episodes(episodes, "yaw-swap")

    # ---------------------------------------------------------------- honesty gate
    print("[ch0.5 vizdata] regenerated reading vs meta.yaml exercise_checks.ex1:")
    print(f"  n_episodes        : {summary['n_episodes']}   (meta {META_EX1['n_episodes']})")
    print(f"  success_rate none : {summary['success_rate']}   (meta {META_EX1['success_rate_none']})")
    print(f"  success_rate swap : {summary_yaw['success_rate']}   (meta {META_EX1['success_rate_yaw_swap']})")
    fail = []
    if summary["n_episodes"] != META_EX1["n_episodes"]:
        fail.append(f"n_episodes {summary['n_episodes']} != {META_EX1['n_episodes']}")
    if abs(summary["success_rate"] - META_EX1["success_rate_none"]) > 1e-9:
        fail.append(f"success_rate(none) {summary['success_rate']} != {META_EX1['success_rate_none']}")
    if abs(summary_yaw["success_rate"] - META_EX1["success_rate_yaw_swap"]) > 1e-9:
        fail.append(f"success_rate(yaw-swap) {summary_yaw['success_rate']} != {META_EX1['success_rate_yaw_swap']}")
    if fail:
        print("\nSTOP — regenerated reading does NOT match meta.yaml ex1:", file=sys.stderr)
        for f in fail:
            print("  x " + f, file=sys.stderr)
        return 1

    # -------------------------------------------------------- dataset-wide ranges
    # The honest dataset SHAPE: per-dim min/max/mean over every frame, for the
    # 10-D state (what the robot saw) and the 2-D action (what it did). These are
    # the numbers BC will fit a function between.
    all_states = np.concatenate([ep["states"] for ep in episodes], axis=0)   # (n_frames, 10)
    all_actions = np.concatenate([ep["actions"] for ep in episodes], axis=0)  # (n_frames, 2)

    def ranges(arr: np.ndarray, names: list[str]) -> list[dict]:
        return [
            {
                "name": names[i],
                "min": round(float(arr[:, i].min()), 4),
                "max": round(float(arr[:, i].max()), 4),
                "mean": round(float(arr[:, i].mean()), 4),
            }
            for i in range(arr.shape[1])
        ]

    ACTION_NAMES = ["pusher_vx", "pusher_vy"]

    # -------------------------------------------------- pack a scrubbable episode
    # Every episode is browsable; each is subsampled (endpoints kept, so the
    # terminal "reached" frame survives) and carries the RAW 10-D state + the
    # recorded 2-D action per frame. The toy decodes pusher/tee/yaw for the
    # top-down picture (atan2(sin, cos) — the one correct read), exactly the way
    # inspect.py does, so "seeing like a robot" is literal.
    gx, gy, gyaw = (float(v) for v in PushTEnv.TARGET_POSE)

    def pack_episode(ep_idx: int, ep: dict) -> dict:
        states, actions = ep["states"], ep["actions"]
        n = len(states)
        if n <= MAX_FRAMES_PER_EP:
            idxs = list(range(n))
        else:
            idxs = sorted(set(np.linspace(0, n - 1, MAX_FRAMES_PER_EP).round().astype(int).tolist()))
        frames = []
        for i in idxs:
            s = states[i]
            pos_err, ang_err = frame_errors(s, "none")
            frames.append({
                "i": int(i),
                "state": [round(float(v), 4) for v in s],
                "action": [round(float(v), 4) for v in actions[i]],
                "tee_yaw": round(float(decode_tee_yaw(s, "none")), 4),   # atan2(sin, cos)
                "pos_err": round(pos_err, 4),
                "ang_err": round(ang_err, 4),
            })
        return {
            "index": ep_idx,
            "length": int(n),
            "reached": bool(summary["reached"][ep_idx]),
            "final_pos_err": summary["final_pos_err"][ep_idx],
            "final_ang_err": summary["final_ang_err"][ep_idx],
            "frames": frames,
        }

    packed = [pack_episode(i, ep) for i, ep in enumerate(episodes)]

    # world half-extent that comfortably holds every pusher + tee position, so the
    # toy maps every frame into one fixed square viewBox (no reflow while scrubbing).
    extent = float(np.max(np.abs(all_states[:, :4]))) if len(all_states) else 0.35
    world_half_extent_m = round(max(0.35, extent * 1.12), 3)

    data = {
        "provenance": {
            "source": "curriculum/phase0_foundations/ch0.5_inspect/inspect.py",
            "generator": "site/scripts/vizdata/ch0.5_inspect.py",
            "seed": SEED,
            "device": "cpu",
            "config": f"default stand-in (ScriptedExpert noise=0.08), --episodes {EPISODES}, "
                      "natural episode ends, --break none",
            "stack": "lerobot 0.4.4 (CODEBASE_VERSION v3.0) / mujoco 3.10.0 / numpy 2.4.6",
            "note": "Real rows from a REAL LeRobot v3 dataset, provisioned + read back "
                    "through inspect.py's OWN provision_dataset + load_episodes + "
                    "inspect_episodes, then subsampled. Matches meta.yaml "
                    "exercise_checks.ex1 (n_episodes 3, success_rate 1.0; yaw-swap 0.0). "
                    "The dataset stores NO success column — success_rate is RECONSTRUCTED "
                    "from the terminal frame with the env's own POS_TOL/ANG_TOL. NO "
                    "binaries: the provisioned parquet lives in a temp dir, deleted on exit.",
        },
        "task": TASK,
        "schema": {
            "obs_dim": info["obs_dim"],
            "act_dim": info["act_dim"],
            "fps": info["fps"],
            "feature_keys": info["feature_keys"],
            "state_names": STATE_NAMES,
            "action_names": ACTION_NAMES,
        },
        "tolerances": {"pos_tol_m": PushTEnv.POS_TOL, "ang_tol_rad": PushTEnv.ANG_TOL},
        "target": {"x": gx, "y": gy, "yaw": gyaw},
        "world_half_extent_m": world_half_extent_m,
        "stats": {
            "n_episodes": summary["n_episodes"],
            "n_frames": summary["n_frames"],
            "episode_lengths": summary["episode_lengths"],
            "mean_episode_length": summary["mean_episode_length"],
            "n_reached": summary["n_reached"],
            "success_rate": summary["success_rate"],
            "success_rate_yaw_swap": summary_yaw["success_rate"],   # the reading-bug caveat
            "reached": summary["reached"],
            "final_pos_err": summary["final_pos_err"],
            "final_ang_err": summary["final_ang_err"],
        },
        "state_ranges": ranges(all_states, STATE_NAMES),
        "action_ranges": ranges(all_actions, ACTION_NAMES),
        "episodes": packed,
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\n[ch0.5 vizdata] wrote {OUT_JSON.relative_to(REPO)}  ({kb:.1f} KB, text only)")
    print(f"OK — {summary['n_episodes']} episodes / {summary['n_frames']} frames; "
          f"success_rate {summary['success_rate']:g} (yaw-swap {summary_yaw['success_rate']:g}); "
          f"matches meta.yaml ex1.")
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ch0.5-viz-") as tmp:
        return build(Path(tmp))


if __name__ == "__main__":
    raise SystemExit(main())
