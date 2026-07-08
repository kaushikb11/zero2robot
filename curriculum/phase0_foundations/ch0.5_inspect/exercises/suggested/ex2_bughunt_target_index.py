"""SUGGESTED exercise candidate (humans promote) — bug-hunt, ch0.5.

This is `frame_errors` — the reconstruction that turns one stored observation
into (pos_err, ang_err), the numbers you decide "reached the target" from. It is
supposed to measure how far the BLOCK (tee, obs[2:4]) is from the TARGET
(obs[6:8]). It doesn't: it measures the block's distance from the wrong reference
point. The bug reads plausibly and never crashes — it just quietly reports the
wrong distance, so your success rate is a lie.

The observation layout is documented in pusht_env.py and the chapter's Setup /
Success regions:

    obs[0:2] pusher_x, pusher_y      obs[2:4] tee_x, tee_y
    obs[4:6] sin(tee_yaw), cos(tee_yaw)
    obs[6:8] target_x, target_y      obs[8:10] sin/cos target_yaw

Before you fix the index, write one sentence: which two points is pos_err
supposed to measure between, and what would a block sitting exactly on the
target read as if you were secretly measuring from the pusher instead?

Find the wrong index, fix it, and re-run checks.py until the reading matches the
contract (block on target -> reached; block adrift -> not reached).

Run:  python ex2_bughunt_target_index.py
Estimated learner time: 15 minutes.
"""

import math

import numpy as np

METADATA = {
    "type": "bug-hunt",
    "chapter": "ch0.5-inspect",
}

# The env's success tolerances (mirrored from pusht_env.py so this file is
# self-contained; the real inspect.py imports them from the env, not a copy).
POS_TOL = 0.03
ANG_TOL = 0.20


def wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def frame_errors(state: np.ndarray) -> tuple:
    """(pos_err, ang_err) for one observation. pos_err is the block's distance
    from the TARGET; ang_err is the block's yaw error vs the (zero) target yaw."""
    tee_xy = state[2:4]
    target_xy = state[0:2]
    pos_err = float(np.hypot(tee_xy[0] - target_xy[0], tee_xy[1] - target_xy[1]))
    ang_err = float(abs(wrap_angle(math.atan2(float(state[4]), float(state[5])))))
    return pos_err, ang_err


def reached(state: np.ndarray) -> bool:
    pos_err, ang_err = frame_errors(state)
    return pos_err < POS_TOL and ang_err < ANG_TOL


# Two hand-built terminal states with a known reading (see meta.yaml ex2).
ON_TARGET = np.array([0.15, -0.10, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
OFF_TARGET = np.array([0.0, 0.0, 0.20, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


if __name__ == "__main__":
    on_pos, _ = frame_errors(ON_TARGET)
    off_pos, _ = frame_errors(OFF_TARGET)
    print(f"block ON target:  pos_err={on_pos:.3f}  reached={reached(ON_TARGET)}  (want pos_err~0.0, reached=True)")
    print(f"block 0.20m away: pos_err={off_pos:.3f}  reached={reached(OFF_TARGET)}  (want pos_err~0.20, reached=False)")
    ok = reached(ON_TARGET) and not reached(OFF_TARGET)
    print("MATCH" if ok else "MISMATCH — frame_errors is measuring against the wrong point; fix the index")
