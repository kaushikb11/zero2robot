"""SUGGESTED exercise candidate (humans promote) — code-completion, ch0.5.

You get `frame_errors` (already correct) and two golden trajectories. Complete
`episode_reached`: decide whether a demonstration reached the target. The idea
the chapter's Success region makes: the dataset stores NO success column, and the
recorder stops the instant the env latches success — so the last frame it stored
is the block sitting on the goal. Reading "reached" is reading whether the
episode ENDED within tolerance (POS_TOL and ANG_TOL).

Complete the one line where marked, then run checks.py — it applies your function
to a trajectory that ends on target (should read reached) and one that ends
adrift (should not).

Run:  python ex3_complete_episode_reached.py
Estimated learner time: 12 minutes.
"""

import math

import numpy as np

METADATA = {
    "type": "code-completion",
    "chapter": "ch0.5-inspect",
}

POS_TOL = 0.03   # mirrored from pusht_env.py; the real inspect.py imports these
ANG_TOL = 0.20


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def frame_errors(state: np.ndarray) -> tuple:
    """(pos_err, ang_err) for one observation: block(obs[2:4]) vs target(obs[6:8]),
    yaw decoded from the sin/cos pair at obs[4:6] against the zero target yaw."""
    pos_err = float(np.hypot(state[2] - state[6], state[3] - state[7]))
    ang_err = float(abs(wrap_angle(math.atan2(float(state[4]), float(state[5])))))
    return pos_err, ang_err


def episode_reached(states: np.ndarray) -> bool:
    """Did this demonstration reach the goal? Read the TERMINAL frame: the demo
    ended there, so the block's final pose is where it left it."""
    # TODO(you): compute (pos_err, ang_err) for the LAST frame, states[-1], with
    # frame_errors, and return True iff pos_err < POS_TOL AND ang_err < ANG_TOL.
    raise NotImplementedError("complete episode_reached (see inspect.py's Success region)")


def _traj(end_state: np.ndarray) -> np.ndarray:
    """A 6-frame trajectory that drifts in, ending at end_state."""
    start = np.array([0.1, 0.1, 0.25, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return np.stack([start + (end_state - start) * t for t in np.linspace(0.0, 1.0, 6)]).astype(np.float32)


ENDS_ON_TARGET = _traj(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))
ENDS_ADRIFT = _traj(np.array([0.0, 0.0, 0.20, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))


if __name__ == "__main__":
    print(f"ends on target: reached={episode_reached(ENDS_ON_TARGET)}  (want True)")
    print(f"ends adrift:    reached={episode_reached(ENDS_ADRIFT)}  (want False)")
