"""Scripted AlohaCube expert: a bimanual pick -> handoff -> place state machine.

Six phases, re-evaluated every control step:

  R_APPROACH  right arm drives onto the cube (gripper open); left arm parks at
              the handoff point, gripper open, waiting.
  R_GRASP     right arm holds position on the cube and closes; once the env's
              weld latches (env.right_held), advance.
  R_CARRY     right arm carries the cube to the handoff point; wait until both
              end-effectors are at the handoff point.
  HANDOFF     both grippers close on the cube at the handoff point; once the
              left weld latches (env.left_held), advance.
  RELEASE     left keeps holding; right opens and retreats to clear the cube;
              once the right weld drops (not env.right_held), advance.
  L_CARRY     left arm carries the cube to the delivery target and holds it in
              tolerance so success can latch.

The plan is purely geometric (proportional pursuit toward waypoints), so it is
deterministic. Optional exploration noise for demo diversity is Gaussian on the
action and seeded: same seed => same noisy demo, bit for bit.
"""

from __future__ import annotations

import numpy as np

try:
    from .aloha_cube_env import AlohaCubeEnv
except ImportError:  # running as a loose script
    from aloha_cube_env import AlohaCubeEnv

_HANDOFF = np.array([0.0, 0.0])   # mid-workspace point where the arms overlap
_OPEN, _CLOSE = -1.0, 1.0         # gripper action extremes


def _pursue(cur: np.ndarray, goal: np.ndarray, gain: float, vmax: float) -> np.ndarray:
    """Proportional pursuit velocity toward `goal`, magnitude-capped at vmax."""
    v = gain * (goal - cur)
    n = float(np.linalg.norm(v))
    return v * (vmax / n) if n > vmax else v


class ScriptedExpert:
    APPROACH_GAIN = 6.0
    APPROACH_SPEED = 1.0
    SETTLE_GAIN = 3.0        # gentler gain when precision-holding on a contact
    SETTLE_SPEED = 0.6
    CARRY_SPEED = 1.0
    GRASP_TOL = 0.018        # end-effector within this of the cube before closing
    HANDOFF_TOL = 0.025      # arms within this of the handoff point before pinching
    RETREAT = np.array([1.0, 0.0])   # right arm backs off to the right on release

    def __init__(self, noise: float = 0.0, seed: int = 0) -> None:
        self.noise = noise
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self._phase = "R_APPROACH"

    # ------------------------------------------------------------------ api

    def action(self, env: AlohaCubeEnv) -> np.ndarray:
        r = env.right_ee_pos
        le = env.left_ee_pos
        c = env.cube_pos
        a = np.zeros(6, dtype=np.float32)

        # Left arm parks at the handoff point (open) until the handoff begins.
        if self._phase in ("R_APPROACH", "R_GRASP", "R_CARRY"):
            a[3:5] = _pursue(le, _HANDOFF, self.APPROACH_GAIN, self.APPROACH_SPEED)
            a[5] = _OPEN

        if self._phase == "R_APPROACH":
            a[0:2] = _pursue(r, c, self.APPROACH_GAIN, self.APPROACH_SPEED)
            a[2] = _OPEN
            if np.linalg.norm(r - c) < self.GRASP_TOL:
                self._phase = "R_GRASP"

        elif self._phase == "R_GRASP":
            a[0:2] = _pursue(r, c, self.SETTLE_GAIN, self.SETTLE_SPEED)
            a[2] = _CLOSE
            if env.right_held:
                self._phase = "R_CARRY"

        elif self._phase == "R_CARRY":
            a[0:2] = _pursue(r, _HANDOFF, self.APPROACH_GAIN, self.CARRY_SPEED)
            a[2] = _CLOSE
            if (np.linalg.norm(r - _HANDOFF) < self.HANDOFF_TOL
                    and np.linalg.norm(le - _HANDOFF) < self.HANDOFF_TOL + 0.01):
                self._phase = "HANDOFF"

        elif self._phase == "HANDOFF":
            a[0:2] = _pursue(r, _HANDOFF, self.SETTLE_GAIN, self.SETTLE_SPEED)
            a[2] = _CLOSE
            a[3:5] = _pursue(le, _HANDOFF, self.SETTLE_GAIN, self.SETTLE_SPEED)
            a[5] = _CLOSE
            if env.left_held:
                self._phase = "RELEASE"

        elif self._phase == "RELEASE":
            a[3:5] = _pursue(le, _HANDOFF, self.SETTLE_GAIN, self.SETTLE_SPEED)
            a[5] = _CLOSE
            a[0:2] = self.RETREAT * self.APPROACH_SPEED  # right backs off...
            a[2] = _OPEN                                 # ...and opens
            if not env.right_held:
                self._phase = "L_CARRY"

        elif self._phase == "L_CARRY":
            a[3:5] = _pursue(le, env.TARGET_XY, self.APPROACH_GAIN, self.CARRY_SPEED)
            a[5] = _CLOSE
            a[0:2] = self.RETREAT * self.APPROACH_SPEED  # keep right clear
            a[2] = _OPEN

        if self.noise > 0.0:
            a = a + self.rng.normal(0.0, self.noise, size=6).astype(np.float32)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    __call__ = action


def expert_action(env: AlohaCubeEnv, expert: ScriptedExpert | None = None) -> np.ndarray:
    """Functional convenience: one noiseless expert action for env's state.

    NOTE: the expert is a stateful state machine; construct one ScriptedExpert
    per episode and call it every step (as gen_demos does) rather than using
    this helper in a loop.
    """
    return (expert or ScriptedExpert()).action(env)
