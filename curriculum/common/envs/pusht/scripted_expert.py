"""Scripted PushT expert: waypoint (pure-pursuit style) push controller.

A two-phase state machine, re-planned every control step:

  APPROACH  navigate to a staging point behind the chosen contact point,
            detouring around a safety circle so the pusher never plows
            through the block on the way.
  PUSH      drive through the contact point in a straight stroke, with a
            proportional correction that keeps the pusher on the push line.
            A stroke ends after a fixed number of steps or when the pusher
            drifts off the line; then we re-approach.

Contact-point selection:
  * TRANSLATE (block far from target): push through the block's center of
    mass toward the target. The push line is offset laterally in proportion
    to the yaw error, so translation strokes also steer the block's angle.
  * ROTATE (position close, yaw not): push tangentially at a bar tip
    (max lever arm) in the direction that reduces yaw error.

Everything is deterministic. Optional exploration noise (demo diversity) is
Gaussian on the action and seeded: same seed => same noisy demo, bit for bit.
"""

from __future__ import annotations

import numpy as np

try:
    from .pusht_env import PushTEnv, wrap_angle
except ImportError:  # running as a loose script
    from pusht_env import PushTEnv, wrap_angle

# T outline in the body frame: (cx, cy, half_x, half_y) per box (see pusht.xml)
_RECTS = [(0.0, 0.0, 0.06, 0.015), (0.0, -0.06, 0.015, 0.045)]
_COM_BODY = np.array([0.0, -0.02571])  # mass-weighted: bar 0.06 kg, stem 0.045 kg
_PUSHER_R = 0.015
_BAR_HALF = 0.06


def _rot(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def _outline_dist(origin_body: np.ndarray, u_body: np.ndarray) -> float:
    """Distance from origin_body along u_body to the far edge of the T outline."""
    ts = np.linspace(0.0, 0.2, 201)
    pts = origin_body[None, :] + ts[:, None] * u_body[None, :]
    inside = np.zeros(len(ts), dtype=bool)
    for cx, cy, hx, hy in _RECTS:
        inside |= (np.abs(pts[:, 0] - cx) <= hx) & (np.abs(pts[:, 1] - cy) <= hy)
    hits = np.nonzero(inside)[0]
    return float(ts[hits[-1]]) if len(hits) else 0.0


class ScriptedExpert:
    APPROACH_SPEED = 0.9
    PUSH_SPEED = 0.4
    ROTATE_SPEED = 0.3
    SAFE_R = 0.11            # detour radius around the block's COM
    STAGE_MARGIN = 0.05      # staging distance behind the contact point
    STROKE_STEPS = 22        # max control steps per push stroke (2.2 s)
    POS_ENTER = 0.045        # pos_err above this -> translate task
    K_LATERAL = 0.05         # lateral push-line offset per rad of yaw error
    LATERAL_MAX = 0.022

    def __init__(self, noise: float = 0.0, seed: int = 0) -> None:
        self.noise = noise
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self._phase = "approach"
        self._stroke = 0

    # ------------------------------------------------------------------ api

    def action(self, env: PushTEnv) -> np.ndarray:
        p = env.pusher_pos
        tee_xy = env.tee_pose[:2]
        yaw = env.tee_pose[2]
        gx, gy, gyaw = env.TARGET_POSE
        rot = _rot(yaw)
        com_w = tee_xy + rot @ _COM_BODY

        pos_err = float(np.linalg.norm(tee_xy - np.array([gx, gy])))
        ang_err = wrap_angle(yaw - gyaw)

        if pos_err < env.POS_TOL * 0.9 and abs(ang_err) < env.ANG_TOL * 0.9:
            self._phase, self._stroke = "approach", 0
            v = np.zeros(2)  # inside tolerance: hold still, let success latch
        else:
            if pos_err > self.POS_ENTER:
                contact, push_dir = self._translate_plan(com_w, yaw, ang_err, gx, gy, gyaw)
            else:
                contact, push_dir = self._rotate_plan(p, com_w, yaw, ang_err)
            v = self._drive(p, com_w, contact, push_dir,
                            speed=self.PUSH_SPEED if pos_err > self.POS_ENTER
                            else self.ROTATE_SPEED)

        if self.noise > 0.0:
            v = v + self.rng.normal(0.0, self.noise, size=2)
        return np.clip(v, -1.0, 1.0).astype(np.float32)

    __call__ = action

    # -------------------------------------------------------------- planning

    def _translate_plan(self, com_w, yaw, ang_err, gx, gy, gyaw):
        """Contact point + direction pushing the COM toward its goal pose."""
        com_goal = np.array([gx, gy]) + _rot(gyaw) @ _COM_BODY
        d = com_goal - com_w
        d = d / np.linalg.norm(d)
        perp = np.array([-d[1], d[0]])
        off = float(np.clip(self.K_LATERAL * ang_err, -self.LATERAL_MAX, self.LATERAL_MAX))

        # cast the ray along the *actual* (offset) push line, in the body frame
        inv = _rot(-yaw)
        origin_b = _COM_BODY + inv @ (perp * off)
        depth = _outline_dist(origin_b, inv @ (-d))
        depth = max(depth, 0.02)  # offset ray can graze past the stem; keep sane
        contact = com_w + perp * off - d * (depth + _PUSHER_R)
        return contact, d

    def _rotate_plan(self, p, com_w, yaw, ang_err):
        """Tangential push at a bar tip, spinning the block toward target yaw."""
        rot = _rot(yaw)
        sgn = 1.0 if ang_err < 0.0 else -1.0  # +1: block needs CCW spin
        x_tip = _BAR_HALF - 0.012
        options = [  # (contact in body frame, push direction in body frame)
            (np.array([x_tip, -sgn * (0.015 + _PUSHER_R)]), np.array([0.0, sgn])),
            (np.array([-x_tip, sgn * (0.015 + _PUSHER_R)]), np.array([0.0, -sgn])),
        ]
        tee_origin = com_w - rot @ _COM_BODY
        c_body, f_body = min(
            options, key=lambda o: np.linalg.norm(p - (tee_origin + rot @ o[0]))
        )
        return tee_origin + rot @ c_body, rot @ f_body

    # --------------------------------------------------------------- control

    def _drive(self, p, com_w, contact, push_dir, speed) -> np.ndarray:
        stage = contact - push_dir * self.STAGE_MARGIN
        perp = np.array([-push_dir[1], push_dir[0]])
        along = float((p - contact) @ push_dir)     # >0: past the contact point
        lateral = float((p - contact) @ perp)

        if self._phase == "push":
            self._stroke += 1
            off_line = abs(lateral) > 0.035
            if self._stroke > self.STROKE_STEPS or off_line:
                self._phase, self._stroke = "approach", 0
            else:
                return push_dir * speed + perp * np.clip(-4.0 * lateral, -0.3, 0.3)

        # approach the staging point; start a stroke once staged on the line
        if abs(lateral) < 0.012 and -self.STAGE_MARGIN - 0.02 < along < -0.01:
            self._phase, self._stroke = "push", 0
            return push_dir * speed + perp * np.clip(-4.0 * lateral, -0.3, 0.3)
        return self._goto(p, stage, com_w)

    def _goto(self, p, waypoint, obstacle) -> np.ndarray:
        """Pure pursuit toward waypoint, detouring around the block."""
        waypoint = np.clip(waypoint, -0.33, 0.33)  # stay inside the walls
        to_wp = waypoint - p
        dist = float(np.linalg.norm(to_wp))
        if dist < 1e-6:
            return np.zeros(2)
        heading = to_wp / dist
        rel = obstacle - p
        d_rel = float(np.linalg.norm(rel))
        proj = float(rel @ heading)
        closest = float(np.linalg.norm(rel - max(proj, 0.0) * heading))
        if 0.0 < proj < dist and closest < self.SAFE_R:
            rel_hat = rel / d_rel
            if d_rel > self.SAFE_R + 0.005:
                # head for the safety circle's tangent point (whichever side
                # deviates least from the direct heading)
                beta = 1.15 * np.arcsin(min(self.SAFE_R / d_rel, 1.0))
                cands = [_rot(beta) @ rel_hat, _rot(-beta) @ rel_hat]
            else:
                # inside the circle: slide tangentially with an outward bias
                t = np.array([rel_hat[1], -rel_hat[0]])
                cands = [t - 0.6 * rel_hat, -t - 0.6 * rel_hat]
            heading = max(cands, key=lambda c: float(c @ heading))
            heading = heading / np.linalg.norm(heading)
        # never command into a wall we are already touching (no wall-pinning)
        for i in (0, 1):
            if (p[i] > 0.33 and heading[i] > 0) or (p[i] < -0.33 and heading[i] < 0):
                heading[i] = 0.0
        n = float(np.linalg.norm(heading))
        if n < 1e-6:
            heading, n = -p / (np.linalg.norm(p) + 1e-9), 1.0
        heading = heading / n
        speed = min(self.APPROACH_SPEED, 5.0 * dist + 0.05)
        return heading * speed


def expert_action(env: PushTEnv, expert: ScriptedExpert | None = None) -> np.ndarray:
    """Functional convenience: one noiseless expert action for env's state."""
    return (expert or ScriptedExpert()).action(env)
