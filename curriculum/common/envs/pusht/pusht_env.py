"""PushT: push a T-shaped block to a fixed target pose. zero2robot Phase-0 env.

Deterministic on CPU: two fresh envs reset with the same seed produce
byte-identical observations (CI-enforced, root CLAUDE.md invariant 2).
All randomness flows through np.random.Generator(PCG64(seed)); no global RNG,
no wall clock.

Observation (float32[10]) -- chapter 1.1's BC trains on exactly this layout:

    idx  name              units
    0    pusher_x          m
    1    pusher_y          m
    2    tee_x             m      (block body origin = center of the bar)
    3    tee_y             m
    4    sin(tee_yaw)      -
    5    cos(tee_yaw)      -
    6    target_x          m      (fixed at 0.0)
    7    target_y          m      (fixed at 0.0)
    8    sin(target_yaw)   -      (fixed at sin 0 = 0.0)
    9    cos(target_yaw)   -      (fixed at cos 0 = 1.0)

Action (float32[2]): pusher target velocity [vx, vy] in m/s, clipped to
[-1, 1]. Applied through MuJoCo velocity actuators, held for FRAME_SKIP
physics steps (control at 10 Hz over 100 Hz physics).

Reward (shaped, in [-1, 0] plus a success bonus):
    pos_err = ||tee_xy - target_xy||        (m)
    ang_err = |wrap(tee_yaw - target_yaw)|  (rad)
    reward  = -0.5 * (pos_err / 0.5 + ang_err / pi)
    +1.0 extra on the step success first latches.

Success (info["success"]): pos_err < POS_TOL and ang_err < ANG_TOL for
SUCCESS_HOLD consecutive control steps. Episode ends at success or MAX_STEPS.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

_XML_PATH = Path(__file__).parent / "pusht.xml"

# Rerun logging follows .claude/skills/rerun-instrument: world/objects/**,
# world/robot/**, policy/action, timeline "sim_time". `import rerun` stays
# lazy so CI without rerun installed never pays for it.

_TEE_CENTERS = [(0.0, 0.0, 0.0), (0.0, -0.06, 0.0)]
_TEE_HALF_SIZES = [(0.06, 0.015, 0.015), (0.015, 0.045, 0.015)]


def wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class PushTEnv:
    OBS_DIM = 10
    ACT_DIM = 2
    MAX_STEPS = 300          # 30 s of sim time at 10 Hz control
    CONTROL_HZ = 10
    FRAME_SKIP = 10          # 100 Hz physics / 10 Hz control
    POS_TOL = 0.03           # m
    ANG_TOL = 0.20           # rad (~11.5 deg)
    SUCCESS_HOLD = 5         # consecutive in-tolerance steps to latch success
    TARGET_POSE = np.array([0.0, 0.0, 0.0])  # x, y, yaw -- fixed

    # reset() sampling ranges
    _SPAWN_R = (0.10, 0.24)      # block distance from target
    _PUSHER_BOUND = 0.32         # |x|,|y| bound for the pusher spawn
    _PUSHER_CLEAR = 0.13         # min pusher distance from block center

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
        self.data = mujoco.MjData(self.model)
        self._jadr = {
            name: self.model.joint(name).qposadr[0]
            for name in ("tee_x", "tee_y", "tee_yaw", "pusher_x", "pusher_y")
        }
        self._step_count = 0
        self._success_streak = 0
        self._success = False
        self._renderer: mujoco.Renderer | None = None
        self._rr = None  # rerun module once enabled

    # ---------------------------------------------------------------- state

    @property
    def pusher_pos(self) -> np.ndarray:
        q = self.data.qpos
        return np.array([q[self._jadr["pusher_x"]], q[self._jadr["pusher_y"]]])

    @property
    def tee_pose(self) -> np.ndarray:
        """(x, y, yaw) of the block; yaw wrapped to [-pi, pi)."""
        q = self.data.qpos
        return np.array(
            [
                q[self._jadr["tee_x"]],
                q[self._jadr["tee_y"]],
                wrap_angle(q[self._jadr["tee_yaw"]]),
            ]
        )

    def _obs(self) -> np.ndarray:
        px, py = self.pusher_pos
        tx, ty, tyaw = self.tee_pose
        gx, gy, gyaw = self.TARGET_POSE
        return np.array(
            [px, py, tx, ty, np.sin(tyaw), np.cos(tyaw),
             gx, gy, np.sin(gyaw), np.cos(gyaw)],
            dtype=np.float32,
        )

    def _errors(self) -> tuple[float, float]:
        tx, ty, tyaw = self.tee_pose
        pos_err = float(np.hypot(tx - self.TARGET_POSE[0], ty - self.TARGET_POSE[1]))
        ang_err = float(abs(wrap_angle(tyaw - self.TARGET_POSE[2])))
        return pos_err, ang_err

    # ------------------------------------------------------------- reset/step

    def reset(self, seed: int) -> np.ndarray:
        rng = np.random.Generator(np.random.PCG64(seed))
        mujoco.mj_resetData(self.model, self.data)

        # block: uniform annulus around the target, uniform yaw
        r = rng.uniform(*self._SPAWN_R)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        tee_xy = np.array([r * np.cos(phi), r * np.sin(phi)])
        tee_yaw = rng.uniform(-np.pi, np.pi)

        # pusher: rejection-sample clear of the block (deterministic given seed)
        while True:
            pusher_xy = rng.uniform(-self._PUSHER_BOUND, self._PUSHER_BOUND, size=2)
            if np.linalg.norm(pusher_xy - tee_xy) > self._PUSHER_CLEAR:
                break

        q = self.data.qpos
        q[self._jadr["tee_x"]], q[self._jadr["tee_y"]] = tee_xy
        q[self._jadr["tee_yaw"]] = tee_yaw
        q[self._jadr["pusher_x"]], q[self._jadr["pusher_y"]] = pusher_xy
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._success_streak = 0
        self._success = False
        if self._rr is not None:
            self._log_rerun(action=None)
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.data.ctrl[:] = action
        for _ in range(self.FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        pos_err, ang_err = self._errors()
        in_tol = pos_err < self.POS_TOL and ang_err < self.ANG_TOL
        self._success_streak = self._success_streak + 1 if in_tol else 0

        reward = -0.5 * (pos_err / 0.5 + ang_err / np.pi)
        if not self._success and self._success_streak >= self.SUCCESS_HOLD:
            self._success = True
            reward += 1.0

        done = self._success or self._step_count >= self.MAX_STEPS
        info = {"success": self._success, "pos_err": pos_err, "ang_err": ang_err}

        if self._rr is not None:
            self._log_rerun(action=action)
        return self._obs(), float(reward), done, info

    # ---------------------------------------------------------------- render

    def render_frame(self, height: int = 96, width: int = 96) -> np.ndarray:
        """Top-down RGB frame (H, W, 3) uint8, used for dataset mp4s."""
        # Rebuild the cached renderer when the requested size changes — a
        # Renderer is fixed to the (height, width) it was built with, and
        # rebuilding is cheap. Otherwise a later call would silently return
        # the first call's resolution.
        if (self._renderer is None
                or (self._renderer.height, self._renderer.width) != (height, width)):
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera="top")
        return self._renderer.render()

    # ----------------------------------------------------------------- rerun

    def enable_rerun(self, path: str | Path | None = None, spawn: bool = False) -> None:
        """Log to rerun per the repo-wide entity-path conventions. Off by default."""
        import rerun as rr  # lazy: CI without rerun never imports it

        self._rr = rr
        rr.init("zero2robot/pusht", spawn=spawn)
        if path is not None:
            rr.save(str(path))
        # static geometry: goal T (fixed pose) and the two shapes we move per step
        gx, gy, _ = self.TARGET_POSE
        rr.log(
            "world/objects/target",
            rr.Boxes3D(
                centers=[(gx + c[0], gy + c[1], 0.0) for c in _TEE_CENTERS],
                half_sizes=_TEE_HALF_SIZES,
                colors=(90, 205, 100, 120),
            ),
            static=True,
        )
        rr.log(
            "world/objects/tee",
            rr.Boxes3D(centers=_TEE_CENTERS, half_sizes=_TEE_HALF_SIZES,
                       colors=(115, 128, 242)),
            static=True,
        )
        rr.log(
            "world/robot/pusher",
            rr.Cylinders3D(lengths=[0.04], radii=[0.015], colors=(230, 102, 90)),
            static=True,
        )

    def _log_rerun(self, action: np.ndarray | None) -> None:
        rr = self._rr
        rr.set_time("sim_time", duration=self.data.time)
        tx, ty, tyaw = self.tee_pose
        rr.log(
            "world/objects/tee",
            rr.Transform3D(
                translation=(tx, ty, 0.0152),
                rotation=rr.RotationAxisAngle(axis=(0, 0, 1), radians=tyaw),
            ),
        )
        px, py = self.pusher_pos
        rr.log("world/robot/pusher", rr.Transform3D(translation=(px, py, 0.02)))
        if action is not None:
            rr.log("policy/action", rr.Scalars(action.astype(np.float64)))
            pos_err, ang_err = self._errors()
            rr.log("train/success", rr.Scalars([float(self._success)]))
            rr.log("train/pos_err", rr.Scalars([pos_err]))
            rr.log("train/ang_err", rr.Scalars([ang_err]))
