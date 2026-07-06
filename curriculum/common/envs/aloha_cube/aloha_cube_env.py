"""AlohaCube: a simplified bimanual cube-transfer env. zero2robot Phase-1 env.

The task ACT (chapter 1.3) is built on: a right end-effector must pick up a
cube it alone can reach, carry it to the middle, hand it to a left end-effector,
which alone can reach the delivery target. Neither arm can do the task by
itself, so a mid-air handoff is mandatory — the coordination structure ACT's
action chunking exists to exploit.

This is an infra-grade env, mirroring PushT's discipline (reset/step/obs/success
API, bitwise CPU determinism, scripted expert + demo generator). It is a
*simplified* ALOHA: planar end-effectors instead of 14-DOF ViperX arms, and a
weld-constraint grasp abstraction instead of frictional pinch physics. See the
README "Honesty" section for the tradeoffs and why they hold the free-tier +
determinism floor (root CLAUDE.md invariants 1 and 2).

Deterministic on CPU: two fresh envs reset with the same seed produce
byte-identical observations (CI-enforced). All randomness flows through
np.random.Generator(PCG64(seed)); no global RNG, no wall clock.

Observation (float32[10]) -- chapter 1.3's ACT trains on this layout (plus the
top-down image when --video demos are used):

    idx  name              units
    0    right_ee_x        m
    1    right_ee_y        m
    2    right_grip        -      (0 = fully open ... 1 = fully closed)
    3    left_ee_x         m
    4    left_ee_y         m
    5    left_grip         -      (0 = fully open ... 1 = fully closed)
    6    cube_x            m
    7    cube_y            m
    8    target_x          m      (fixed at -0.30)
    9    target_y          m      (fixed at  0.00)

Action (float32[6]): [right_vx, right_vy, right_grip, left_vx, left_vy,
left_grip], clipped to [-1, 1]. The velocity channels are end-effector target
velocities [m/s] via MuJoCo velocity actuators. The grip channels command the
gripper: +1 = close, -1 = open (mapped to each finger's position actuator).
Control is 10 Hz over 100 Hz physics (FRAME_SKIP = 10).

Grasp (the abstraction): before each control step, for each gripper, if it is
closed (finger travel past CLOSE_FRAC of full) AND its end-effector is within
GRASP_R of the cube, the env activates a `weld` equality constraint binding the
cube to that gripper (snapping the cube to the gripper center so the weld starts
satisfied). Opening the gripper releases the weld. This is deterministic and
robust; it is not frictional grasping.

Reward (shaped, in [-1, 0] plus a success bonus):
    dist   = ||cube_xy - target_xy||         (m)
    reward = -(dist / DIST_SCALE)
    +1.0 extra on the step success first latches.

Success (info["success"]): dist < POS_TOL for SUCCESS_HOLD consecutive control
steps. Episode ends at success or MAX_STEPS.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

_XML_PATH = Path(__file__).parent / "aloha_cube.xml"

# Rerun logging follows .claude/skills/rerun-instrument: world/objects/**,
# world/robot/**, policy/action, timeline "sim_time". `import rerun` stays
# lazy so CI without rerun installed never pays for it.

_GRIP_TRAVEL = 0.04  # finger stroke [m]; |grip_qpos| in [0, _GRIP_TRAVEL]


class AlohaCubeEnv:
    OBS_DIM = 10
    ACT_DIM = 6
    MAX_STEPS = 400          # 40 s of sim time at 10 Hz control
    CONTROL_HZ = 10
    FRAME_SKIP = 10          # 100 Hz physics / 10 Hz control
    POS_TOL = 0.04           # m; cube-to-target delivery tolerance
    SUCCESS_HOLD = 5         # consecutive in-tolerance steps to latch success
    DIST_SCALE = 0.7         # reward normalizer (~max cube-target distance)
    TARGET_XY = np.array([-0.30, 0.0])   # fixed delivery target (left-only reach)

    GRASP_R = 0.035          # gripper must be within this of the cube to grasp
    CLOSE_FRAC = 0.70        # |grip_qpos| > CLOSE_FRAC * travel counts as closed

    # reset() sampling ranges
    _CUBE_X = (0.15, 0.38)       # cube spawn x (right-arm exclusive reach)
    _CUBE_Y = (-0.15, 0.15)      # cube spawn y
    _RIGHT_HOME = (0.32, 0.0)    # right ee home (x, y-center)
    _LEFT_HOME = (-0.32, 0.0)    # left ee home (x, y-center)
    _HOME_JITTER = 0.08          # +-uniform y jitter on each arm home

    # body-frame origins (from the MJCF), used to convert world xy <-> joint qpos
    _CUBE_ORIGIN = np.array([0.30, 0.0])

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
        self.data = mujoco.MjData(self.model)
        self._jadr = {
            name: self.model.joint(name).qposadr[0]
            for name in ("cube_x", "cube_y", "right_x", "right_y", "right_grip",
                         "left_x", "left_y", "left_grip")
        }
        self._bid = {name: self.model.body(name).id
                     for name in ("cube", "right_ee", "left_ee")}
        self._eid = {name: self.model.equality(name).id
                     for name in ("right_hold", "left_hold")}
        # A weld holds body2 at a FIXED relative pose to body1, captured from the
        # MJCF home configuration by default — for the left arm that home gap is
        # 0.60 m, which would fling the cube sideways when the weld fires. Force
        # both welds to hold the cube COINCIDENT with the gripper center
        # (identity relpose, zero anchor); combined with snapping the cube to the
        # gripper on grasp, the weld then starts and stays satisfied.
        for eid in self._eid.values():
            self.model.eq_data[eid, :] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        self._step_count = 0
        self._success_streak = 0
        self._success = False
        self._renderer: mujoco.Renderer | None = None
        self._rr = None  # rerun module once enabled

    # ---------------------------------------------------------------- state

    @property
    def right_ee_pos(self) -> np.ndarray:
        return np.array(self.data.body(self._bid["right_ee"]).xpos[:2])

    @property
    def left_ee_pos(self) -> np.ndarray:
        return np.array(self.data.body(self._bid["left_ee"]).xpos[:2])

    @property
    def cube_pos(self) -> np.ndarray:
        return np.array(self.data.body(self._bid["cube"]).xpos[:2])

    @property
    def right_grip(self) -> float:
        """Closedness in [0, 1] (0 = open, 1 = closed)."""
        return float(abs(self.data.qpos[self._jadr["right_grip"]]) / _GRIP_TRAVEL)

    @property
    def left_grip(self) -> float:
        return float(abs(self.data.qpos[self._jadr["left_grip"]]) / _GRIP_TRAVEL)

    @property
    def right_held(self) -> bool:
        return bool(self.data.eq_active[self._eid["right_hold"]])

    @property
    def left_held(self) -> bool:
        return bool(self.data.eq_active[self._eid["left_hold"]])

    def _obs(self) -> np.ndarray:
        rx, ry = self.right_ee_pos
        lx, ly = self.left_ee_pos
        cx, cy = self.cube_pos
        gx, gy = self.TARGET_XY
        return np.array(
            [rx, ry, self.right_grip, lx, ly, self.left_grip, cx, cy, gx, gy],
            dtype=np.float32,
        )

    def _dist(self) -> float:
        return float(np.linalg.norm(self.cube_pos - self.TARGET_XY))

    # ------------------------------------------------------------- reset/step

    def reset(self, seed: int) -> np.ndarray:
        rng = np.random.Generator(np.random.PCG64(seed))
        mujoco.mj_resetData(self.model, self.data)

        cube_xy = np.array([rng.uniform(*self._CUBE_X), rng.uniform(*self._CUBE_Y)])
        right_y = self._RIGHT_HOME[1] + rng.uniform(-self._HOME_JITTER, self._HOME_JITTER)
        left_y = self._LEFT_HOME[1] + rng.uniform(-self._HOME_JITTER, self._HOME_JITTER)

        q = self.data.qpos
        q[self._jadr["cube_x"]] = cube_xy[0] - self._CUBE_ORIGIN[0]
        q[self._jadr["cube_y"]] = cube_xy[1] - self._CUBE_ORIGIN[1]
        # arm slide qpos is displacement from the body's MJCF home; the homes in
        # the XML are (+-0.30, 0), so x-qpos = target_x - (+-0.30), y-qpos = y.
        q[self._jadr["right_x"]] = self._RIGHT_HOME[0] - 0.30
        q[self._jadr["right_y"]] = right_y
        q[self._jadr["left_x"]] = self._LEFT_HOME[0] + 0.30
        q[self._jadr["left_y"]] = left_y
        q[self._jadr["right_grip"]] = 0.0   # open
        q[self._jadr["left_grip"]] = 0.0    # open

        self.data.eq_active[self._eid["right_hold"]] = 0
        self.data.eq_active[self._eid["left_hold"]] = 0
        # ctrl matches the reset pose: zero velocities, grippers commanded open
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
        # velocity channels pass straight through; grip channels map [-1,1] ->
        # finger position target ([+1] = closed). Right finger travels +, left -.
        self.data.ctrl[0] = action[0]   # right_vx
        self.data.ctrl[1] = action[1]   # right_vy
        self.data.ctrl[2] = action[3]   # left_vx
        self.data.ctrl[3] = action[4]   # left_vy
        self.data.ctrl[4] = 0.5 * _GRIP_TRAVEL * (action[2] + 1.0)    # right_grip -> [0, .04]
        self.data.ctrl[5] = -0.5 * _GRIP_TRAVEL * (action[5] + 1.0)   # left_grip  -> [-.04, 0]

        self._update_grasp()
        for _ in range(self.FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        dist = self._dist()
        in_tol = dist < self.POS_TOL
        self._success_streak = self._success_streak + 1 if in_tol else 0

        reward = -(dist / self.DIST_SCALE)
        if not self._success and self._success_streak >= self.SUCCESS_HOLD:
            self._success = True
            reward += 1.0

        done = self._success or self._step_count >= self.MAX_STEPS
        info = {
            "success": self._success, "dist": dist,
            "right_held": self.right_held, "left_held": self.left_held,
        }

        if self._rr is not None:
            self._log_rerun(action=action)
        return self._obs(), float(reward), done, info

    # ----------------------------------------------------------------- grasp

    def _update_grasp(self) -> None:
        """Deterministically weld/release the cube per gripper closed-ness.

        A gripper grabs when it is closed AND within GRASP_R of the cube; it
        releases when it opens. On grab we snap the cube to the gripper center
        so the weld constraint starts satisfied (no solver transient). Both
        welds may be active briefly during the handoff — that is the point.
        """
        cube = self.cube_pos
        for side, ee_pos, grip in (
            ("right", self.right_ee_pos, self.right_grip),
            ("left", self.left_ee_pos, self.left_grip),
        ):
            eid = self._eid[f"{side}_hold"]
            held = bool(self.data.eq_active[eid])
            closed = grip > self.CLOSE_FRAC
            if not held and closed and np.linalg.norm(ee_pos - cube) < self.GRASP_R:
                self._snap_cube_to(ee_pos)
                self.data.eq_active[eid] = 1
            elif held and not closed:
                self.data.eq_active[eid] = 0

    def _snap_cube_to(self, xy: np.ndarray) -> None:
        self.data.qpos[self._jadr["cube_x"]] = xy[0] - self._CUBE_ORIGIN[0]
        self.data.qpos[self._jadr["cube_y"]] = xy[1] - self._CUBE_ORIGIN[1]
        self.data.qvel[self._jadr["cube_x"]] = 0.0
        self.data.qvel[self._jadr["cube_y"]] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ---------------------------------------------------------------- render

    def render_frame(self, height: int = 96, width: int = 96) -> np.ndarray:
        """Top-down RGB frame (H, W, 3) uint8, used for dataset mp4s."""
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
        rr.init("zero2robot/aloha_cube", spawn=spawn)
        if path is not None:
            rr.save(str(path))
        gx, gy = self.TARGET_XY
        rr.log(
            "world/objects/target",
            rr.Points3D([(gx, gy, 0.0)], radii=[0.045], colors=(90, 205, 100, 120)),
            static=True,
        )
        rr.log(
            "world/objects/cube",
            rr.Boxes3D(centers=[(0, 0, 0)], half_sizes=[(0.022, 0.022, 0.022)],
                       colors=(217, 89, 76)),
            static=True,
        )
        rr.log("world/robot/right_ee",
               rr.Cylinders3D(lengths=[0.044], radii=[0.010], colors=(230, 140, 51)),
               static=True)
        rr.log("world/robot/left_ee",
               rr.Cylinders3D(lengths=[0.044], radii=[0.010], colors=(64, 140, 230)),
               static=True)

    def _log_rerun(self, action: np.ndarray | None) -> None:
        rr = self._rr
        rr.set_time("sim_time", duration=self.data.time)
        cx, cy = self.cube_pos
        rr.log("world/objects/cube", rr.Transform3D(translation=(cx, cy, 0.025)))
        rx, ry = self.right_ee_pos
        lx, ly = self.left_ee_pos
        rr.log("world/robot/right_ee", rr.Transform3D(translation=(rx, ry, 0.025)))
        rr.log("world/robot/left_ee", rr.Transform3D(translation=(lx, ly, 0.025)))
        if action is not None:
            rr.log("policy/action", rr.Scalars(action.astype(np.float64)))
            rr.log("train/success", rr.Scalars([float(self._success)]))
            rr.log("train/dist", rr.Scalars([self._dist()]))
            rr.log("train/right_held", rr.Scalars([float(self.right_held)]))
            rr.log("train/left_held", rr.Scalars([float(self.left_held)]))
