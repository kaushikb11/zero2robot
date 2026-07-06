"""Pusher-reach: drive a planar 2-link arm's fingertip to a random target.

The canonical dense-reward continuous-control task (OpenAI Gym's "Reacher"
lineage) and ch2.2's SAC teaching env. A two-link arm moves in the x-y plane;
the policy applies a torque to each joint and must place the fingertip on a
seeded-random target. The reward is the NEGATIVE fingertip->target distance, so
every step carries a gradient toward the goal.

Why this env for SAC (the off-policy bargain, ch2.2). Cartpole (ch2.1, PPO) has
a sparse-ish alive bonus: +1/step, no gradient telling you *which way* is
better, which suits an on-policy method that learns from whole trajectories.
Pusher-reach is the opposite: a DENSE per-step signal (-distance) that an
off-policy learner can bootstrap on from a replay buffer full of old
transitions. That dense reward is exactly what SAC exploits — and exactly why
the chapter pairs the two envs.

Infra-grade env, mirroring Cartpole / PushT discipline (reset/step/obs/render
API, bitwise CPU determinism, a `--seed`-able rollout demo). The action space is
continuous torque, the same Gaussian-policy shape ch2.1's PPO used on cartpole,
so ch2.2's policy/critic code carries over unchanged.

Deterministic on CPU: two fresh envs reset with the same seed produce
byte-identical observations (CI-enforced, root CLAUDE.md invariant 2). All
randomness flows through np.random.Generator(PCG64(seed)); no global RNG, no
wall clock. The scene has no collisions and no joint limits (a pure articulated
2-body chain, no contact/constraint solver), so the dynamics are cheap and
reproducible. The seeded target is a mocap body — it never enters qpos, so it
cannot perturb the arm's dynamics.

Observation (float32[8]) -- ch2.2's SAC trains on exactly this layout. Each
joint angle is encoded as (cos, sin) so the observation is continuous through
+-pi (no wrap seam a Gaussian policy would trip on):

    idx  name                    units
    0    cos(shoulder_angle)     -
    1    sin(shoulder_angle)     -
    2    cos(elbow_angle)        -
    3    sin(elbow_angle)        -
    4    shoulder_angvel         rad/s
    5    elbow_angvel            rad/s
    6    fingertip_to_target_x   m       (target_x - fingertip_x)
    7    fingertip_to_target_y   m       (target_y - fingertip_y)

The last two entries carry the dense signal: their norm IS the distance the
reward penalizes. (There is no wrap seam on the arm-relative vector, so it is
left raw rather than sin/cos-encoded.)

Action (float32[2]): joint torques [tau_shoulder, tau_elbow], clipped to
[-1, 1] and applied through the MuJoCo motors (gear 0.5 => +-0.5 N*m). Held for
FRAME_SKIP = 2 physics steps (50 Hz control over 100 Hz physics, timestep
0.01 s).

Reward (DENSE, negative distance + an optional one-time success bonus):

    dist   = ||fingertip_xy - target_xy||        (m)
    reward = -dist                               (every step)
             + SUCCESS_BONUS (1.0) once, on the step success first latches.

Return is therefore dominated by how quickly the fingertip closes on the target
and how tightly it holds there -- a smooth, informative signal at every step.

Success (info["success"]): dist < SUCCESS_TOL. It latches (stays True once
reached) and the bonus is paid once.

Termination vs truncation:

    terminated  only if terminate_on_success=True (default False) AND success
                latched -- an early "you reached it" stop.
    truncated   the step budget ran out (step_count >= MAX_STEPS).

Default is Reacher-style: NO early termination (terminate_on_success=False), so
the episode always runs the full horizon and the arm must *hold* at the target,
not just touch it. This gives SAC a richer dense-reward signal (staying on
target keeps paying ~0 reward vs. drifting away). Flip the flag for a
reach-and-stop variant. (Flagged for the ch2.2 author -- see README.)

`step()` returns the Cartpole/PushT 4-tuple `(obs, reward, done, info)` with
`done = terminated or truncated`; `info` carries the two flags separately plus
`dist` and `success`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

_XML_PATH = Path(__file__).parent / "pusher_reach.xml"

# Rerun logging follows .claude/skills/rerun-instrument: world/objects/**,
# world/robot/**, policy/action, timeline "sim_time". `import rerun` stays
# lazy so CI without rerun installed never pays for it.


def wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class PusherReachEnv:
    OBS_DIM = 8
    ACT_DIM = 2
    MAX_STEPS = 100          # 2 s of sim time at 50 Hz control
    CONTROL_HZ = 50
    FRAME_SKIP = 2           # 100 Hz physics / 50 Hz control (timestep 0.01 s)
    LINK_LEN = 0.1           # m -- each link; total reach = 2 * LINK_LEN = 0.2 m
    SUCCESS_TOL = 0.02       # m -- fingertip within this of target => success
    SUCCESS_BONUS = 1.0      # one-time reward when success first latches

    # reset() target sampling: uniform annulus around the base, strictly inside
    # the arm's reach so the analytic IK always has a solution.
    _TARGET_R = (0.05, 0.19)  # m; max < 2 * LINK_LEN = 0.20

    def __init__(self, terminate_on_success: bool = False) -> None:
        self.terminate_on_success = terminate_on_success
        self.model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
        self.data = mujoco.MjData(self.model)
        self._jadr = {
            name: self.model.joint(name).qposadr[0]
            for name in ("shoulder", "elbow")
        }
        self._vadr = {
            name: self.model.joint(name).dofadr[0]
            for name in ("shoulder", "elbow")
        }
        self._fingertip_id = self.model.site("fingertip").id
        self._elbow_body_id = self.model.body("link2").id
        self._step_count = 0
        self._success = False
        self._renderer: mujoco.Renderer | None = None
        self._rr = None  # rerun module once enabled

    # ---------------------------------------------------------------- state

    @property
    def shoulder_angle(self) -> float:
        return wrap_angle(float(self.data.qpos[self._jadr["shoulder"]]))

    @property
    def elbow_angle(self) -> float:
        return wrap_angle(float(self.data.qpos[self._jadr["elbow"]]))

    @property
    def shoulder_angvel(self) -> float:
        return float(self.data.qvel[self._vadr["shoulder"]])

    @property
    def elbow_angvel(self) -> float:
        return float(self.data.qvel[self._vadr["elbow"]])

    @property
    def fingertip_pos(self) -> np.ndarray:
        """World (x, y) of the end-effector (from MuJoCo forward kinematics)."""
        return self.data.site_xpos[self._fingertip_id][:2].copy()

    @property
    def target_pos(self) -> np.ndarray:
        """World (x, y) of the seeded mocap target."""
        return self.data.mocap_pos[0][:2].copy()

    def _fingertip_to_target(self) -> np.ndarray:
        return self.target_pos - self.fingertip_pos

    def _dist(self) -> float:
        return float(np.linalg.norm(self._fingertip_to_target()))

    def _obs(self) -> np.ndarray:
        sh, el = self.shoulder_angle, self.elbow_angle
        dx, dy = self._fingertip_to_target()
        return np.array(
            [np.cos(sh), np.sin(sh), np.cos(el), np.sin(el),
             self.shoulder_angvel, self.elbow_angvel, dx, dy],
            dtype=np.float32,
        )

    # ------------------------------------------------------------- reset/step

    def reset(self, seed: int) -> np.ndarray:
        rng = np.random.Generator(np.random.PCG64(seed))
        mujoco.mj_resetData(self.model, self.data)

        # arm: both joint angles uniform in [-pi, pi); velocities start at rest
        shoulder, elbow = rng.uniform(-np.pi, np.pi, size=2)
        self.data.qpos[self._jadr["shoulder"]] = shoulder
        self.data.qpos[self._jadr["elbow"]] = elbow
        self.data.qvel[:] = 0.0

        # target: uniform annulus around the base (always inside reach)
        r = rng.uniform(*self._TARGET_R)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        self.data.mocap_pos[0] = np.array([r * np.cos(phi), r * np.sin(phi), 0.0])

        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._success = False
        if self._rr is not None:
            self._log_rerun(action=None)
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        self.data.ctrl[:] = action
        for _ in range(self.FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        dist = self._dist()
        reward = -dist  # DENSE: negative distance, a gradient every step
        if not self._success and dist < self.SUCCESS_TOL:
            self._success = True
            reward += self.SUCCESS_BONUS  # one-time bonus on first reach

        terminated = self.terminate_on_success and self._success
        truncated = self._step_count >= self.MAX_STEPS
        done = terminated or truncated
        info = {
            "terminated": terminated,
            "truncated": truncated,
            "success": self._success,
            "dist": dist,
        }

        if self._rr is not None:
            self._log_rerun(action=action)
        return self._obs(), float(reward), done, info

    # ---------------------------------------------------------------- render

    def render_frame(self, height: int = 240, width: int = 240) -> np.ndarray:
        """Top-down RGB frame (H, W, 3) uint8."""
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
        rr.init("zero2robot/pusher_reach", spawn=spawn)
        if path is not None:
            rr.save(str(path))

    def _log_rerun(self, action: np.ndarray | None) -> None:
        rr = self._rr
        rr.set_time("sim_time", duration=self.data.time)
        # the arm as a polyline base -> elbow -> fingertip
        base = (0.0, 0.0, 0.0)
        elbow = tuple(self.data.xpos[self._elbow_body_id])
        ftip = self.data.site_xpos[self._fingertip_id]
        rr.log(
            "world/robot/arm",
            rr.LineStrips3D([[base, elbow, tuple(ftip)]], colors=(77, 89, 153)),
        )
        rr.log(
            "world/robot/fingertip",
            rr.Points3D([tuple(ftip)], radii=0.012, colors=(230, 102, 90)),
        )
        tx, ty = self.target_pos
        rr.log(
            "world/objects/target",
            rr.Points3D([(tx, ty, 0.0)], radii=0.012, colors=(90, 205, 100)),
        )
        if action is not None:
            rr.log("policy/action", rr.Scalars(action.astype(np.float64)))
            rr.log("train/dist", rr.Scalars([self._dist()]))
            rr.log("train/success", rr.Scalars([float(self._success)]))


def reach_action(env: PusherReachEnv, kp: float = 25.0, kd: float = 2.0) -> np.ndarray:
    """A scripted "solve IK, then PD to it" reach policy (baseline, not learned).

    For a planar 2-link arm the inverse kinematics are closed-form: given the
    target (x, y), the elbow angle follows from the law of cosines and the
    shoulder angle from the resulting triangle. We take the elbow-"down" branch
    (elbow >= 0). A PD law then drives the joints toward those target angles.

    It exists to (a) prove the dense reward really rewards reaching (this crushes
    random) and (b) give the ch2.2 author a non-learned reference return. SAC
    should match or beat it. Because the target is always sampled strictly inside
    the arm's reach, the IK always has a solution.
    """
    tx, ty = env.target_pos
    r2 = tx * tx + ty * ty
    cos_elbow = np.clip(
        (r2 - 2.0 * env.LINK_LEN ** 2) / (2.0 * env.LINK_LEN ** 2), -1.0, 1.0
    )
    elbow_des = np.arccos(cos_elbow)  # elbow-down branch, in [0, pi]
    shoulder_des = np.arctan2(ty, tx) - np.arctan2(
        env.LINK_LEN * np.sin(elbow_des), env.LINK_LEN * (1.0 + cos_elbow)
    )
    err = np.array(
        [wrap_angle(shoulder_des - env.shoulder_angle),
         wrap_angle(elbow_des - env.elbow_angle)]
    )
    qvel = np.array([env.shoulder_angvel, env.elbow_angvel])
    return np.clip(kp * err - kd * qvel, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------- demo


def _rollout(env: PusherReachEnv, seed: int, policy: str,
             rng: np.random.Generator) -> tuple[float, float, bool]:
    """Run one episode; return (final_dist, return, success)."""
    env.reset(seed)
    done, total = False, 0.0
    dist = env._dist()
    success = False
    while not done:
        if policy == "random":
            action = rng.uniform(-1.0, 1.0, size=PusherReachEnv.ACT_DIM).astype(np.float32)
        elif policy == "scripted":
            action = reach_action(env)
        else:  # "zero" -- no torque (arm coasts to rest)
            action = np.zeros(PusherReachEnv.ACT_DIM, dtype=np.float32)
        _, reward, done, info = env.step(action)
        total += reward
        dist = info["dist"]
        success = info["success"]
    return dist, total, success


def main() -> None:
    parser = argparse.ArgumentParser(description="Pusher-reach rollout demo / baseline.")
    parser.add_argument("--seed", type=int, default=0, help="base seed (episode i uses seed+i)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--policy", choices=("random", "scripted", "zero"), default="scripted")
    parser.add_argument("--rerun", type=str, default=None, help="write an .rrd to this path")
    args = parser.parse_args()

    env = PusherReachEnv()
    if args.rerun is not None:
        env.enable_rerun(path=args.rerun)

    # A single seeded generator drives the random policy so the whole run is
    # reproducible from (--seed, --episodes, --policy) alone.
    rng = np.random.Generator(np.random.PCG64(args.seed))
    finals, returns, successes = [], [], []
    for i in range(args.episodes):
        d, ret, ok = _rollout(env, args.seed + i, args.policy, rng)
        finals.append(d)
        returns.append(ret)
        successes.append(ok)
    finals, returns = np.array(finals), np.array(returns)
    print(f"policy={args.policy}  episodes={args.episodes}  seed={args.seed}")
    print(f"  mean final distance: {finals.mean():.4f} +- {finals.std():.4f} m")
    print(f"  mean return:         {returns.mean():.2f} +- {returns.std():.2f}")
    print(f"  success rate:        {sum(successes)}/{len(successes)} "
          f"(dist < {PusherReachEnv.SUCCESS_TOL} m)")


if __name__ == "__main__":
    main()
