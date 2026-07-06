"""Quadruped: make a minimal 4-legged robot STAND, then WALK forward.

The locomotion env for zero2robot Phase 2 — the reward ch2.4 (Reward Design)
shapes and hacks, the task ch2.5 (Locomotion: The Quadruped Walks) trains a
policy on, and the base env ch2.7 (Domain Randomization) perturbs. A box torso
on a floating (free-joint) base carries four identical two-joint legs (a hip
hinge + a knee hinge => 8 actuated joints), each ending in a foot that touches a
friction floor under gravity. The policy commands the 8 joints; a good policy
holds the torso up at height and drives it forward in +x.

Minimal, FROM-SCRATCH quadruped (NOT mujoco_menagerie): a hand-written 2-DOF/leg
robot, not a Go1/ANYmal import. A menagerie model would need a decision-log
dependency entry and is heavier than the free-tier teaching env needs. See the
README for the honest tradeoffs. This is the third infra-grade Phase-2 env,
mirroring cartpole (ch2.1) and pusher-reach (ch2.2): same reset/step/obs API,
same continuous-action Gaussian-policy shape, same bitwise-CPU-determinism
discipline — so the RL chapters reuse the pattern unchanged.

Deterministic on CPU EVEN WITH GROUND CONTACT (CI-enforced, root CLAUDE.md
invariant 2). All randomness flows through np.random.Generator(PCG64(seed)); no
global RNG, no wall clock. The scene pins the contact solve (Newton, fixed
iteration counts, pyramidal cone, explicit foot friction/softness — see
quadruped.xml) and MuJoCo's CPU mj_step is single-threaded and deterministic, so
two runs of the same seed + actions are byte-identical, contacts and all. The
only collidable pairs are the four foot-floor contacts.

Observation (float32[23]) -- ch2.5's policy trains on exactly this layout:

    idx    name                     units
    0..7   joint angles             rad   (FL_hip, FL_knee, FR_hip, FR_knee,
                                            HL_hip, HL_knee, HR_hip, HR_knee)
    8..15  joint velocities         rad/s (same joint order)
    16     torso height             m     (world z of the torso center)
    17..19 torso up-vector          -     (world-frame torso body z-axis;
                                            (0,0,1) = perfectly upright, so
                                            entry 19 alone is the "uprightness")
    20..22 torso linear velocity    m/s   (world frame; entry 20 is FORWARD vx)

The up-vector (not a raw quaternion) encodes orientation with no sign/wrap seam a
Gaussian policy would trip on, and its z-component is directly the term the
reward and the fall-check read.

Action (float32[8]): a target-angle OFFSET for each joint around the nominal
standing pose, clipped to [-1, 1] and scaled by ACTION_SCALE (rad). The env
commands DEFAULT_POSE + ACTION_SCALE * action to the 8 PD position servos (the
standard legged-RL "residual around a default crouch" setup). Held for
FRAME_SKIP = 4 physics steps (50 Hz control over 200 Hz physics, timestep
0.005 s). Same joint order as the observation.

Reward (a locomotion reward built to be SHAPED in ch2.4 -- five named, weighted
terms, each returned in info["reward_terms"] so a chapter can reweight/ablate
them one at a time):

    forward  = W_FORWARD * clip(vx, -MAX_VX, MAX_VX)   reward moving forward (+x)
    upright  = W_UPRIGHT * up_z                        reward the torso staying level
    height   = -W_HEIGHT * (height - TARGET_HEIGHT)^2  penalize crouching/bouncing
    alive    = W_ALIVE                                 a per-step bonus for not falling
    ctrl     = -W_CTRL * sum(action^2)                 penalize large joint commands
    reward   = forward + upright + height + alive + ctrl

The default weights make "stand up and hold" already positive (alive + upright
dominate) and "walk forward" strictly better (the forward term adds on top) —
the shaping ch2.4 explores is which weights make walking emerge vs. which get
hacked (e.g. a huge forward weight => a policy that dives forward and faceplants).

Termination vs truncation:

    terminated  the robot fell: torso height < FALL_HEIGHT, OR it tipped past
                up_z < UPRIGHT_MIN (flipped / on its side) -- a real failure.
    truncated   the step budget ran out (step_count >= MAX_STEPS) still upright
                -- a time limit, NOT a failure. Bootstrap the value here.

`step()` returns the cartpole/pusher-reach 4-tuple `(obs, reward, done, info)`
with `done = terminated or truncated`; `info` carries the two flags separately,
the per-term reward breakdown, plus height / up_z / forward_vel for logging.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

_XML_PATH = Path(__file__).parent / "quadruped.xml"

# Joint order used everywhere (obs, action, DEFAULT_POSE). Diagonal trot pairs
# are {FL, HR} and {FR, HL}.
JOINT_NAMES = (
    "FL_hip", "FL_knee", "FR_hip", "FR_knee",
    "HL_hip", "HL_knee", "HR_hip", "HR_knee",
)

# Nominal standing pose (a gentle crouch); hip 0.6 / knee -1.2 per leg puts each
# foot roughly under its hip. Reset places the torso at STAND_HEIGHT so the feet
# just touch the floor, then adds small seeded joint noise around this pose.
DEFAULT_HIP = 0.6
DEFAULT_KNEE = -1.2
DEFAULT_POSE = np.array(
    [DEFAULT_HIP if "hip" in n else DEFAULT_KNEE for n in JOINT_NAMES],
    dtype=np.float64,
)

# Rerun logging follows .claude/skills/rerun-instrument: world/objects/**,
# world/robot/**, policy/action, timeline "sim_time". `import rerun` stays lazy
# so CI without rerun installed never pays for it.


class QuadrupedEnv:
    OBS_DIM = 23
    ACT_DIM = 8
    MAX_STEPS = 500          # 10 s of sim time at 50 Hz control
    CONTROL_HZ = 50
    FRAME_SKIP = 4           # 200 Hz physics / 50 Hz control (timestep 0.005 s)

    # Nominal standing pose (a gentle crouch) — see module-level DEFAULT_POSE.
    DEFAULT_POSE = DEFAULT_POSE
    STAND_HEIGHT = 0.257     # torso-center z at which the crouch's feet touch z=0
    ACTION_SCALE = 0.5       # action in [-1,1] -> +-0.5 rad target offset per joint
    RESET_NOISE = 0.05       # rad; reset joint-angle jitter uniform[-b, b]

    # ------------------------------------------------------------- reward terms
    # (ch2.4 shapes these; each is logged separately in info["reward_terms"])
    TARGET_HEIGHT = 0.25     # m; height the height-penalty is measured against
    MAX_VX = 1.0             # m/s; forward-velocity reward is clipped to +-this
    W_FORWARD = 1.0          # weight on clip(vx)      -- the "go forward" drive
    W_UPRIGHT = 0.2          # weight on up_z          -- "stay level"
    W_HEIGHT = 5.0           # weight on -(h - h*)^2   -- "hold your ride height"
    W_ALIVE = 0.2            # per-step "not fallen" bonus
    W_CTRL = 0.001           # weight on -||action||^2 -- "don't thrash"

    # ------------------------------------------------------------- termination
    FALL_HEIGHT = 0.14       # m; torso this low => fallen (crouch rides ~0.25)
    UPRIGHT_MIN = 0.4        # up_z below this => tipped past ~66 deg => fallen

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
        self.data = mujoco.MjData(self.model)
        self._jadr = np.array(
            [self.model.joint(n).qposadr[0] for n in JOINT_NAMES]
        )
        self._vadr = np.array(
            [self.model.joint(n).dofadr[0] for n in JOINT_NAMES]
        )
        self._root_qadr = self.model.joint("root").qposadr[0]  # free joint: qpos[0:7]
        self._root_vadr = self.model.joint("root").dofadr[0]   # free joint: qvel[0:6]
        self._torso_id = self.model.body("torso").id
        self._step_count = 0
        self._renderer: mujoco.Renderer | None = None
        self._rr = None  # rerun module once enabled

    # ---------------------------------------------------------------- state

    @property
    def joint_angles(self) -> np.ndarray:
        return self.data.qpos[self._jadr].copy()

    @property
    def joint_vels(self) -> np.ndarray:
        return self.data.qvel[self._vadr].copy()

    @property
    def torso_height(self) -> float:
        return float(self.data.qpos[self._root_qadr + 2])

    @property
    def torso_up(self) -> np.ndarray:
        """World-frame torso z-axis (the 'up' vector). (0,0,1) = upright."""
        return self.data.xmat[self._torso_id].reshape(3, 3)[:, 2].copy()

    @property
    def torso_linvel(self) -> np.ndarray:
        """World-frame torso linear velocity [vx, vy, vz]; vx is forward."""
        return self.data.qvel[self._root_vadr:self._root_vadr + 3].copy()

    @property
    def forward_vel(self) -> float:
        return float(self.data.qvel[self._root_vadr])  # world +x

    def _obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.joint_angles,          # 0..7
                self.joint_vels,            # 8..15
                [self.torso_height],        # 16
                self.torso_up,              # 17..19
                self.torso_linvel,          # 20..22
            ]
        ).astype(np.float32)

    def _fallen(self) -> bool:
        return self.torso_height < self.FALL_HEIGHT or self.torso_up[2] < self.UPRIGHT_MIN

    # ------------------------------------------------------------- reset/step

    def reset(self, seed: int) -> np.ndarray:
        rng = np.random.Generator(np.random.PCG64(seed))
        mujoco.mj_resetData(self.model, self.data)

        # floating base: place the torso at the standing height, upright, at rest
        self.data.qpos[self._root_qadr:self._root_qadr + 3] = [0.0, 0.0, self.STAND_HEIGHT]
        self.data.qpos[self._root_qadr + 3:self._root_qadr + 7] = [1.0, 0.0, 0.0, 0.0]  # wxyz

        # legs: nominal crouch + small seeded joint-angle noise (still a stand)
        noise = rng.uniform(-self.RESET_NOISE, self.RESET_NOISE, size=self.ACT_DIM)
        self.data.qpos[self._jadr] = self.DEFAULT_POSE + noise
        self.data.qvel[:] = 0.0
        # PD servos hold the (noiseless) nominal pose at reset -> action 0 = stand
        self.data.ctrl[:] = self.DEFAULT_POSE
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        if self._rr is not None:
            self._log_rerun(action=None)
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        # residual position control: target = nominal crouch + scaled offset
        self.data.ctrl[:] = self.DEFAULT_POSE + self.ACTION_SCALE * action
        for _ in range(self.FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        reward, terms = self._reward(action)
        terminated = self._fallen()
        truncated = self._step_count >= self.MAX_STEPS
        done = terminated or truncated
        info = {
            "terminated": terminated,
            "truncated": truncated,
            "reward_terms": terms,
            "height": self.torso_height,
            "up_z": float(self.torso_up[2]),
            "forward_vel": self.forward_vel,
        }

        if self._rr is not None:
            self._log_rerun(action=action)
        return self._obs(), float(reward), done, info

    def _reward(self, action: np.ndarray) -> tuple[float, dict]:
        """The shapeable locomotion reward: five named, weighted terms.

        ch2.4 reweights/ablates these (they are returned individually in
        info["reward_terms"]). Defaults make standing positive and walking
        strictly better; see the module docstring / README for the design.
        """
        up_z = float(self.torso_up[2])
        vx = float(np.clip(self.forward_vel, -self.MAX_VX, self.MAX_VX))
        terms = {
            "forward": self.W_FORWARD * vx,
            "upright": self.W_UPRIGHT * up_z,
            "height": -self.W_HEIGHT * (self.torso_height - self.TARGET_HEIGHT) ** 2,
            "alive": self.W_ALIVE,
            "ctrl": -self.W_CTRL * float(np.sum(action ** 2)),
        }
        return float(sum(terms.values())), terms

    # ---------------------------------------------------------------- render

    def render_frame(self, height: int = 240, width: int = 320) -> np.ndarray:
        """Side-view RGB frame (H, W, 3) uint8."""
        if (self._renderer is None
                or (self._renderer.height, self._renderer.width) != (height, width)):
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera="side")
        return self._renderer.render()

    # ----------------------------------------------------------------- rerun

    def enable_rerun(self, path: str | Path | None = None, spawn: bool = False) -> None:
        """Log to rerun per the repo-wide entity-path conventions. Off by default."""
        import rerun as rr  # lazy: CI without rerun never imports it

        self._rr = rr
        rr.init("zero2robot/quadruped", spawn=spawn)
        if path is not None:
            rr.save(str(path))
        rr.log(
            "world/robot/torso",
            rr.Boxes3D(centers=[(0, 0, 0)], half_sizes=[(0.18, 0.09, 0.035)],
                       colors=(77, 89, 153)),
            static=True,
        )

    def _log_rerun(self, action: np.ndarray | None) -> None:
        rr = self._rr
        rr.set_time("sim_time", duration=self.data.time)
        pos = self.data.qpos[self._root_qadr:self._root_qadr + 3]
        quat = self.data.qpos[self._root_qadr + 3:self._root_qadr + 7]  # wxyz
        rr.log(
            "world/robot/torso",
            rr.Transform3D(translation=tuple(pos),
                           rotation=rr.Quaternion(xyzw=[quat[1], quat[2], quat[3], quat[0]])),
        )
        # feet as points (the only contacts)
        feet = np.array([self.data.geom(f"{leg}_foot").xpos
                         for leg in ("FL", "FR", "HL", "HR")])
        rr.log("world/robot/feet", rr.Points3D(feet, radii=0.022, colors=(204, 89, 76)))
        if action is not None:
            rr.log("policy/action", rr.Scalars(action.astype(np.float64)))
            rr.log("train/forward_vel", rr.Scalars([self.forward_vel]))
            rr.log("train/height", rr.Scalars([self.torso_height]))
            rr.log("train/up_z", rr.Scalars([float(self.torso_up[2])]))


# ------------------------------------------------------------------ baselines


def stand_action(env: QuadrupedEnv) -> np.ndarray:
    """Hold the nominal standing pose: zero residual => PD servos hold the crouch.

    The trivial baseline. It exists to prove the reward rewards *staying up*
    (this rides out the full horizon while random flails and falls) and to give
    the ch2.4/2.5 author a non-learned "just stand" reference return. A learned
    policy must at least match this before it earns the forward term.
    """
    return np.zeros(env.ACT_DIM, dtype=np.float32)


def trot_action(env: QuadrupedEnv, hip_amp: float = 0.25, knee_amp: float = 0.5,
                freq: float = 2.5) -> np.ndarray:
    """Open-loop diagonal trot: an offset gait that walks forward (+x).

    Diagonal pairs {FL, HR} and {FR, HL} move in antiphase. Each hip sweeps
    fore/aft (a sinusoid) to push the body forward; each knee flexes on the
    swing half to lift the foot clear. Returns the ACTION-SPACE command (the
    target offset / ACTION_SCALE, clipped), so it drives the env through the
    exact same step() path a policy would.

    It is not learned and not closed-loop (no feedback), yet it both STAYS UP and
    MOVES FORWARD far better than random — the bar ch2.5's policy must clear.
    Phase advances with env._step_count, so a fixed seed gives a fixed gait.
    """
    # phase in control-time (dt_control = FRAME_SKIP * model timestep)
    dt_control = env.FRAME_SKIP * env.model.opt.timestep
    phase = 2.0 * np.pi * freq * env._step_count * dt_control
    leg_phase = {"FL": 0.0, "HR": 0.0, "FR": np.pi, "HL": np.pi}
    offset = np.zeros(env.ACT_DIM, dtype=np.float64)
    for i, name in enumerate(JOINT_NAMES):
        ph = phase + leg_phase[name[:2]]
        if "hip" in name:
            offset[i] = -hip_amp * np.sin(ph)              # sweep back in stance => go +x
        else:  # knee: flex (more negative) on the swing half to lift the foot
            offset[i] = -knee_amp * max(0.0, np.sin(ph))
    return np.clip(offset / env.ACTION_SCALE, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------- demo


def _rollout(env: QuadrupedEnv, seed: int, policy: str,
             rng: np.random.Generator) -> tuple[int, float, float]:
    """Run one episode; return (episode_length, forward_distance_m, return)."""
    env.reset(seed)
    x0 = env.data.qpos[env._root_qadr]
    done, steps, total = False, 0, 0.0
    while not done:
        if policy == "random":
            action = rng.uniform(-1.0, 1.0, size=QuadrupedEnv.ACT_DIM).astype(np.float32)
        elif policy == "stand":
            action = stand_action(env)
        elif policy == "trot":
            action = trot_action(env)
        else:  # "zero" -- raw zero ctrl offset (same as stand here)
            action = np.zeros(QuadrupedEnv.ACT_DIM, dtype=np.float32)
        _, reward, done, _ = env.step(action)
        total += reward
        steps += 1
    forward = float(env.data.qpos[env._root_qadr] - x0)
    return steps, forward, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Quadruped rollout demo / baseline.")
    parser.add_argument("--seed", type=int, default=0, help="base seed (episode i uses seed+i)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--policy", choices=("random", "stand", "trot", "zero"), default="trot")
    parser.add_argument("--rerun", type=str, default=None, help="write an .rrd to this path")
    args = parser.parse_args()

    env = QuadrupedEnv()
    if args.rerun is not None:
        env.enable_rerun(path=args.rerun)

    # A single seeded generator drives the random policy so the whole run is
    # reproducible from (--seed, --episodes, --policy) alone.
    rng = np.random.Generator(np.random.PCG64(args.seed))
    lengths, forwards, returns = [], [], []
    for i in range(args.episodes):
        n, fwd, ret = _rollout(env, args.seed + i, args.policy, rng)
        lengths.append(n)
        forwards.append(fwd)
        returns.append(ret)
    lengths, forwards, returns = np.array(lengths), np.array(forwards), np.array(returns)
    print(f"policy={args.policy}  episodes={args.episodes}  seed={args.seed}")
    print(f"  mean episode length:  {lengths.mean():.1f} +- {lengths.std():.1f}  (cap {QuadrupedEnv.MAX_STEPS})")
    print(f"  mean forward distance:{forwards.mean():+.3f} +- {forwards.std():.3f} m")
    print(f"  mean return:          {returns.mean():.2f} +- {returns.std():.2f}")


if __name__ == "__main__":
    main()
