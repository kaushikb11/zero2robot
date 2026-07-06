"""Cartpole: balance a hinged pole upright by pushing a cart. zero2robot Phase-2 env.

The classic control task and ch2.1's PPO smoke env (CleanRL philosophy: the
first thing you run PPO on is CartPole). A cart slides on a rail; a pole is
hinged to it and falls under gravity. The policy applies a horizontal force to
the cart and must keep the pole upright and the cart on the rail. Underactuated:
one actuator, two degrees of freedom — you can only move the cart, never the
pole directly.

Infra-grade env, mirroring PushT / AlohaCube discipline (reset/step/obs/render
API, bitwise CPU determinism, a `--seed`-able rollout demo). Continuous action
is a deliberate choice for PPO's Gaussian policy — see README.

Deterministic on CPU: two fresh envs reset with the same seed produce
byte-identical observations (CI-enforced, root CLAUDE.md invariant 2). All
randomness flows through np.random.Generator(PCG64(seed)); no global RNG, no
wall clock. The scene has no collisions, so the dynamics are a pure inverted
pendulum with no contact solver — cheap and reproducible.

Observation (float32[5]) -- ch2.1's PPO trains on exactly this layout. The pole
angle is encoded as (cos, sin) of the angle from upright so the observation is
continuous through the vertical (no +-pi wrap seam):

    idx  name              units
    0    cart_pos          m       (0 = rail center; + is toward +x)
    1    cart_vel          m/s
    2    cos(pole_angle)   -       (+1.0 exactly upright)
    3    sin(pole_angle)   -       (0.0 upright; + when the tip leans toward +x)
    4    pole_angvel       rad/s

Action (float32[1]): horizontal force on the cart, clipped to [-1, 1] and
applied through the MuJoCo motor (gear 10 => +-10 N). Held for FRAME_SKIP = 2
physics steps (50 Hz control over 100 Hz physics, timestep 0.01 s).

Reward: +1.0 for every step the pole stays up and the cart stays on the rail
(the classic CartPole "alive bonus"). Episode return therefore equals the
number of steps survived, capped at MAX_STEPS. Balancing longer is the only way
to score higher, so the reward directly rewards balancing.

Termination vs truncation (PPO needs the distinction to bootstrap correctly):

    terminated  the pole fell (|pole_angle| > ANGLE_LIMIT) OR the cart ran off
                the rail (|cart_pos| > CART_LIMIT) -- a real failure state.
    truncated   the step budget ran out (step_count >= MAX_STEPS) with the pole
                still up -- a time limit, NOT a failure. Bootstrap value here.

`step()` returns the PushT/AlohaCube 4-tuple `(obs, reward, done, info)` with
`done = terminated or truncated`; `info` carries the two flags separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

_XML_PATH = Path(__file__).parent / "cartpole.xml"

# Rerun logging follows .claude/skills/rerun-instrument: world/objects/**,
# world/robot/**, policy/action, timeline "sim_time". `import rerun` stays
# lazy so CI without rerun installed never pays for it.


def wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class CartpoleEnv:
    OBS_DIM = 5
    ACT_DIM = 1
    MAX_STEPS = 500          # classic CartPole-v1 horizon (10 s at 50 Hz control)
    CONTROL_HZ = 50
    FRAME_SKIP = 2           # 100 Hz physics / 50 Hz control (timestep 0.01 s)
    ANGLE_LIMIT = 0.2095     # rad (~12 deg from upright) -- pole-fall threshold
    CART_LIMIT = 2.4         # m -- cart-off-rail threshold
    RESET_BOUND = 0.05       # reset draws each state var from uniform[-b, b]

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
        self.data = mujoco.MjData(self.model)
        self._jadr = {
            name: self.model.joint(name).qposadr[0]
            for name in ("slider", "hinge")
        }
        self._vadr = {
            name: self.model.joint(name).dofadr[0]
            for name in ("slider", "hinge")
        }
        self._step_count = 0
        self._renderer: mujoco.Renderer | None = None
        self._rr = None  # rerun module once enabled

    # ---------------------------------------------------------------- state

    @property
    def cart_pos(self) -> float:
        return float(self.data.qpos[self._jadr["slider"]])

    @property
    def cart_vel(self) -> float:
        return float(self.data.qvel[self._vadr["slider"]])

    @property
    def pole_angle(self) -> float:
        """Angle from upright, wrapped to [-pi, pi). 0 = vertical."""
        return wrap_angle(float(self.data.qpos[self._jadr["hinge"]]))

    @property
    def pole_angvel(self) -> float:
        return float(self.data.qvel[self._vadr["hinge"]])

    def _obs(self) -> np.ndarray:
        theta = self.pole_angle
        return np.array(
            [self.cart_pos, self.cart_vel, np.cos(theta), np.sin(theta), self.pole_angvel],
            dtype=np.float32,
        )

    def _fallen(self) -> bool:
        return abs(self.pole_angle) > self.ANGLE_LIMIT or abs(self.cart_pos) > self.CART_LIMIT

    # ------------------------------------------------------------- reset/step

    def reset(self, seed: int) -> np.ndarray:
        rng = np.random.Generator(np.random.PCG64(seed))
        mujoco.mj_resetData(self.model, self.data)

        # classic init: every state var uniform in [-RESET_BOUND, RESET_BOUND]
        cart_pos, pole_angle = rng.uniform(-self.RESET_BOUND, self.RESET_BOUND, size=2)
        cart_vel, pole_angvel = rng.uniform(-self.RESET_BOUND, self.RESET_BOUND, size=2)
        self.data.qpos[self._jadr["slider"]] = cart_pos
        self.data.qpos[self._jadr["hinge"]] = pole_angle
        self.data.qvel[self._vadr["slider"]] = cart_vel
        self.data.qvel[self._vadr["hinge"]] = pole_angvel
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        if self._rr is not None:
            self._log_rerun(action=None)
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        self.data.ctrl[0] = action[0]
        for _ in range(self.FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        terminated = self._fallen()
        truncated = self._step_count >= self.MAX_STEPS
        reward = 1.0  # alive bonus: +1 for surviving this step
        done = terminated or truncated
        info = {
            "terminated": terminated,
            "truncated": truncated,
            "pole_angle": self.pole_angle,
            "cart_pos": self.cart_pos,
        }

        if self._rr is not None:
            self._log_rerun(action=action)
        return self._obs(), float(reward), done, info

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
        rr.init("zero2robot/cartpole", spawn=spawn)
        if path is not None:
            rr.save(str(path))
        rr.log(
            "world/robot/cart",
            rr.Boxes3D(centers=[(0, 0, 0)], half_sizes=[(0.1, 0.05, 0.05)],
                       colors=(77, 89, 153)),
            static=True,
        )
        rr.log(
            "world/objects/pole",
            rr.Capsules3D(lengths=[1.0], radii=[0.02], colors=(204, 89, 76)),
            static=True,
        )

    def _log_rerun(self, action: np.ndarray | None) -> None:
        rr = self._rr
        rr.set_time("sim_time", duration=self.data.time)
        x = self.cart_pos
        rr.log("world/robot/cart", rr.Transform3D(translation=(x, 0.0, 0.0)))
        rr.log(
            "world/objects/pole",
            rr.Transform3D(
                translation=(x, 0.0, 0.0),
                rotation=rr.RotationAxisAngle(axis=(0, 1, 0), radians=self.pole_angle),
            ),
        )
        if action is not None:
            rr.log("policy/action", rr.Scalars(action.astype(np.float64)))
            rr.log("train/pole_angle", rr.Scalars([self.pole_angle]))
            rr.log("train/cart_pos", rr.Scalars([self.cart_pos]))


def balance_action(env: CartpoleEnv) -> np.ndarray:
    """A hand-tuned "push toward upright" balance policy (baseline, not learned).

    Linear feedback on the four physical state variables — a textbook pole
    balancer. It exists to (a) prove the reward really rewards balancing (this
    beats random by a wide margin) and (b) give the ch2.1 author a non-learned
    reference return. PPO should match or beat it.

    Push in the direction the pole is falling (positive gain on angle + angle
    rate) with a gentle pull back toward the rail center.
    """
    theta = env.pole_angle
    theta_dot = env.pole_angvel
    u = 10.0 * theta + 2.0 * theta_dot + 0.4 * env.cart_pos + 0.8 * env.cart_vel
    return np.array([np.clip(u, -1.0, 1.0)], dtype=np.float32)


# --------------------------------------------------------------------- demo


def _rollout(env: CartpoleEnv, seed: int, policy: str, rng: np.random.Generator) -> int:
    """Run one episode; return its length (== return, since reward is +1/step)."""
    env.reset(seed)
    done, steps = False, 0
    while not done:
        if policy == "random":
            action = rng.uniform(-1.0, 1.0, size=CartpoleEnv.ACT_DIM).astype(np.float32)
        elif policy == "scripted":
            action = balance_action(env)
        else:  # "zero" -- apply no force (pole free-falls)
            action = np.zeros(CartpoleEnv.ACT_DIM, dtype=np.float32)
        _, _, done, _ = env.step(action)
        steps += 1
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Cartpole rollout demo / baseline.")
    parser.add_argument("--seed", type=int, default=0, help="base seed (episode i uses seed+i)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--policy", choices=("random", "scripted", "zero"), default="random")
    parser.add_argument("--rerun", type=str, default=None, help="write an .rrd to this path")
    args = parser.parse_args()

    env = CartpoleEnv()
    if args.rerun is not None:
        env.enable_rerun(path=args.rerun)

    # A single seeded generator drives the random policy so the whole run is
    # reproducible from (--seed, --episodes, --policy) alone.
    rng = np.random.Generator(np.random.PCG64(args.seed))
    lengths = [_rollout(env, args.seed + i, args.policy, rng) for i in range(args.episodes)]
    lengths = np.array(lengths)
    print(f"policy={args.policy}  episodes={args.episodes}  seed={args.seed}")
    print(f"  mean return (== episode length): {lengths.mean():.1f} +- {lengths.std():.1f}")
    print(f"  min / max: {lengths.min()} / {lengths.max()}  (cap {CartpoleEnv.MAX_STEPS})")


if __name__ == "__main__":
    main()
