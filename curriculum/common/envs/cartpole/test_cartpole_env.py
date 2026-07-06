"""Cartpole env: bitwise reset/step determinism (root CLAUDE.md invariant 2)
plus the reward/termination contract ch2.1's PPO relies on.

This test lives beside the env (not under tests/) because the env is shared
infrastructure that must stay self-verifying, and `curriculum/**/tests/` are
human-owned. It is collected by `make check`'s `pytest curriculum` lane.
"""

import sys
from pathlib import Path

import numpy as np

# Resolve `curriculum.*` the same way chapter artifacts do, regardless of the
# pytest import mode.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from curriculum.common.envs.cartpole import CartpoleEnv, balance_action  # noqa: E402


def _rollout_len(env: CartpoleEnv, seed: int, policy) -> tuple[int, bytes]:
    """Run one episode under `policy(env)->action`; return (length, last-obs bytes)."""
    env.reset(seed)
    done, steps = False, 0
    obs = None
    while not done:
        obs, _, done, _ = env.step(policy(env))
        steps += 1
    return steps, obs.tobytes()


def test_reset_bitwise_determinism():
    env_a, env_b = CartpoleEnv(), CartpoleEnv()
    for seed in (0, 1, 42, 123456):
        obs_a, obs_b = env_a.reset(seed), env_b.reset(seed)
        assert obs_a.dtype == np.float32 and obs_a.shape == (CartpoleEnv.OBS_DIM,)
        assert obs_a.tobytes() == obs_b.tobytes(), f"seed {seed} not bitwise equal"


def test_reset_different_seeds_differ():
    env = CartpoleEnv()
    blobs = {env.reset(seed).tobytes() for seed in (0, 1, 2)}
    assert len(blobs) == 3, "different seeds must give different initial states"


def test_step_determinism():
    """Same seed + same action sequence => byte-identical trajectory."""
    def rollout() -> bytes:
        env = CartpoleEnv()
        env.reset(7)
        rng = np.random.Generator(np.random.PCG64(7))
        frames = []
        for _ in range(50):
            action = rng.uniform(-1.0, 1.0, size=CartpoleEnv.ACT_DIM).astype(np.float32)
            obs, reward, done, _ = env.step(action)
            frames.append(obs.tobytes())
            if done:
                break
        return b"".join(frames)

    assert rollout() == rollout()


def test_obs_layout():
    env = CartpoleEnv()
    obs = env.reset(0)
    assert obs[0] == np.float32(env.cart_pos)
    assert obs[1] == np.float32(env.cart_vel)
    assert obs[2] == np.float32(np.cos(env.pole_angle))
    assert obs[3] == np.float32(np.sin(env.pole_angle))
    assert obs[4] == np.float32(env.pole_angvel)
    # reset stays within the small init band (pole near upright => cos ~ 1)
    assert abs(env.pole_angle) < env.RESET_BOUND + 1e-6
    assert obs[2] > 0.99


def test_alive_reward_and_return_equals_length():
    """Reward is +1 per surviving step, so return == episode length."""
    env = CartpoleEnv()
    env.reset(0)
    total, done, steps = 0.0, False, 0
    while not done:
        _, reward, done, _ = env.step(balance_action(env))
        assert reward == 1.0
        total += reward
        steps += 1
    assert total == float(steps)


def test_termination_flags_mutually_consistent():
    """done == terminated or truncated; truncation only at the step cap."""
    env = CartpoleEnv()
    env.reset(3)
    done, info, steps = False, {}, 0
    while not done:
        _, _, done, info = env.step(np.zeros(CartpoleEnv.ACT_DIM, dtype=np.float32))
        steps += 1
    # zero force => the pole free-falls and TERMINATES well before the cap
    assert info["terminated"] and not info["truncated"]
    assert steps < CartpoleEnv.MAX_STEPS


def test_scripted_beats_random():
    """The reward rewards balancing: a scripted balancer survives far longer
    than a random policy across seeds (the bar ch2.1's PPO must clear)."""
    env = CartpoleEnv()
    rng = np.random.Generator(np.random.PCG64(0))

    def random_policy(_env):
        return rng.uniform(-1.0, 1.0, size=CartpoleEnv.ACT_DIM).astype(np.float32)

    scripted = [_rollout_len(env, s, balance_action)[0] for s in range(10)]
    random_ = [_rollout_len(env, s, random_policy)[0] for s in range(10)]
    assert np.mean(scripted) > 5.0 * np.mean(random_), (
        f"scripted {np.mean(scripted):.1f} not >> random {np.mean(random_):.1f}"
    )
    # the balancer should routinely ride out the full horizon
    assert np.mean(scripted) > 400.0, f"scripted balancer too weak: {np.mean(scripted):.1f}"
