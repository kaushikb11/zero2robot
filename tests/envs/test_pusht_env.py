"""PushT env: bitwise reset determinism (root CLAUDE.md invariant 2) + expert."""

import numpy as np
import pytest

from curriculum.common.envs.pusht import PushTEnv, ScriptedExpert


def _run_expert(env: PushTEnv, seed: int, noise: float = 0.0) -> dict:
    env.reset(seed)
    expert = ScriptedExpert(noise=noise, seed=seed)
    done, info = False, {}
    while not done:
        _, _, done, info = env.step(expert.action(env))
    return info


def test_reset_bitwise_determinism():
    env_a, env_b = PushTEnv(), PushTEnv()
    for seed in (0, 1, 42, 123456):
        obs_a, obs_b = env_a.reset(seed), env_b.reset(seed)
        assert obs_a.dtype == np.float32 and obs_a.shape == (PushTEnv.OBS_DIM,)
        assert obs_a.tobytes() == obs_b.tobytes(), f"seed {seed} not bitwise equal"


def test_reset_different_seeds_differ():
    env = PushTEnv()
    blobs = {env.reset(seed).tobytes() for seed in (0, 1, 2)}
    assert len(blobs) == 3, "different seeds must give different initial states"


def test_step_determinism():
    actions = np.random.default_rng(7).uniform(-1, 1, (20, 2)).astype(np.float32)

    def rollout() -> np.ndarray:
        env = PushTEnv()
        env.reset(5)
        for a in actions:
            obs, _, _, _ = env.step(a)
        return obs

    assert rollout().tobytes() == rollout().tobytes()


def test_obs_layout():
    env = PushTEnv()
    obs = env.reset(0)
    # pusher/tee slots match the underlying state accessors
    np.testing.assert_array_equal(obs[0:2], env.pusher_pos.astype(np.float32))
    np.testing.assert_array_equal(obs[2:4], env.tee_pose[:2].astype(np.float32))
    # target pose is fixed at (0, 0, yaw=0) -> sin 0, cos 1
    np.testing.assert_array_equal(obs[6:10], np.array([0, 0, 0, 1], dtype=np.float32))
    # sin/cos slots are unit-normalized
    assert np.isclose(obs[4] ** 2 + obs[5] ** 2, 1.0, atol=1e-6)


def test_expert_success():
    env = PushTEnv()
    successes = sum(_run_expert(env, seed)["success"] for seed in range(10))
    assert successes >= 7, f"expert solved only {successes}/10 (seeds 0..9)"


@pytest.mark.slow
def test_expert_success_full():
    env = PushTEnv()
    successes = sum(_run_expert(env, seed)["success"] for seed in range(50))
    assert successes >= 40, f"expert solved only {successes}/50 (seeds 0..49, bar 80%)"
