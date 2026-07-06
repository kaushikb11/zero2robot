"""AlohaCube env: bitwise reset/step determinism (root CLAUDE.md invariant 2)
plus the bimanual-handoff contract the scripted expert relies on.

This test lives beside the env (not under tests/) because the env is shared
infrastructure that must stay self-verifying, and `curriculum/**/tests/` are
human-owned. It is collected by `make check`'s `pytest curriculum` lane.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Resolve `curriculum.*` the same way chapter artifacts do, regardless of the
# pytest import mode.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from curriculum.common.envs.aloha_cube import AlohaCubeEnv, ScriptedExpert  # noqa: E402


def _run_expert(env: AlohaCubeEnv, seed: int, noise: float = 0.0) -> dict:
    env.reset(seed)
    expert = ScriptedExpert(noise=noise, seed=seed)
    done, info = False, {}
    while not done:
        _, _, done, info = env.step(expert.action(env))
    return info


def test_reset_bitwise_determinism():
    env_a, env_b = AlohaCubeEnv(), AlohaCubeEnv()
    for seed in (0, 1, 42, 123456):
        obs_a, obs_b = env_a.reset(seed), env_b.reset(seed)
        assert obs_a.dtype == np.float32 and obs_a.shape == (AlohaCubeEnv.OBS_DIM,)
        assert obs_a.tobytes() == obs_b.tobytes(), f"seed {seed} not bitwise equal"


def test_reset_different_seeds_differ():
    env = AlohaCubeEnv()
    blobs = {env.reset(seed).tobytes() for seed in (0, 1, 2)}
    assert len(blobs) == 3, "different seeds must give different initial states"


def test_step_determinism():
    """Same seed + same expert => byte-identical rollout (welds toggle and all)."""
    def rollout() -> bytes:
        env = AlohaCubeEnv()
        env.reset(5)
        expert = ScriptedExpert(seed=5)
        done = False
        obs = None
        while not done:
            obs, _, done, _ = env.step(expert.action(env))
        return obs.tobytes()

    assert rollout() == rollout()


def test_obs_layout():
    env = AlohaCubeEnv()
    obs = env.reset(0)
    np.testing.assert_array_equal(obs[0:2], env.right_ee_pos.astype(np.float32))
    np.testing.assert_array_equal(obs[3:5], env.left_ee_pos.astype(np.float32))
    np.testing.assert_array_equal(obs[6:8], env.cube_pos.astype(np.float32))
    # target is fixed in the left arm's exclusive reach
    np.testing.assert_array_equal(obs[8:10], AlohaCubeEnv.TARGET_XY.astype(np.float32))
    # grippers reset open
    assert obs[2] == 0.0 and obs[5] == 0.0


def test_reach_forces_handoff():
    """Neither arm can span the task alone: the cube spawns out of the left
    arm's reach and the target sits out of the right arm's reach."""
    env = AlohaCubeEnv()
    for seed in range(10):
        env.reset(seed)
        cube_x = env.cube_pos[0]
        assert cube_x > 0.05, "cube must spawn in the right arm's exclusive zone"
    assert AlohaCubeEnv.TARGET_XY[0] < -0.05, "target must be left-arm exclusive"


def test_expert_transfers_and_succeeds():
    env = AlohaCubeEnv()
    successes = sum(_run_expert(env, seed)["success"] for seed in range(10))
    assert successes >= 8, f"expert solved only {successes}/10 (seeds 0..9)"


@pytest.mark.slow
def test_expert_success_full():
    env = AlohaCubeEnv()
    successes = sum(_run_expert(env, seed)["success"] for seed in range(50))
    assert successes >= 45, f"expert solved only {successes}/50 (seeds 0..49, bar 90%)"
