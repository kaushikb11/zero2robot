"""Pusher-reach env: bitwise reset/step determinism (root CLAUDE.md invariant 2)
plus the dense-reward / termination contract ch2.2's SAC relies on.

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

from curriculum.common.envs.pusher_reach import PusherReachEnv, reach_action  # noqa: E402


def _episode(env: PusherReachEnv, seed: int, policy) -> tuple[float, bool, bytes]:
    """Run one episode under `policy(env)->action`; return (final_dist, success, last-obs bytes)."""
    env.reset(seed)
    done, info, obs = False, {}, None
    while not done:
        obs, _, done, info = env.step(policy(env))
    return info["dist"], info["success"], obs.tobytes()


def test_reset_bitwise_determinism():
    env_a, env_b = PusherReachEnv(), PusherReachEnv()
    for seed in (0, 1, 42, 123456):
        obs_a, obs_b = env_a.reset(seed), env_b.reset(seed)
        assert obs_a.dtype == np.float32 and obs_a.shape == (PusherReachEnv.OBS_DIM,)
        assert obs_a.tobytes() == obs_b.tobytes(), f"seed {seed} not bitwise equal"


def test_reset_different_seeds_differ():
    env = PusherReachEnv()
    blobs = {env.reset(seed).tobytes() for seed in (0, 1, 2)}
    assert len(blobs) == 3, "different seeds must give different initial states"


def test_step_determinism():
    """Same seed + same action sequence => byte-identical trajectory."""
    def rollout() -> bytes:
        env = PusherReachEnv()
        env.reset(7)
        rng = np.random.Generator(np.random.PCG64(7))
        frames = []
        for _ in range(PusherReachEnv.MAX_STEPS):
            action = rng.uniform(-1.0, 1.0, size=PusherReachEnv.ACT_DIM).astype(np.float32)
            obs, reward, done, _ = env.step(action)
            frames.append(obs.tobytes())
            if done:
                break
        return b"".join(frames)

    assert rollout() == rollout()


def test_obs_layout():
    env = PusherReachEnv()
    obs = env.reset(0)
    assert obs[0] == np.float32(np.cos(env.shoulder_angle))
    assert obs[1] == np.float32(np.sin(env.shoulder_angle))
    assert obs[2] == np.float32(np.cos(env.elbow_angle))
    assert obs[3] == np.float32(np.sin(env.elbow_angle))
    assert obs[4] == np.float32(env.shoulder_angvel)
    assert obs[5] == np.float32(env.elbow_angvel)
    # last two entries are the fingertip->target vector; its norm is the distance
    ftt = env.target_pos - env.fingertip_pos
    assert obs[6] == np.float32(ftt[0]) and obs[7] == np.float32(ftt[1])
    assert np.isclose(np.hypot(obs[6], obs[7]), env._dist(), atol=1e-6)


def test_target_within_reach():
    """Every seeded target is strictly inside the arm's reach (IK always solvable)."""
    env = PusherReachEnv()
    reach = 2.0 * PusherReachEnv.LINK_LEN
    for seed in range(20):
        env.reset(seed)
        assert np.linalg.norm(env.target_pos) < reach


def test_dense_reward_is_negative_distance():
    """Reward equals -dist every step (until success latches a one-time bonus)."""
    env = PusherReachEnv()
    env.reset(0)
    done = False
    while not done:
        _, reward, done, info = env.step(np.zeros(PusherReachEnv.ACT_DIM, dtype=np.float32))
        if not info["success"]:
            assert reward == -info["dist"], "pre-success reward must be exactly -dist"


def test_truncation_at_horizon():
    """With no early-termination flag, an episode always runs the full horizon."""
    env = PusherReachEnv()  # terminate_on_success defaults to False
    env.reset(3)
    done, info, steps = False, {}, 0
    while not done:
        _, _, done, info = env.step(reach_action(env))
        steps += 1
    assert steps == PusherReachEnv.MAX_STEPS
    assert info["truncated"] and not info["terminated"]


def test_terminate_on_success_flag():
    """With the flag on, reaching the target ends the episode early."""
    env = PusherReachEnv(terminate_on_success=True)
    env.reset(0)
    done, info, steps = False, {}, 0
    while not done:
        _, _, done, info = env.step(reach_action(env))
        steps += 1
    assert info["terminated"] and info["success"]
    assert steps < PusherReachEnv.MAX_STEPS


def test_scripted_beats_random():
    """The dense reward rewards reaching: a scripted IK+PD reach gets far closer
    than a random policy across seeds (the bar ch2.2's SAC must clear)."""
    env = PusherReachEnv()
    rng = np.random.Generator(np.random.PCG64(0))

    def random_policy(_env):
        return rng.uniform(-1.0, 1.0, size=PusherReachEnv.ACT_DIM).astype(np.float32)

    scripted = [_episode(env, s, reach_action) for s in range(10)]
    random_ = [_episode(env, s, random_policy) for s in range(10)]
    scripted_final = np.mean([d for d, _, _ in scripted])
    random_final = np.mean([d for d, _, _ in random_])
    scripted_succ = sum(ok for _, ok, _ in scripted)

    assert scripted_final < 0.25 * random_final, (
        f"scripted final {scripted_final:.4f} not << random {random_final:.4f}"
    )
    assert scripted_succ >= 9, f"scripted reach should succeed on >=9/10, got {scripted_succ}"
