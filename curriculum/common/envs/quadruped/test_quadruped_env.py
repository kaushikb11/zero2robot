"""Quadruped env: bitwise reset/step determinism WITH GROUND CONTACT (root
CLAUDE.md invariant 2) plus the shapeable-reward / termination contract that
ch2.4 (reward design), ch2.5 (locomotion), and ch2.7 (domain randomization)
rely on.

This test lives beside the env (not under tests/) because the env is shared
infrastructure that must stay self-verifying, and `curriculum/**/tests/` are
human-owned. It is collected by `make check`'s `pytest curriculum` lane.

The determinism tests deliberately exercise the CONTACT-rich regime (feet on the
floor): contact solvers can be a source of nondeterminism, so the value of the
guarantee is that it holds *with* contacts, not just for a floating body.
"""

import sys
from pathlib import Path

import numpy as np

# Resolve `curriculum.*` the same way chapter artifacts do, regardless of the
# pytest import mode.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from curriculum.common.envs.quadruped import (  # noqa: E402
    QuadrupedEnv,
    stand_action,
    trot_action,
)


def _episode_len_fwd(env: QuadrupedEnv, seed: int, policy) -> tuple[int, float]:
    """Run one episode under `policy(env)->action`; return (length, forward_dist)."""
    env.reset(seed)
    x0 = env.data.qpos[env._root_qadr]
    done, steps = False, 0
    while not done:
        _, _, done, _ = env.step(policy(env))
        steps += 1
    return steps, float(env.data.qpos[env._root_qadr] - x0)


# ----------------------------------------------------------------- determinism


def test_reset_bitwise_determinism():
    env_a, env_b = QuadrupedEnv(), QuadrupedEnv()
    for seed in (0, 1, 42, 123456):
        obs_a, obs_b = env_a.reset(seed), env_b.reset(seed)
        assert obs_a.dtype == np.float32 and obs_a.shape == (QuadrupedEnv.OBS_DIM,)
        assert obs_a.tobytes() == obs_b.tobytes(), f"seed {seed} not bitwise equal"


def test_reset_different_seeds_differ():
    env = QuadrupedEnv()
    blobs = {env.reset(seed).tobytes() for seed in (0, 1, 2)}
    assert len(blobs) == 3, "different seeds must give different initial states"


def test_step_determinism_contact_rich():
    """Same seed + same action sequence => byte-identical trajectory, WITH the
    feet in contact with the floor the whole time. This is the load-bearing
    guarantee: contact-rich stepping is bitwise-reproducible twice-run."""
    def rollout() -> tuple[bytes, int]:
        env = QuadrupedEnv()
        env.reset(7)
        rng = np.random.Generator(np.random.PCG64(7))
        frames, contact_steps = [], 0
        for _ in range(QuadrupedEnv.MAX_STEPS):
            # a small random gait around the stance keeps feet loading the floor
            action = (0.4 * rng.uniform(-1.0, 1.0, size=QuadrupedEnv.ACT_DIM)).astype(np.float32)
            obs, _, done, _ = env.step(action)
            frames.append(obs.tobytes())
            contact_steps += int(env.data.ncon > 0)
            if done:
                break
        return b"".join(frames), contact_steps

    traj_a, contacts_a = rollout()
    traj_b, contacts_b = rollout()
    assert traj_a == traj_b, "contact-rich trajectory not bitwise-reproducible twice-run"
    # the guarantee is meaningless if no contacts were actually solved
    assert contacts_a > 10 and contacts_a == contacts_b, "expected sustained foot-floor contact"


def test_scripted_trot_determinism():
    """The scripted trot (a full contact-rich walking rollout) is reproducible."""
    def rollout() -> bytes:
        env = QuadrupedEnv()
        env.reset(3)
        frames = []
        done = False
        while not done:
            obs, _, done, _ = env.step(trot_action(env))
            frames.append(obs.tobytes())
        return b"".join(frames)

    assert rollout() == rollout()


# ------------------------------------------------------------------- contract


def test_obs_layout():
    env = QuadrupedEnv()
    obs = env.reset(0)
    assert obs.shape == (23,)
    np.testing.assert_array_equal(obs[0:8], env.joint_angles.astype(np.float32))
    np.testing.assert_array_equal(obs[8:16], env.joint_vels.astype(np.float32))
    assert obs[16] == np.float32(env.torso_height)
    np.testing.assert_array_equal(obs[17:20], env.torso_up.astype(np.float32))
    np.testing.assert_array_equal(obs[20:23], env.torso_linvel.astype(np.float32))
    # reset is a near-upright crouch: up-vector ~ (0,0,1), height near STAND_HEIGHT
    assert obs[19] > 0.99
    assert abs(obs[16] - QuadrupedEnv.STAND_HEIGHT) < 0.05


def test_reward_is_sum_of_named_terms():
    """Reward equals the sum of the five named, shapeable terms (the ch2.4
    contract): forward + upright + height + alive + ctrl."""
    env = QuadrupedEnv()
    env.reset(0)
    for _ in range(20):
        _, reward, _, info = env.step(trot_action(env))
        terms = info["reward_terms"]
        assert set(terms) == {"forward", "upright", "height", "alive", "ctrl"}
        assert np.isclose(reward, sum(terms.values()), atol=1e-6)


def test_forward_term_rewards_moving_forward():
    """The forward term is positive while walking forward, ~0 while standing."""
    env = QuadrupedEnv()
    env.reset(0)
    trot_forward = np.mean([env.step(trot_action(env))[3]["reward_terms"]["forward"]
                            for _ in range(100)])
    env.reset(0)
    stand_forward = np.mean([env.step(stand_action(env))[3]["reward_terms"]["forward"]
                             for _ in range(100)])
    assert trot_forward > 0.05, f"trot forward term not positive: {trot_forward:.3f}"
    assert trot_forward > stand_forward


def test_stand_truncates_at_horizon():
    """Standing never falls => the episode runs the full horizon and TRUNCATES
    (a time limit), it does not TERMINATE (a fall)."""
    env = QuadrupedEnv()
    steps, _ = _episode_len_fwd(env, 0, stand_action)
    assert steps == QuadrupedEnv.MAX_STEPS
    # re-run to read the final info flags
    env.reset(0)
    done, info = False, {}
    while not done:
        _, _, done, info = env.step(stand_action(env))
    assert info["truncated"] and not info["terminated"]


def test_fall_terminates():
    """A collapsing robot (legs commanded fully up => it drops) TERMINATES on the
    height/orientation fall-check well before the step cap."""
    env = QuadrupedEnv()
    env.reset(0)
    # command a full negative offset on every joint => folds up and collapses
    fold = -np.ones(QuadrupedEnv.ACT_DIM, dtype=np.float32)
    done, info, steps = False, {}, 0
    while not done:
        _, _, done, info = env.step(fold)
        steps += 1
    assert info["terminated"] and not info["truncated"]
    assert steps < QuadrupedEnv.MAX_STEPS


def test_scripted_beats_random():
    """The reward rewards STAYING UP and MOVING FORWARD. Across seeds:
      - the scripted stand rides out the full horizon; random falls earlier
        (scripted stays upright LONGER), and
      - the scripted trot walks forward; random drifts nowhere
        (scripted MOVES FORWARD).
    This is the bar ch2.5's learned policy must clear."""
    env = QuadrupedEnv()
    rng = np.random.Generator(np.random.PCG64(0))

    def random_policy(_env):
        return rng.uniform(-1.0, 1.0, size=QuadrupedEnv.ACT_DIM).astype(np.float32)

    stand = [_episode_len_fwd(env, s, stand_action) for s in range(10)]
    trot = [_episode_len_fwd(env, s, trot_action) for s in range(10)]
    random_ = [_episode_len_fwd(env, s, random_policy) for s in range(10)]

    stand_len = np.mean([n for n, _ in stand])
    trot_len = np.mean([n for n, _ in trot])
    random_len = np.mean([n for n, _ in random_])
    trot_fwd = np.mean([f for _, f in trot])
    random_fwd = np.mean([f for _, f in random_])

    # staying upright: both scripted policies survive longer than random
    assert stand_len == QuadrupedEnv.MAX_STEPS, f"stand should never fall, got {stand_len}"
    assert stand_len > random_len, f"stand {stand_len} not > random {random_len}"
    assert trot_len > random_len, f"trot {trot_len} not > random {random_len}"
    # moving forward: the trot walks; random goes nowhere (or backward)
    assert trot_fwd > 1.0, f"trot forward distance too small: {trot_fwd:.2f} m"
    assert trot_fwd > random_fwd + 1.0, f"trot {trot_fwd:.2f} not >> random {random_fwd:.2f}"
