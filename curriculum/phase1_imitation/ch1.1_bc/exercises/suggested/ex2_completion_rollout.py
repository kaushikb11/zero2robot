"""SUGGESTED exercise candidate (humans promote) — code-completion, ch1.1.

Objective tested: run the full loop — an evaluation rollout is the loop from
chapter 0.1 with a policy writing the ctrl line. You get the signature and
the docstring; the body is yours.

Everything around the blank is real: checks.py drives your rollout with the
scripted expert (which should succeed every episode) and with a do-nothing
policy (which should never succeed). If both behave, your loop is right —
same reset seeding, same step contract, same return accounting as bc.py.

Run:  python ex2_completion_rollout.py      (rolls out the scripted expert)
Estimated learner time: 20 minutes.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from curriculum.common.envs.pusht import PushTEnv, ScriptedExpert  # noqa: E402

METADATA = {
    "type": "code-completion",
    "chapter": "ch1.1-bc",
    "blanked_region": "eval rollout loop",
}


def rollout_episode(policy, env: PushTEnv, seed: int) -> tuple[bool, float]:
    """Run ONE evaluation episode; return (success, episode_return).

    `policy` is a callable mapping a raw observation float32[10] to an action
    float32[2] (exactly what bc.py's trained BCPolicy is, minus the tensor
    plumbing). Your job, mirroring bc.py's eval region:

      1. reset the env with `seed` and keep the first observation
      2. until the episode is done: ask the policy for an action, step the
         env, accumulate the reward into episode_return
      3. return (info["success"], episode_return) from the FINAL step —
         success latches inside the env, so the last info tells the truth

    Rough size: 6-8 lines. No torch needed — the policy handles itself.
    """
    # YOUR CODE HERE (delete the next line once you start)
    raise NotImplementedError("complete the rollout loop: obs -> action -> step")


if __name__ == "__main__":
    # Smoke-run your implementation with the scripted expert standing in for
    # a trained policy. Expert success is 100% on these seeds (env README).
    env = PushTEnv()
    results = []
    for episode in range(5):
        expert = ScriptedExpert(seed=episode)  # expert reads env state directly;
        results.append(rollout_episode(lambda obs: expert.action(env), env, seed=10_000 + episode))
    for episode, (success, episode_return) in enumerate(results):
        print(f"episode {episode}: {'success' if success else 'FAIL'}  return {episode_return:.2f}")
    if not all(success for success, _ in results):
        print("the expert never fails from these seeds — the loop, not the expert, is the suspect")
    print(f"mean return {np.mean([r for _, r in results]):.2f} "
          "(expert reference is around -6; a big negative number means the episode timed out)")
