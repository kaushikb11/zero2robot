"""SUGGESTED exercise candidate (humans promote) — code-completion, ch2.1.

Objective tested: the one line cartpole forces you to get right — bootstrapping
on TRUNCATION but not on TERMINATION. You reimplement Generalized Advantage
Estimation as a pure function; checks.py drives it with a hand-built rollout
that contains one fall (terminated) and one time-out (truncated), so the ONLY
way to pass is to treat them differently — exactly what ppo.py's
compute_advantages does.

This check is DETERMINISTIC: it runs on fixed arrays, not on a training run, so
it never flakes on RL variance. That is the point — the *algorithm* is
checkable even though the *metric it produces* is noisy.

Run:  python ex1_completion_gae.py     (prints advantages for a toy rollout)
Estimated learner time: 25 minutes.
"""

import numpy as np

METADATA = {
    "type": "code-completion",
    "chapter": "ch2.1-ppo",
    "blanked_region": "GAE bootstrap / truncation mask",
}


def compute_gae(rewards, values, terminated, done, bootstrap, next_value,
                gamma: float, gae_lambda: float) -> np.ndarray:
    """Generalized Advantage Estimation over one rollout, walking backward.

    Shapes are (T,) 1-D arrays for a single env. Your job mirrors ppo.py's
    compute_advantages exactly:

      - `values[t]`    is V(obs[t]); `next_value` is V of the obs AFTER the last
        step (the live continuation).
      - `bootstrap[t]` is V(the real next state) for steps that ENDED an episode
        (done[t] == 1) — the value you cut off. Ignore it when done[t] == 0.
      - `terminated[t]` is 1 only for a TRUE terminal (the pole fell): the future
        value ahead of it is 0, so it must be MASKED out of the bootstrap.
      - `done[t]` is 1 for ANY episode boundary (terminated OR truncated): it
        resets the GAE lambda accumulation so advantage never leaks across
        episodes.

    The recursion, per step t (from the last step backward):
        next_v   = bootstrap[t]  if done[t]  else  (values[t+1] or next_value)
        delta    = rewards[t] + gamma * next_v * (1 - terminated[t]) - values[t]
        adv[t]   = delta + gamma * gae_lambda * (1 - done[t]) * adv[t+1]

    A TRUNCATED step has done==1 but terminated==0, so its bootstrap survives the
    (1 - terminated) mask — that is the whole lesson. Return the (T,) advantages.

    Rough size: 8-10 lines.
    """
    # YOUR CODE HERE (delete the next line once you start)
    raise NotImplementedError("implement GAE — bootstrap on truncation, not on termination")


if __name__ == "__main__":
    # A 6-step toy rollout: it falls at t=2 (terminated) and times out at t=5
    # (truncated). If your bootstrap masking is right, the advantage at the
    # truncated step is clearly larger than at the terminated step.
    rewards = np.ones(6, dtype=np.float64)
    values = np.array([5.0, 4.0, 3.0, 6.0, 5.0, 4.0])
    terminated = np.array([0, 0, 1, 0, 0, 0], dtype=np.float64)
    done = np.array([0, 0, 1, 0, 0, 1], dtype=np.float64)
    bootstrap = np.array([0, 0, 0.0, 0, 0, 10.0])  # V(real next state) at the time-out
    adv = compute_gae(rewards, values, terminated, done, bootstrap,
                      next_value=4.0, gamma=0.99, gae_lambda=0.95)
    print("advantages:", np.round(adv, 3))
    print(f"terminated-step adv {adv[2]:+.3f} vs truncated-step adv {adv[5]:+.3f} "
          "(the time-out should look far better than the fall)")
