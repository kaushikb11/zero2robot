"""SUGGESTED exercise candidate (humans promote) — bug-hunt, ch2.4.

Objective tested: the single most common silent bug in a from-scratch PPO — the
truncation-vs-termination conflation in GAE (the ch2.1 lesson, reused unchanged in
`rewards.py`). Nothing crashes, no loss looks wrong; the policy just quietly learns
that surviving to the time limit is worthless, because a *truncated* (out-of-time)
step was masked as if the episode had *terminated* (a real failure) and its
bootstrap value V(next state) was thrown away.

The function below is `compute_gae`, lifted verbatim from `rewards.py` (region
`ppo`) — the only adaptation is that the artifact's globals (`args.gamma`,
`args.gae_lambda`, `args.num_steps`, `args.num_envs`) are passed in / read off the
input shapes, so the function runs standalone with no training. Exactly ONE line
carries an injected bug. Find it and fix it.

This check is DETERMINISTIC: it runs GAE on a hand-built two-episode trajectory,
not on a training run, so it never flakes on RL variance (the same reason ch2.4's
fix-the-hack check and ch2.1's GAE check are pure-function checks). The buggy line
is invisible in any single training curve; it is glaring the moment you feed it one
truncated episode next to one terminated episode and compare.

The trajectory: two envs, two steps, IDENTICAL in every way except the last step —
env 0 is *truncated* (ran out of time while still standing; the future is real, so
bootstrap V(next)) and env 1 is *terminated* (fell; the future really is zero). A
correct GAE credits env 0's last step with the discarded future and env 1's with
nothing, so env 0's advantage must come out LARGER. The bug erases that difference:
it drops env 0's bootstrap too, and the two envs come out identical.

Run:  python ex3_bughunt_gae_truncation.py   (prints both envs' advantages)
Estimated learner time: 25 minutes.
"""

import torch

METADATA = {
    "type": "bug-hunt",
    "chapter": "ch2.4-rewards",
    "buggy_region": "compute_gae (the truncation-vs-termination bootstrap mask)",
}

# Chapter defaults (rewards.py argparse). Kept here so the check is self-contained.
GAMMA, GAE_LAMBDA = 0.99, 0.95

# Before you run anything: predict which env's LAST-STEP advantage should be larger
# — the truncated env (0) or the terminated env (1) — and say why in one sentence.
# Set PREDICTION to "env0" or "env1". (The site enforces predict-before-run; here
# it is honor-system — the check only nudges you to commit before you measure.)
PREDICTION = None  # "env0" | "env1"


def compute_gae(rewards, values, terminated, done, bootstrap, next_value,
                gamma=GAMMA, gae_lambda=GAE_LAMBDA):
    """GAE walking backward. (1 - terminated) masks the bootstrap after a fall; a
    truncated step keeps its stored bootstrap value (learned in ch2.1).

    Lifted from rewards.py `compute_gae`; `args.*` replaced by params / shape reads.
    """
    num_steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(num_envs)
    for t in reversed(range(num_steps)):
        next_v = next_value if t == num_steps - 1 else values[t + 1]
        next_v = torch.where(done[t].bool(), bootstrap[t], next_v)
        delta = rewards[t] + gamma * next_v * (1.0 - done[t]) - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - done[t]) * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def build_trajectory():
    """A fixed two-step, two-env rollout. env 0 truncates at t=1, env 1 terminates
    at t=1; everything else is identical, so the only thing that can separate their
    advantages is whether the last step bootstraps the future. Returns the exact
    inputs `compute_gae` takes in rewards.py, as torch tensors."""
    #                       t=0        t=1
    rewards   = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    values    = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    #  env0 truncated (done, NOT terminated) | env1 terminated (done AND terminated)
    terminated = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    done       = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    # bootstrap = V(real next state) at the boundary; only read on `done` steps.
    bootstrap  = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    next_value = torch.tensor([0.0, 0.0])  # value after the window; unused here (both ended)
    return rewards, values, terminated, done, bootstrap, next_value


if __name__ == "__main__":
    adv, ret = compute_gae(*build_trajectory())
    print("advantages (rows = timestep, cols = [env0 truncated, env1 terminated]):")
    print(adv)
    print(f"\nlast-step advantage:  env0 (truncated) {adv[1, 0]:+.4f}   "
          f"env1 (terminated) {adv[1, 1]:+.4f}")
    if torch.allclose(adv[:, 0], adv[:, 1], atol=1e-4):
        print("=> env0 == env1: the truncated episode was NOT bootstrapped. "
              "A time-limit was scored as a real failure. That is the bug.")
    else:
        print("=> env0 > env1: the truncated episode keeps its future value. Fixed.")
