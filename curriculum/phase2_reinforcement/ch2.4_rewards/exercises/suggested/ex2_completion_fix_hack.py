"""SUGGESTED exercise candidate (humans promote) — code-completion, ch2.4.

Objective tested: fixing a reward hack is a PROGRAMMING fix — you say what you
actually meant, you don't "optimize harder". The chapter's `r_hack` rewards raw
height and the policy rears up and goes nowhere. Your job: complete `fixed_reward`
so the program also values FORWARD progress, turning the height-hack into a reward
that would actually train a walk.

This check is DETERMINISTIC: it evaluates your reward function on hand-built state
dicts, not on a training run, so it never flakes on RL variance (the same reason
ch2.1's GAE exercise is a pure-function check). The property it verifies is the
one that matters — your fixed reward must RESPOND to forward velocity, which the
broken height-only reward does not.

Run:  python ex2_completion_fix_hack.py   (prints your reward on a few states)
Estimated learner time: 20 minutes.
"""

METADATA = {
    "type": "code-completion",
    "chapter": "ch2.4-rewards",
    "blanked_region": "the forward term that fixes the height hack",
}

# The env's reward-term weights, mirrored from QuadrupedEnv so this exercise is
# self-contained (no training, no import of the artifact). These are the SAME
# formulas the env uses to fill info["reward_terms"].
W_FORWARD, W_UPRIGHT, W_HEIGHT, W_ALIVE, W_CTRL = 1.0, 0.2, 5.0, 0.2, 0.001
TARGET_HEIGHT, MAX_VX = 0.25, 1.0
HACK_HEIGHT_W = 10.0  # the broken design's weight on raw height


def make_info(forward_vel: float, height: float, up_z: float, action) -> dict:
    """Build an `info` dict exactly as QuadrupedEnv.step would, so `fixed_reward`
    sees the real contract: the five named terms plus the raw signals."""
    import numpy as np
    vx = float(np.clip(forward_vel, -MAX_VX, MAX_VX))
    terms = {
        "forward": W_FORWARD * vx,
        "upright": W_UPRIGHT * up_z,
        "height": -W_HEIGHT * (height - TARGET_HEIGHT) ** 2,
        "alive": W_ALIVE,
        "ctrl": -W_CTRL * float(np.sum(np.asarray(action) ** 2)),
    }
    return {"reward_terms": terms, "height": height, "up_z": up_z, "forward_vel": forward_vel}


def fixed_reward(info, action, frac):
    """Repair the height-hack: keep whatever you like about staying tall/upright,
    but the reward MUST also increase when the robot moves forward faster.

    You have `info["reward_terms"]` (the five named terms: forward, upright,
    height, alive, ctrl) and the raw `info["forward_vel"]`, `info["height"]`,
    `info["up_z"]`. The broken original was simply:

        return HACK_HEIGHT_W * info["height"]

    which ignores forward motion entirely — that is why the policy games it. Add
    the missing forward drive (the env already computes it as
    info["reward_terms"]["forward"]). A reasonable fix keeps an upright/height
    incentive AND adds the forward term, e.g. the env's own shaped sum. The check
    only requires that your reward RISES with forward velocity and still isn't the
    height-only hack.

    Rough size: 1-3 lines.

    Before you write the fix, say it in one sentence: why did rewarding height
    *alone* make the policy rear up and cover zero distance, instead of walking —
    and why is adding a forward term a change to the *specification*, not to how
    hard you optimize?
    """
    # YOUR CODE HERE (delete the next line once you start)
    raise NotImplementedError("add a forward-progress term so the reward isn't the height-only hack")


if __name__ == "__main__":
    slow = make_info(forward_vel=0.0, height=0.30, up_z=1.0, action=[0.0] * 8)
    fast = make_info(forward_vel=0.9, height=0.25, up_z=1.0, action=[0.0] * 8)
    print(f"reward when standing tall, not moving: {fixed_reward(slow, [0.0] * 8, 1.0):+.3f}")
    print(f"reward when walking forward:           {fixed_reward(fast, [0.0] * 8, 1.0):+.3f}")
    print("(the walking-forward state should score higher — that's the fix)")
