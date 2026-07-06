"""SUGGESTED exercise candidate (humans promote) — code-completion, ch3.6.

Objective tested: the two lines that let a MuJoCo-trained policy plug into the
engine YOU built. The policy is a fixed function obs[10] -> action[2]; the ONLY
way it runs unchanged in a from-scratch engine is if that engine reports the
EXACT same observation contract MuJoCo did. Two pieces carry it:

  1. YAW FROM TWO MASSES. Your T-block is two point masses (bar center, stem
     center) on a ch3.4 rigid link — there is no yaw variable, only geometry. The
     body-frame line bar->stem is (0, -L), so the WORLD angle of (stem - bar) is
     yaw - pi/2. Recover yaw by inverting that (and wrap to [-pi, pi)).
  2. THE obs[10] ASSEMBLY. Pack pusher xy, block xy, sin/cos of the block yaw, and
     the fixed target (origin, yaw 0) into the pusht layout — the same 10 numbers,
     in the same order, that ch1.1 trained on.

The geometry (masses, link, contact) is given. You fill the two blanks. checks.py
compares your completed functions against the reference on random states; while a
blank is unfilled it raises NotImplementedError and the check SKIPS.
Estimated learner time: 20 minutes.
"""

import numpy as np

METADATA = {"type": "code-completion", "chapter": "ch3.6-compare"}

STEM_OFFSET = 0.06  # body-frame distance bar-center -> stem-center (from pusht.xml)


def wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi) — same helper as pusht_env."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def block_yaw(p_bar: np.ndarray, p_stem: np.ndarray) -> float:
    """The T's yaw, recovered from its two point masses. The body-frame line
    bar->stem is (0, -L); the world angle of (stem - bar) is therefore yaw - pi/2.
    """
    d = p_stem - p_bar  # noqa: F841  (used once you fill the blank below)
    # TODO (1): invert `world_angle(stem - bar) = yaw - pi/2` for yaw, wrapped to
    # [-pi, pi). Hint: wrap_angle(np.arctan2(d[1], d[0]) + np.pi / 2.0).
    yaw = None  # <- replace
    if yaw is None:
        raise NotImplementedError("recover yaw from the two masses (hint: arctan2(dy, dx) + pi/2, then wrap)")
    return yaw


def engine_obs(pusher_xy: np.ndarray, bar_xy: np.ndarray, stem_xy: np.ndarray) -> np.ndarray:
    """Build the pusht obs[10] so the ch1.1 policy plugs in unchanged. Layout
    (pusht_env.py): [pusher_x, pusher_y, tee_x, tee_y, sin(yaw), cos(yaw),
    target_x, target_y, sin(target_yaw), cos(target_yaw)]; target is fixed at
    (0, 0, yaw 0), so its four dims are the constants 0, 0, 0, 1.
    """
    px, py = pusher_xy
    tx, ty = bar_xy                    # bar center == PushT body origin (tee_xy)
    tyaw = block_yaw(bar_xy, stem_xy)  # noqa: F841  (used once you fill the blank below)
    # TODO (2): assemble the 10-vector in the layout above (float32). The block
    # yaw enters as sin(tyaw), cos(tyaw); the target block is the constants 0,0,0,1.
    obs = None  # <- replace
    if obs is None:
        raise NotImplementedError("assemble obs[10] in the pusht layout (hint: sin/cos of tyaw, then 0,0,0,1 for the fixed target)")
    return obs


if __name__ == "__main__":
    try:
        bar = np.array([0.1, 0.05])
        stem = bar + np.array([STEM_OFFSET * np.sin(0.3), -STEM_OFFSET * np.cos(0.3)])
        print("recovered yaw (want ~0.3):", round(block_yaw(bar, stem), 4))
        print("obs[10]:", np.round(engine_obs(np.array([-0.2, 0.1]), bar, stem), 3))
    except NotImplementedError as exc:
        print("not yet:", exc)
