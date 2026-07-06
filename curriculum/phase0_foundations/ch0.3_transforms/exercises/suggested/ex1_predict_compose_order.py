"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch0.3.

Objective tested: rigid-transform composition does not commute. The chapter
composes world_from_tee with tee_from_pusher to place the pusher in the world.
Swap the order and you get a different answer — even here, where BOTH rotations
are yaws about the same z axis and therefore commute on their own.

THE TWO COMPOSITIONS (same two frames, opposite order):

    correct = world_from_tee.compose(tee_from_pusher)   # world <- tee <- pusher
    swapped = tee_from_pusher.compose(world_from_tee)    # the reversed order

PREDICT before you run: comparing `correct` and `swapped`, the pusher...
  A) lands in the same place — rigid transforms commute
  B) keeps the SAME final rotation, but lands somewhere else
  C) ends up with a different rotation AND a different place

Record your answer in PREDICTION below, then run this file.
Estimated learner time: 15 minutes.
"""

import numpy as np

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch0.3-transforms",
    "choices": ["A", "B", "C"],
    "gate_before_run": True,
}


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rotate_vector(q, v):
    pure = np.array([0.0, v[0], v[1], v[2]])
    conj = np.array([q[0], -q[1], -q[2], -q[3]])
    return quat_multiply(quat_multiply(q, pure), conj)[1:]


class Frame:
    def __init__(self, rotation, translation):
        self.rotation = np.asarray(rotation, dtype=float)
        self.translation = np.asarray(translation, dtype=float)

    def transform_point(self, point):
        return rotate_vector(self.rotation, point) + self.translation

    def compose(self, other):
        return Frame(quat_multiply(self.rotation, other.rotation),
                     self.transform_point(other.translation))


def yaw_quaternion(yaw):
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def compose_both_orders():
    """Build the chapter's two frames and compose them both ways.

    Returns (rotation_gap, translation_gap): how far apart the two orders land,
    measured on the resulting frame's rotation and translation separately.
    """
    world_from_tee = Frame(yaw_quaternion(0.9), np.array([0.12, -0.05, 0.0]))
    tee_from_pusher = Frame(yaw_quaternion(0.7), np.array([0.167228, 0.030575, 0.0]))
    correct = world_from_tee.compose(tee_from_pusher)
    swapped = tee_from_pusher.compose(world_from_tee)
    rotation_gap = float(np.max(np.abs(correct.rotation - swapped.rotation)))
    translation_gap = float(np.max(np.abs(correct.translation - swapped.translation)))
    return rotation_gap, translation_gap


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    rotation_gap, translation_gap = compose_both_orders()
    print(f"rotation gap between the two orders:    {rotation_gap:.2e}")
    print(f"translation gap between the two orders: {translation_gap:.4f} m")
    print(f"your prediction: {PREDICTION} — now explain WHY the translation moved but the rotation didn't.")
