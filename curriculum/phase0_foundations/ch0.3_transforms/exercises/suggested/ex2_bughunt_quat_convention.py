"""SUGGESTED exercise candidate (humans promote) — bug-hunt, ch0.3.

This routine rotates the PushT block's long axis from the block's frame into
the world, using the block's orientation quaternion. It should agree with
MuJoCo's mju_rotVecQuat to machine precision. It doesn't — it's off by ~0.10 m,
about the length of the bar itself, as if the block were pointed the wrong way.

EXACTLY ONE conceptual bug is injected, and it's the one pusht_env warns about:
a quaternion component-order mix-up. MuJoCo (and this book) order a quaternion
[w, x, y, z]. Somewhere below, a quaternion is handled as if it were ordered
[x, y, z, w]. Find it, fix it, and re-run checks.py until the error collapses
to ~1e-16.

Run:  python ex2_bughunt_quat_convention.py
Estimated learner time: 20 minutes.
"""

import mujoco
import numpy as np

METADATA = {
    "type": "bug-hunt",
    "chapter": "ch0.3-transforms",
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


def block_quaternion(yaw):
    """The block's orientation: a yaw about +z, returned as [x, y, z, w]."""
    # <-- the bug lives in this file's handling of THIS return value.
    return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)])


def tee_axis_in_world(yaw=0.9):
    """Rotate the block's long axis [0.06, 0, 0] into world coordinates."""
    quaternion = block_quaternion(yaw)
    return rotate_vector(quaternion, np.array([0.06, 0.0, 0.0]))


def break_error(yaw=0.9):
    """Max error between this file's result and MuJoCo's ground truth."""
    ours = tee_axis_in_world(yaw)
    reference = np.zeros(3)
    truth_quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])  # correct [w,x,y,z]
    mujoco.mju_rotVecQuat(reference, np.array([0.06, 0.0, 0.0]), truth_quat)
    return float(np.max(np.abs(ours - reference)))


if __name__ == "__main__":
    print(f"tee axis in world:  {np.round(tee_axis_in_world(), 4)}")
    print(f"error vs MuJoCo:    {break_error():.4f} m  (should be ~1e-16 once fixed)")
