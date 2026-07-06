"""SUGGESTED exercise candidate (humans promote) — code-completion, ch0.3.

You get quat_multiply and quat_conjugate for free. Complete rotate_vector using
the quaternion "sandwich": promote the vector v to a pure quaternion (0, v),
compute q * (0, v) * conj(q), and return the vector part.

The whole rotation lives in that one product. Fill in the body where marked,
then run checks.py — it rotates 256 random vectors by 256 random quaternions
and compares your result to MuJoCo's mju_rotVecQuat.

Run:  python ex3_complete_rotate_vector.py
Estimated learner time: 15 minutes.
"""

import numpy as np

METADATA = {
    "type": "code-completion",
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


def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def rotate_vector(q, v):
    """Rotate 3-vector v by unit quaternion q ([w, x, y, z]). Return a 3-vector."""
    # TODO(you): build the pure quaternion (0, v), sandwich it as
    # q * (0, v) * conj(q), and return the vector (last three) components.
    raise NotImplementedError("complete rotate_vector using quat_multiply and quat_conjugate")


if __name__ == "__main__":
    q = np.array([np.cos(0.45), 0.0, 0.0, np.sin(0.45)])  # a 0.9 rad yaw
    print("rotating [0.06, 0, 0] by a 0.9 rad yaw ->", np.round(rotate_vector(q, np.array([0.06, 0.0, 0.0])), 4))
