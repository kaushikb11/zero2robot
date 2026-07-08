"""zero2robot 0.3 — Spatial Reasoning Without Tears.

A robot's whole job is knowing where things are relative to other things: the
block in the gripper's frame, the gripper in the arm's frame, the arm in the
world. That bookkeeping is quaternions and rigid transforms, and it is where
three bugs bite everyone — quaternion component order, composition order, and
point-vs-frame direction. This file builds the toolkit FROM SCRATCH in numpy
(quaternion multiply, quat<->matrix, rotate-a-vector, and a rigid Frame with
compose/inverse/transform-point) and then, on every op, checks it agrees with
MuJoCo's own mju_* functions to machine precision. The lesson: you can trust
this code because MuJoCo, which drives every sim in this book, agrees with it.

Convention, fixed for the whole book: a quaternion is [w, x, y, z] (scalar
first), the MuJoCo order. Chapter 0.1 already crossed this seam once (MuJoCo
stores wxyz, rerun wants xyzw); Break It below is that seam turned into a bug.

Run it:      python transforms.py
CI smoke:    python transforms.py --smoke --seed 0 --no-rerun
Break It:    python transforms.py --break quat-convention --no-rerun
"""

# --- region: setup ---
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import rerun as rr

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as ch0.1). device.py
# keeps its torch import lazy, so this torch-free chapter stays torch-free.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curriculum.common.device import banner  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=0, help="seeds the random quaternions/points the checks run on")
parser.add_argument("--samples", type=int, default=512, help="how many random (quat, point) pairs to check against mju_*")
parser.add_argument("--smoke", action="store_true", help="fixed 512-sample run for CI; two runs must match byte-for-byte")
parser.add_argument("--out", type=Path, default=Path("outputs/ch0.3-transforms"))
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)  # recording is the default; opt OUT, not in
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip the .rrd recording (CI smoke)")
# The three bugs everyone writes, each selectable so you can watch it diverge:
parser.add_argument("--break", dest="bug", default="none",
                    choices=["none", "quat-convention", "compose-order", "point-vs-frame"],
                    help="inject a classic bug and measure the error it introduces")
args = parser.parse_args()

banner("ch0.3-transforms", device="cpu")  # pure-numpy/mujoco-CPU: honest cpu tier, never the host's mps/cuda. startup contract: tier + measured wall-clock to stdout, not metrics.json
num_samples = 512 if args.smoke else args.samples  # smoke count is FIXED so CI can diff runs exactly
args.out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)  # PCG64 — the only source of randomness in this file
# --- endregion ---

# --- region: quaternions ---
# A unit quaternion [w, x, y, z] encodes a 3D rotation in four numbers. Two
# operations generate everything else: multiplying two of them composes their
# rotations, and conjugating one inverts it. Multiplication is the Hamilton
# product — NOT componentwise, and NOT commutative (q1*q2 rotates differently
# from q2*q1, which is Break It #2). Write it out once, by hand, and never again.
def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # w: the scalar part
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # y
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,  # z
    ])


def quat_conjugate(q):
    # For a UNIT quaternion, the conjugate is the inverse rotation: flip the
    # vector part, keep the scalar. (Non-unit quaternions also need a /|q|^2,
    # but every rotation quaternion is unit, so we never pay for that here.)
    w, x, y, z = q
    return np.array([w, -x, -y, -z])
# --- endregion ---

# --- region: rotations ---
# Two ways to spend a quaternion. First, bake it into a 3x3 rotation matrix —
# handy when you'll rotate many vectors, and the form MuJoCo hands back from
# mju_quat2Mat. Every term is quadratic in the quaternion components; this is
# the exact expansion of q*(0,v)*conj(q) collected into a matrix.
def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


# Second, rotate a single vector directly: promote v to a pure quaternion
# (0, v), sandwich it as q * (0, v) * conj(q), and read off the vector part.
# This is what mju_rotVecQuat computes, and feeding it a WRONG-order quaternion
# is Break It #1 — the bug is invisible in the code and loud in the result.
def rotate_vector(q, v):
    pure = np.array([0.0, v[0], v[1], v[2]])  # a vector as a quaternion with zero scalar part
    rotated = quat_multiply(quat_multiply(q, pure), quat_conjugate(q))
    return rotated[1:]  # drop the (now ~0) scalar part; the vector part is the answer
# --- endregion ---

# --- region: frames ---
# A Frame is a rigid transform: a rotation (as a quaternion) plus a translation.
# Read it as "world_from_tee" — it takes a point written in the tee's local
# coordinates and returns the same point in world coordinates. That naming is
# not decoration: it's how you catch Break It #3, because frame_a.compose(
# frame_b) only type-checks in your head when a's "from" matches b's "to".
class Frame:
    def __init__(self, rotation, translation):
        self.rotation = np.asarray(rotation, dtype=float)        # unit quaternion [w, x, y, z]
        self.translation = np.asarray(translation, dtype=float)  # [x, y, z] in the parent frame

    def transform_point(self, point):
        # Rotate the point out of local coordinates, THEN shift by translation.
        # Order matters: translating first would rotate the offset too.
        return rotate_vector(self.rotation, point) + self.translation

    def compose(self, other):
        # self is world_from_a, other is a_from_b  ->  result is world_from_b.
        # Rotations multiply; the child's origin rides through the parent transform.
        return Frame(quat_multiply(self.rotation, other.rotation),
                     self.transform_point(other.translation))

    def inverse(self):
        # world_from_tee -> tee_from_world. The inverse rotation is the
        # conjugate; the inverse translation is where the world origin sits
        # once you've undone the rotation.
        inverse_rotation = quat_conjugate(self.rotation)
        return Frame(inverse_rotation, -rotate_vector(inverse_rotation, self.translation))


def yaw_quaternion(yaw):
    # A planar rotation about +z, the only rotation a PushT block ever makes.
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def random_unit_quaternion(generator):
    q = generator.standard_normal(4)  # a Gaussian in 4D, normalized, is uniform over rotations
    return q / np.linalg.norm(q)
# --- endregion ---

# --- region: demo ---
# The answer key: run every from-scratch op on `num_samples` random quaternions
# and points, and record the worst disagreement with MuJoCo's mju_* functions.
# Correct code lands at ~1e-15 (machine epsilon); that number IS the proof.
def max_error_against_mujoco(generator, count):
    err_multiply = err_matrix = err_rotate = err_roundtrip = 0.0
    reference = np.zeros(4)
    reference_matrix = np.zeros(9)
    reference_vector = np.zeros(3)
    for _ in range(count):
        q1 = random_unit_quaternion(generator)
        q2 = random_unit_quaternion(generator)
        v = generator.standard_normal(3)
        mujoco.mju_mulQuat(reference, q1, q2)
        err_multiply = max(err_multiply, np.max(np.abs(quat_multiply(q1, q2) - reference)))
        mujoco.mju_quat2Mat(reference_matrix, q1)
        err_matrix = max(err_matrix, np.max(np.abs(quat_to_matrix(q1).reshape(9) - reference_matrix)))
        mujoco.mju_rotVecQuat(reference_vector, v, q1)
        err_rotate = max(err_rotate, np.max(np.abs(rotate_vector(q1, v) - reference_vector)))
        # A Frame and its inverse must return any point exactly where it started.
        frame = Frame(q1, generator.standard_normal(3))
        roundtrip = frame.inverse().transform_point(frame.transform_point(v))
        err_roundtrip = max(err_roundtrip, np.max(np.abs(roundtrip - v)))
    return err_multiply, err_matrix, err_rotate, err_roundtrip


# The concrete question this chapter exists to answer: the PushT block sits at
# some pose; the pusher is somewhere in the world. Where is the pusher AS THE
# BLOCK SEES IT — in the block's own frame? That is one inverse and one
# transform_point, the exact move a reward function or a policy input makes.
tee_yaw = 0.9                                             # radians the block is turned
world_from_tee = Frame(yaw_quaternion(tee_yaw), np.array([0.12, -0.05, 0.0]))
pusher_world = np.array([0.20, 0.10, 0.0])               # pusher, written in world coordinates
pusher_in_tee = world_from_tee.inverse().transform_point(pusher_world)
pusher_recovered = world_from_tee.transform_point(pusher_in_tee)  # compose back: must return pusher_world

# Break It: the three bugs everyone writes, each measured against ground truth.
# #1 quat-convention: hand rotate_vector an [x,y,z,w] quaternion where it wants
#    [w,x,y,z] — the exact seam pusht_env documents. #2 compose-order: build
#    world_from_pusher in the wrong order. #3 point-vs-frame: read a world point
#    "in" a frame using the frame itself instead of its inverse. All three are
#    silent — no error, no exception — and all three are large.
tee_long_axis = np.array([0.06, 0.0, 0.0])               # the block's long bar, in its own frame
axis_in_world = np.zeros(3)                               # ground truth: MuJoCo rotates the bar into world
mujoco.mju_rotVecQuat(axis_in_world, tee_long_axis, world_from_tee.rotation)
if args.bug == "quat-convention":
    wrong_order = world_from_tee.rotation[[1, 2, 3, 0]]  # wxyz -> xyzw: the classic silent swap
    bug_error = float(np.max(np.abs(rotate_vector(wrong_order, tee_long_axis) - axis_in_world)))
elif args.bug == "compose-order":
    tee_from_pusher = Frame(yaw_quaternion(0.7), pusher_in_tee)
    correct = world_from_tee.compose(tee_from_pusher).translation           # world_from_tee . tee_from_pusher
    swapped = tee_from_pusher.compose(world_from_tee).translation           # the reversed, wrong order
    bug_error = float(np.max(np.abs(correct - swapped)))
elif args.bug == "point-vs-frame":
    # to express a WORLD point in the tee's frame you apply the INVERSE frame;
    # using world_from_tee forward (transform_point) is the silent direction swap
    wrong_in_tee = world_from_tee.transform_point(pusher_world)
    bug_error = float(np.max(np.abs(wrong_in_tee - pusher_in_tee)))
else:  # --break none: the correct rotation residual, ~machine epsilon
    bug_error = float(np.max(np.abs(rotate_vector(world_from_tee.rotation, tee_long_axis) - axis_in_world)))

err_multiply, err_matrix, err_rotate, err_roundtrip = max_error_against_mujoco(rng, num_samples)

if args.rerun:
    rr.init("zero2robot/ch0.3-transforms", spawn=False)
    rr.save(str(args.out / "transforms.rrd"))
    # A frame is legible as its three basis arrows (x red, y green, z blue) at
    # its origin. Log the world frame (identity) and the tee frame so you can
    # SEE the rotation compose. Under --break the tee axes point wrong.
    shown_rotation = world_from_tee.rotation[[1, 2, 3, 0]] if args.bug == "quat-convention" else world_from_tee.rotation
    for name, frame_rotation, origin in [
        ("world/frames/world", np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)),
        ("world/frames/tee", shown_rotation, world_from_tee.translation),
    ]:
        axes = quat_to_matrix(frame_rotation).T * 0.1  # rows = rotated x/y/z basis, scaled to scene
        rr.log(name, rr.Arrows3D(origins=np.tile(origin, (3, 1)), vectors=axes,
                                 colors=[[229, 76, 64], [76, 178, 76], [76, 102, 229]]))
    rr.log("world/objects/pusher", rr.Points3D([pusher_world], colors=[[64, 115, 217]], radii=0.012))
    # "Watch compositions": sweep an extra yaw onto the tee frame and trace where
    # its long-axis tip lands in the world — the tip draws an arc along the timeline.
    running = world_from_tee
    for step in range(24):
        rr.set_time("compose_step", sequence=step)
        tip_world = running.transform_point(tee_long_axis)
        rr.log("world/objects/tee_tip", rr.Points3D([tip_world], colors=[[217, 76, 64]], radii=0.01))
        running = running.compose(Frame(yaw_quaternion(np.pi / 12.0), np.zeros(3)))

metrics = {
    "seed": args.seed,
    "samples": num_samples,
    "break_mode": args.bug,
    # These four are diagnostic residuals near machine epsilon (~1e-15). They are
    # rounded like every other metric so the smoke JSON is byte-identical across
    # machines/BLAS (a raw ~1e-15 float is not portable). The precise values print
    # to stdout above, where a running learner reads them; rounded, an agreeing op
    # reads 0.0 and a real disagreement (>1e-6) would still survive the round.
    "quat_multiply_max_err": round(float(err_multiply), 6),
    "quat_to_matrix_max_err": round(float(err_matrix), 6),
    "rotate_vector_max_err": round(float(err_rotate), 6),
    "frame_roundtrip_max_err": round(float(err_roundtrip), 6),
    "break_max_err": round(bug_error, 6),  # 0 when correct, ~0.08 when a bug is injected
    "pusher_in_tee": [round(float(v), 6) for v in pusher_in_tee],
    "pusher_recovered": [round(float(v), 6) for v in pusher_recovered],
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

print(f"checked {num_samples} random quats/points against mju_* — worst disagreement:")
print(f"  quat_multiply {err_multiply:.2e}   quat_to_matrix {err_matrix:.2e}   rotate_vector {err_rotate:.2e}")
print(f"  frame round-trip {err_roundtrip:.2e}  (all ~machine epsilon => the from-scratch math is correct)")
print(f"pusher in the tee's frame: {np.round(pusher_in_tee, 4)}  (recovered world: {np.round(pusher_recovered, 4)})")
if args.bug == "none":
    print(f"break: none — the convention check agrees with MuJoCo to {bug_error:.2e}")
else:
    print(f"break: '{args.bug}' introduces {bug_error:.4f} of error — a wrong answer with no error message")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'transforms.rrd'} — open it with: rerun {args.out / 'transforms.rrd'}")
# --- endregion ---
