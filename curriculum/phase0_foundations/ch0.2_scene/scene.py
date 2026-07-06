"""zero2robot 0.2 — Bodies, Joints, and MJCF.

Last chapter the scene was handed to you. This chapter you author it. MJCF is
MuJoCo's XML dialect for describing a world: a tree of BODIES, each carrying
GEOMS (the shapes physics collides) and JOINTS (the degrees of freedom that
connect a body to its parent). Get the tree right and the physics follows for
free; get it wrong and the T-block you meant to build falls apart in your hands.

We build the PushT scene the next fifteen chapters train on — a table, a
T-shaped block, a cylindrical pusher, a visual target — one region at a time,
then load it, print the kinematic tree it compiled to, and push the block to
prove it moves as one rigid piece. The Break It flag splits the T into two
bodies; because the reset only ever positions the bar, the split T settles
already broken — you watch the weld lesson fail on purpose, before the push.

Run it:      python scene.py
CI smoke:    python scene.py --smoke --seed 0 --no-rerun
Break it:    python scene.py --break split-tee --no-rerun
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
parser.add_argument("--seed", type=int, default=0, help="seeds the block's start pose, like a PushT reset")
parser.add_argument("--steps", type=int, default=120)  # nudge length; any laptop: instant
parser.add_argument("--smoke", action="store_true", help="fixed-length run for CI; two runs must match byte-for-byte")
parser.add_argument("--out", type=Path, default=Path("outputs/ch0.2-scene"))
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)  # recording is the default; opt OUT, not in
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip the .rrd recording (CI smoke)")
parser.add_argument("--break", dest="break_it", choices=["none", "split-tee"], default="none",
                    help="split-tee: author the T as TWO bodies instead of one welded body — watch it settle already split")
args = parser.parse_args()

banner("ch0.2-scene")  # startup contract: every artifact prints tier + measured wall-clock (to stdout, not metrics.json)
settle_steps = 40  # let the seeded start pose come to rest before we push
nudge_steps = 120 if args.smoke else args.steps  # smoke length is FIXED so CI can diff runs exactly
args.out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)  # PCG64 — the only source of randomness in this file
# --- endregion ---

# --- region: ground ---
# The table is a plane at z=0 with collision turned OFF (contype/conaffinity 0):
# the block never touches it. Planar friction is emulated with joint damping and
# frictionloss instead, which keeps the physics quasi-static and CPU-deterministic
# (chapter 3.x explains why real 3D contact is the hard case). Four walls box the
# block into the +-0.40 m workspace so a hard push can't launch it off the edge.
GROUND_MJCF = """
    <geom name="table" type="plane" size="0.45 0.45 0.1" rgba="0.90 0.90 0.92 1"
          contype="0" conaffinity="0"/>
    <geom name="wall_n" type="box" pos="0  0.41 0.03" size="0.43 0.02 0.03" rgba="0.6 0.6 0.6 1"/>
    <geom name="wall_s" type="box" pos="0 -0.41 0.03" size="0.43 0.02 0.03" rgba="0.6 0.6 0.6 1"/>
    <geom name="wall_e" type="box" pos=" 0.41 0 0.03" size="0.02 0.43 0.03" rgba="0.6 0.6 0.6 1"/>
    <geom name="wall_w" type="box" pos="-0.41 0 0.03" size="0.02 0.43 0.03" rgba="0.6 0.6 0.6 1"/>
"""
# --- endregion ---

# --- region: tee ---
# The T-block is the whole point of this region. It is ONE body carrying TWO box
# geoms — a horizontal bar and a vertical stem, offset so their union is a T.
# Because both geoms live in the same body, MuJoCo treats them as a single rigid
# object: they share one set of joints and can never move relative to each other.
# That shared rigidity IS the weld. The block connects to the world through a
# PLANAR joint set — slide-x, slide-y, hinge-yaw — so it has exactly the three
# degrees of freedom a flat pushing task needs (glide on the table, spin in
# place) and no way to tip up into 3D. qpos for this body is [x, y, yaw].
TEE_WELDED_MJCF = """
    <body name="tee" pos="0 0 0.0152">
      <joint name="tee_x"   type="slide" axis="1 0 0" damping="4"    frictionloss="1.2"/>
      <joint name="tee_y"   type="slide" axis="0 1 0" damping="4"    frictionloss="1.2"/>
      <joint name="tee_yaw" type="hinge" axis="0 0 1" damping="0.02" frictionloss="0.006"/>
      <geom name="tee_bar"  type="box" size="0.06 0.015 0.015" pos="0  0.00 0" rgba="0.45 0.5 0.95 1" mass="0.06"/>
      <geom name="tee_stem" type="box" size="0.015 0.045 0.015" pos="0 -0.06 0" rgba="0.45 0.5 0.95 1" mass="0.045"/>
    </body>
"""
# Break It: the SAME two geoms, but each in its own body with its own planar
# joints. Nothing welds them now — the stem is a separate rigid object. The reset
# below seeds ONLY the bar body's joints (looked up by name); the stem body is
# never positioned, so the T can't even be placed as one piece. It settles
# already split — gap ~0.09 m, not the welded 0.06 m [measured 2026-07-04] — and
# that settled gap is the tell. This is the single most common MJCF authoring
# mistake, and the sim below measures exactly how far the halves sit apart.
TEE_SPLIT_MJCF = """
    <body name="tee" pos="0 0 0.0152">
      <joint name="tee_x"   type="slide" axis="1 0 0" damping="4"    frictionloss="1.2"/>
      <joint name="tee_y"   type="slide" axis="0 1 0" damping="4"    frictionloss="1.2"/>
      <joint name="tee_yaw" type="hinge" axis="0 0 1" damping="0.02" frictionloss="0.006"/>
      <geom name="tee_bar"  type="box" size="0.06 0.015 0.015" pos="0 0 0" rgba="0.45 0.5 0.95 1" mass="0.06"/>
    </body>
    <body name="tee_stem_body" pos="0 -0.06 0.0152">
      <joint name="stem_x"   type="slide" axis="1 0 0" damping="4"    frictionloss="1.2"/>
      <joint name="stem_y"   type="slide" axis="0 1 0" damping="4"    frictionloss="1.2"/>
      <joint name="stem_yaw" type="hinge" axis="0 0 1" damping="0.02" frictionloss="0.006"/>
      <geom name="tee_stem" type="box" size="0.015 0.045 0.015" pos="0 0 0" rgba="0.95 0.5 0.45 1" mass="0.045"/>
    </body>
"""
TEE_MJCF = TEE_SPLIT_MJCF if args.break_it == "split-tee" else TEE_WELDED_MJCF
# --- endregion ---

# --- region: pusher ---
# The pusher is a cylinder on two slide joints — it glides in x and y but, unlike
# the block, cannot rotate: you gave it no hinge. The actuators are VELOCITY
# servos (kv is the gain): ctrl is a target speed in m/s, not a raw force, so a
# policy in chapter 1.1 can command "move at 1 m/s toward the block" and MuJoCo
# finds the force. forcerange caps how hard the servo may push. The block's DOFs
# are unactuated — it only ever moves because the pusher shoves it.
PUSHER_MJCF = """
    <body name="pusher" pos="0 0 0.02">
      <joint name="pusher_x" type="slide" axis="1 0 0" damping="0.5"/>
      <joint name="pusher_y" type="slide" axis="0 1 0" damping="0.5"/>
      <geom name="pusher_tip" type="cylinder" size="0.015 0.02" rgba="0.9 0.4 0.35 1" mass="0.2"/>
    </body>
"""
ACTUATOR_MJCF = """
    <velocity name="pusher_vx" joint="pusher_x" kv="20" ctrlrange="-1 1" forcerange="-30 30"/>
    <velocity name="pusher_vy" joint="pusher_y" kv="20" ctrlrange="-1 1" forcerange="-30 30"/>
"""
# --- endregion ---

# --- region: target ---
# The goal pose the task scores against, drawn as a translucent green T. It is a
# body with NO joint (welded to the world, so it never moves) and geoms with
# collision off — purely something to look at and, later, to measure distance to.
# A site marks its exact origin; sites are massless, collision-free reference
# frames you attach for exactly this kind of bookkeeping.
TARGET_MJCF = """
    <body name="target" pos="0 0 0.0005">
      <site name="target_site" pos="0 0 0" size="0.005" rgba="0 0 0 0"/>
      <geom name="target_bar"  type="box" size="0.06 0.015 0.0005" pos="0  0.00 0"
            rgba="0.35 0.8 0.4 0.5" contype="0" conaffinity="0"/>
      <geom name="target_stem" type="box" size="0.015 0.045 0.0005" pos="0 -0.06 0"
            rgba="0.35 0.8 0.4 0.5" contype="0" conaffinity="0"/>
    </body>
"""
# --- endregion ---

# --- region: build ---
# Assemble the regions into one MJCF document. worldbody is the root of the
# kinematic tree; everything above drops into it, and the actuators sit in their
# own <actuator> block. implicitfast is a stable, cheap integrator for this
# quasi-static task; timestep 0.01 s means 100 physics steps per simulated second.
SCENE_XML = f"""
<mujoco model="pusht_ch02">
  <option timestep="0.01" integrator="implicitfast"/>
  <worldbody>
    <camera name="top" pos="0 0 1.0" quat="1 0 0 0" fovy="50"/>
{GROUND_MJCF}{TARGET_MJCF}{TEE_MJCF}{PUSHER_MJCF}  </worldbody>
  <actuator>
{ACTUATOR_MJCF}  </actuator>
</mujoco>
"""

# from_xml_string compiles the text into an mjModel — the frozen world. If the
# XML is malformed MuJoCo raises HERE, before a single step runs, which is the
# best time to hear about a scene bug.
model = mujoco.MjModel.from_xml_string(SCENE_XML)
data = mujoco.MjData(model)

# Validate: read back the tree MuJoCo actually compiled, not the tree you think
# you wrote. Every body's parent, every joint's type, and the state-vector sizes
# nq (positions) / nv (velocities) are the ground truth for the whole scene.
JOINT_TYPE_NAME = {int(mujoco.mjtJoint.mjJNT_FREE): "free", int(mujoco.mjtJoint.mjJNT_BALL): "ball",
                   int(mujoco.mjtJoint.mjJNT_SLIDE): "slide", int(mujoco.mjtJoint.mjJNT_HINGE): "hinge"}
joint_types = [JOINT_TYPE_NAME[int(model.jnt(i).type[0])] for i in range(model.njnt)]
print(f"kinematic tree: {model.nbody - 1} bodies under worldbody, {model.njnt} joints, nq={model.nq} nv={model.nv}")
for i in range(1, model.nbody):  # body 0 is the world; skip it
    body = model.body(i)
    print(f"  body '{body.name}' <- parent '{model.body(int(body.parentid[0])).name}'")
print(f"joint types (in qpos order): {joint_types}")

# Set the block's start pose from the seed — the same idea as a PushT reset. Look
# the joints up by NAME so this still works when --break split-tee changes the
# qpos layout out from under us (the welded T has 3 dofs; the split T has 6).
tee_x_adr = int(model.joint("tee_x").qposadr[0])
tee_y_adr = int(model.joint("tee_y").qposadr[0])
tee_yaw_adr = int(model.joint("tee_yaw").qposadr[0])
data.qpos[tee_x_adr] = float(rng.uniform(-0.03, 0.03))
data.qpos[tee_y_adr] = float(rng.uniform(0.02, 0.06))
data.qpos[tee_yaw_adr] = float(rng.uniform(-0.2, 0.2))
data.qpos[int(model.joint("pusher_y").qposadr[0])] = -0.25  # start the pusher south of the block
mujoco.mj_forward(model, data)  # derive body/geom world poses from qpos without stepping time

# The weld invariant: the world-space distance between the bar and stem geoms.
# In a correctly welded T it is a constant 0.06 m forever, no matter how the
# block moves. If it EVER leaves 0.06, the two geoms are NOT rigidly attached —
# so the robust rigidity verdict is the max deviation from 0.06 over the whole
# run, not the fragile final gap (a split T's halves can drift and then happen to
# drift back near 0.06 under the push; the SETTLED and PEAK gaps do not lie).
bar_gid, stem_gid = model.geom("tee_bar").id, model.geom("tee_stem").id
def tee_gap() -> float:
    return float(np.linalg.norm(data.geom_xpos[bar_gid] - data.geom_xpos[stem_gid]))

if args.rerun:
    rr.init("zero2robot/ch0.2-scene", spawn=False)
    rr.save(str(args.out / "scene.rrd"))
    # Shapes never change; log them once as static, then stream only transforms.
    rr.log("world/objects/target", rr.Boxes3D(half_sizes=[[0.06, 0.015, 0.001], [0.015, 0.045, 0.001]],
           centers=[[0, 0, 0], [0, -0.06, 0]], colors=[[90, 204, 102]]), static=True)
    rr.log("world/objects/tee", rr.Boxes3D(half_sizes=[[0.06, 0.015, 0.015], [0.015, 0.045, 0.015]],
           centers=[[0, 0, 0], [0, -0.06, 0]], colors=[[115, 128, 242]]), static=True)
    rr.log("world/robot/pusher", rr.Boxes3D(half_sizes=[[0.015, 0.015, 0.02]], colors=[[230, 102, 89]]), static=True)

def log_step() -> None:
    if not args.rerun:
        return
    rr.set_time("sim_time", duration=data.time)
    rr.log("world/objects/tee", rr.Transform3D(translation=data.xpos[model.body("tee").id],
           quaternion=data.xquat[model.body("tee").id][[1, 2, 3, 0]]))  # MuJoCo wxyz -> rerun xyzw
    rr.log("world/robot/pusher", rr.Transform3D(translation=data.xpos[model.body("pusher").id]))

# Settle: no control, let friction bring the seeded start pose to rest. Track the
# largest deviation of the bar-stem gap from the welded 0.06 m across every step
# of the whole run — this max is 0.0 for a truly welded T and large the instant
# the halves are two bodies (the split T is already off 0.06 before the push).
gap_max_dev = abs(tee_gap() - 0.06)
for _ in range(settle_steps):
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)
    gap_max_dev = max(gap_max_dev, abs(tee_gap() - 0.06))
    log_step()
tee_settled = [data.qpos[tee_x_adr], data.qpos[tee_y_adr], data.qpos[tee_yaw_adr]]
gap_settled = tee_gap()

# Nudge: drive the pusher north at full commanded speed. It contacts the block
# and drives it up the table — the whole T should travel as one piece.
for _ in range(nudge_steps):
    data.ctrl[0], data.ctrl[1] = 0.0, 1.0  # [vx, vy] m/s targets for the two velocity servos
    mujoco.mj_step(model, data)
    gap_max_dev = max(gap_max_dev, abs(tee_gap() - 0.06))
    log_step()
tee_final = [data.qpos[tee_x_adr], data.qpos[tee_y_adr], data.qpos[tee_yaw_adr]]

metrics = {
    "break_it": args.break_it,
    "joint_types": joint_types,
    "n_bodies": int(model.nbody - 1),  # excluding the world body
    "n_geoms": int(model.ngeom),
    "n_joints": int(model.njnt),
    "nq": int(model.nq),
    "nu": int(model.nu),
    "nv": int(model.nv),
    "pusher_final": [round(float(data.qpos[int(model.joint("pusher_x").qposadr[0])]), 6),
                     round(float(data.qpos[int(model.joint("pusher_y").qposadr[0])]), 6)],
    "seed": args.seed,
    "tee_final_pose": [round(float(v), 6) for v in tee_final],
    "tee_gap_max_dev": round(gap_max_dev, 6),  # robust verdict: 0.0 iff welded, >=0.03 for a split T (whole-run peak)
    "tee_gap_settled": round(gap_settled, 6),  # already != 0.06 for a split T: the reset can't even place it as one piece
    "tee_settled_pose": [round(float(v), 6) for v in tee_settled],
    "tee_stayed_rigid": bool(gap_max_dev < 1e-4),  # keyed off the robust whole-run peak, not the fragile final gap
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

print(f"nudge: pusher drove block from y={tee_settled[1]:.3f} to y={tee_final[1]:.3f}")
print(f"weld check: bar-stem gap settled at {gap_settled:.4f} m, peak deviation {gap_max_dev:.4f} m from the 0.06 weld "
      f"({'RIGID' if metrics['tee_stayed_rigid'] else 'SPLIT — never one rigid body'})")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'scene.rrd'} — open it with: rerun {args.out / 'scene.rrd'}")
# --- endregion ---
