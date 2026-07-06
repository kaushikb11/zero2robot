"""zero2robot 0.1 — The Simulation Loop.

Every MuJoCo program is a conversation between two objects: mjModel, the
description of the world that never changes while it runs, and mjData, the
complete state of the world right now. This file loads a model, steps its
data forward in time, shoves a box mid-run with a force the simulation
didn't plan for, and records the whole thing for playback.

Run it:      python sim_loop.py
CI smoke:    python sim_loop.py --smoke --seed 0 --no-rerun
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
# sys.path so `curriculum.common` resolves (same pattern as ch1.1). device.py
# keeps its torch import lazy, so this torch-free chapter stays torch-free.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curriculum.common.device import banner  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=0, help="seeds the shove direction and strength")
parser.add_argument("--steps", type=int, default=2000)  # any laptop: instant
parser.add_argument("--timestep", type=float, default=0.002)  # sim seconds per mj_step; raise it and see Break It
parser.add_argument("--smoke", action="store_true", help="fixed 300-step run for CI; two runs must match byte-for-byte")
parser.add_argument("--out", type=Path, default=Path("outputs/ch0.1-sim-loop"))
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)  # recording is the default; opt OUT, not in
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip the .rrd recording (CI smoke)")
parser.add_argument("--no-perturb", dest="perturb", action="store_false", help="skip the shove — a baseline the tests compare against")
args = parser.parse_args()

banner("ch0.1-sim-loop")  # startup contract: every artifact prints tier + measured wall-clock (to stdout, not metrics.json)
num_steps = 300 if args.smoke else args.steps  # smoke length is FIXED so CI can diff runs exactly
args.out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)  # PCG64 — the only source of randomness in this file
# --- endregion ---

# --- region: scene ---
# The scene is handed to you this chapter; in 0.2 you write your own. Read it
# top to bottom anyway: a floor plane, a red box attached to the world by a
# free joint (all six degrees of freedom — loose on the table), and a blue
# pusher that can ONLY slide along x, driven by a single motor.
SCENE_XML = """
<mujoco model="pusher_scene">
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.85 0.85 0.85 1"/>
    <body name="box" pos="0.4 0 0.05">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05" mass="0.5" rgba="0.85 0.3 0.25 1"/>
    </body>
    <body name="pusher" pos="0 0 0.05">
      <joint name="pusher_slide" type="slide" axis="1 0 0" damping="4"/>
      <geom name="pusher_geom" type="box" size="0.04 0.12 0.05" mass="1.0" rgba="0.25 0.45 0.85 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="push_motor" joint="pusher_slide" gear="10" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

# mjModel: the compiled world — geometry, masses, joints, actuator gearing.
# Nothing in it changes while the simulation steps. Build it once.
model = mujoco.MjModel.from_xml_string(SCENE_XML)
model.opt.timestep = args.timestep  # "never changes" means during stepping; between runs it's yours to edit

# mjData: the state — positions, velocities, contact forces, time. Everything
# mj_step writes lives here. One model can drive many independent datas
# (that idea becomes 4096 parallel robots in chapter 2.3).
data = mujoco.MjData(model)

box = model.body("box")  # named lookups beat remembering integer ids
pusher = model.body("pusher")
# --- endregion ---

# --- region: perturb ---
# Mid-run we shove the box sideways with a force the pusher knows nothing
# about. xfrc_applied is mjData's slot for exactly this: an external
# (force, torque) on any body, added on top of whatever physics is already
# doing. It PERSISTS until you clear it — MuJoCo never resets it for you,
# and forgetting that is a classic bug (exercise 2 makes you find it).
push_start = num_steps // 2
push_steps = 50  # 0.1 s of shove at the default timestep
push_newtons = rng.uniform(6.0, 12.0)  # friction on the 0.5 kg box resists with ~5 N; this wins
push_sign = rng.choice([-1.0, 1.0])  # +y or -y: sideways, across the pusher's line of travel
push_force = np.array([0.0, push_sign * push_newtons, 0.0])
# --- endregion ---

# --- region: loop ---
if args.rerun:
    rr.init("zero2robot/ch0.1-sim-loop", spawn=False)
    rr.save(str(args.out / "sim_loop.rrd"))
    # Shapes and colors never change, so log them once as static; the loop
    # then logs only the moving transforms.
    rr.log("world/objects/box", rr.Boxes3D(half_sizes=[[0.05, 0.05, 0.05]], colors=[[217, 76, 64]]), static=True)
    rr.log("world/robot/pusher", rr.Boxes3D(half_sizes=[[0.04, 0.12, 0.05]], colors=[[64, 115, 217]]), static=True)

mujoco.mj_forward(model, data)  # compute body poses from qpos WITHOUT advancing time, so step 0 is loggable
box_pos_at_push = data.xpos[box.id].copy()

for step in range(num_steps):
    data.ctrl[0] = 1.0  # full throttle on the slide motor — in chapter 1.1, a policy writes this line

    in_shove = args.perturb and push_start <= step < push_start + push_steps
    data.xfrc_applied[box.id, :3] = push_force if in_shove else 0.0  # write 0 explicitly, or the shove sticks forever
    if step == push_start:
        box_pos_at_push = data.xpos[box.id].copy()  # .copy(): xpos is a view into mjData and mj_step overwrites it

    mujoco.mj_step(model, data)  # one timestep: collision, contact, actuation, integration — the whole pipeline

    if args.rerun and step % 5 == 0:  # logging every step quintuples file size and teaches nothing extra
        rr.set_time("sim_time", duration=data.time)
        # MuJoCo stores quaternions wxyz; rerun wants xyzw — hence the reindex.
        rr.log("world/objects/box", rr.Transform3D(translation=data.xpos[box.id], quaternion=data.xquat[box.id][[1, 2, 3, 0]]))
        rr.log("world/robot/pusher", rr.Transform3D(translation=data.xpos[pusher.id]))
        shove_arrow = push_force * (0.02 if in_shove else 0.0)  # scaled to scene size; zero-length when inactive
        rr.log("world/objects/box/shove", rr.Arrows3D(origins=[data.xpos[box.id]], vectors=[shove_arrow]))
# --- endregion ---

# --- region: inspect ---
# The box is declared first, so its free joint owns qpos[0:7] — xyz position
# plus a wxyz quaternion. Its qvel slice is qvel[0:6]: 7 position numbers but
# 6 velocity numbers, because a quaternion has 4 components and only 3
# degrees of freedom. Chapter 0.3 lives inside that gap.
final_box_pos = data.qpos[0:3]
final_box_speed = float(np.linalg.norm(data.qvel[0:3]))
lateral_drift = float(abs(final_box_pos[1] - box_pos_at_push[1]))  # how far the shove pushed it off the x axis

metrics = {
    "steps": num_steps,
    "seed": args.seed,
    "perturb": bool(args.perturb),
    "box_final_pos": [round(float(v), 6) for v in final_box_pos],
    "box_final_speed": round(final_box_speed, 6),
    "pusher_final_x": round(float(data.qpos[7]), 6),  # the slide joint's single dof comes after the box's 7
    "lateral_drift": round(lateral_drift, 6),
    "shove_moved_box": bool(lateral_drift > 0.01),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

print(f"stepped {num_steps} steps of {model.opt.timestep} s -> {data.time:.2f} s of sim time")
print(f"box final position {np.round(final_box_pos, 3)}, speed {final_box_speed:.3f} m/s")
if args.perturb:
    print(f"shove: {push_force[1]:+.1f} N in y for {push_steps} steps -> {lateral_drift:.3f} m of sideways drift")
else:
    print("no shove (baseline run)")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'sim_loop.rrd'} — open it with: rerun {args.out / 'sim_loop.rrd'}")
# --- endregion ---
