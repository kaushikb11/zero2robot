"""SUGGESTED exercise candidate (humans promote) — bug-hunt, ch0.2.

This is the chapter scene with EXACTLY ONE authoring bug injected into the MJCF.
Its smoke metrics should match scene.py's (same seed, same physics) — they
don't. Symptom: the pusher drives forward but the block barely moves north
(tee ends at y~0.04 instead of ~0.38 [measured]), as if the pusher can only
travel along one axis.

Before you touch the XML, write one sentence: if the pusher drives forward but
the block barely moves north, what must be true about the two pusher joints'
axes — and which direction can the pusher no longer travel?

Find the bug by READING the MJCF the way the chapter taught: a joint's `axis`
is the direction it moves along, and two joints on a body should span the two
directions you want. Fix it, then re-run checks.py until the metrics agree with
the chapter's.

Run:  python ex2_bughunt_pusher_axis.py --smoke --seed 0
Estimated learner time: 20 minutes.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--out", type=Path, default=Path("outputs/ch0.2-scene-ex2"))
args = parser.parse_args()

settle_steps = 40
nudge_steps = 120 if args.smoke else args.steps
args.out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)

# The chapter scene, verbatim — except for ONE character in ONE joint axis.
SCENE_XML = """
<mujoco model="pusht_ch02">
  <option timestep="0.01" integrator="implicitfast"/>
  <worldbody>
    <geom name="table" type="plane" size="0.45 0.45 0.1" contype="0" conaffinity="0"/>
    <geom name="wall_n" type="box" pos="0  0.41 0.03" size="0.43 0.02 0.03"/>
    <geom name="wall_s" type="box" pos="0 -0.41 0.03" size="0.43 0.02 0.03"/>
    <geom name="wall_e" type="box" pos=" 0.41 0 0.03" size="0.02 0.43 0.03"/>
    <geom name="wall_w" type="box" pos="-0.41 0 0.03" size="0.02 0.43 0.03"/>
    <body name="tee" pos="0 0 0.0152">
      <joint name="tee_x"   type="slide" axis="1 0 0" damping="4"    frictionloss="1.2"/>
      <joint name="tee_y"   type="slide" axis="0 1 0" damping="4"    frictionloss="1.2"/>
      <joint name="tee_yaw" type="hinge" axis="0 0 1" damping="0.02" frictionloss="0.006"/>
      <geom name="tee_bar"  type="box" size="0.06 0.015 0.015" pos="0  0.00 0" mass="0.06"/>
      <geom name="tee_stem" type="box" size="0.015 0.045 0.015" pos="0 -0.06 0" mass="0.045"/>
    </body>
    <body name="pusher" pos="0 0 0.02">
      <joint name="pusher_x" type="slide" axis="1 0 0" damping="0.5"/>
      <joint name="pusher_y" type="slide" axis="1 0 0" damping="0.5"/>
      <geom name="pusher_tip" type="cylinder" size="0.015 0.02" mass="0.2"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="pusher_vx" joint="pusher_x" kv="20" ctrlrange="-1 1" forcerange="-30 30"/>
    <velocity name="pusher_vy" joint="pusher_y" kv="20" ctrlrange="-1 1" forcerange="-30 30"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(SCENE_XML)
data = mujoco.MjData(model)

tee_x_adr = int(model.joint("tee_x").qposadr[0])
tee_y_adr = int(model.joint("tee_y").qposadr[0])
tee_yaw_adr = int(model.joint("tee_yaw").qposadr[0])
data.qpos[tee_x_adr] = float(rng.uniform(-0.03, 0.03))
data.qpos[tee_y_adr] = float(rng.uniform(0.02, 0.06))
data.qpos[tee_yaw_adr] = float(rng.uniform(-0.2, 0.2))
data.qpos[int(model.joint("pusher_y").qposadr[0])] = -0.25
mujoco.mj_forward(model, data)

for _ in range(settle_steps):
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)
for _ in range(nudge_steps):
    data.ctrl[0], data.ctrl[1] = 0.0, 1.0
    mujoco.mj_step(model, data)

tee_final = [round(float(data.qpos[a]), 6) for a in (tee_x_adr, tee_y_adr, tee_yaw_adr)]
metrics = {
    "seed": args.seed,
    "tee_final_pose": tee_final,
    "pusher_final": [round(float(data.qpos[int(model.joint("pusher_x").qposadr[0])]), 6),
                     round(float(data.qpos[int(model.joint("pusher_y").qposadr[0])]), 6)],
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"tee final pose {tee_final} (block should end near y=0.38 when the scene is right)")
print(f"metrics: {args.out / 'metrics.json'}")
