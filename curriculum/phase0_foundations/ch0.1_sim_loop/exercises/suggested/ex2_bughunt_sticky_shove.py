"""SUGGESTED exercise candidate (humans promote) — bug-hunt, ch0.1.

This is the chapter's simulation loop, stripped to essentials (no rerun, no
inspection scaffolding) with EXACTLY ONE conceptual bug injected. Its smoke
metrics should match sim_loop.py's (same seed, same physics) — they don't.
Symptom: the box drifts ~0.42 m sideways instead of ~0.09 m, as if the shove
never let go.

Before you change a line, write one sentence: the box drifts 0.42 m instead of
0.09 — what does a shove that behaves as if it never let go tell you about what
mjData clears between steps and what it doesn't?

Find the bug by reasoning about mjData (rerun the chapter's mental model:
what persists across mj_step, and what doesn't?), fix it, and re-run
checks.py until the metrics agree with the chapter's.

Run:  python ex2_bughunt_sticky_shove.py --smoke --seed 0
Estimated learner time: 20 minutes.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--out", type=Path, default=Path("outputs/ch0.1-sim-loop-ex2"))
parser.add_argument("--no-perturb", dest="perturb", action="store_false")
args = parser.parse_args()

num_steps = 300 if args.smoke else args.steps
args.out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)

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

model = mujoco.MjModel.from_xml_string(SCENE_XML)
data = mujoco.MjData(model)
box = model.body("box")
pusher = model.body("pusher")

push_start = num_steps // 2
push_steps = 50  # 0.1 s of shove
push_newtons = rng.uniform(6.0, 12.0)
push_sign = rng.choice([-1.0, 1.0])
push_force = np.array([0.0, push_sign * push_newtons, 0.0])

for step in range(num_steps):
    in_shove = args.perturb and push_start <= step < push_start + push_steps
    if step == push_start:
        box_pos_at_push = data.xpos[box.id].copy()  # .copy(): xpos is a view; the next mj_step overwrites it in place

    data.ctrl[0] = 1.0
    if in_shove:
        data.xfrc_applied[box.id, :3] = push_force  # apply the shove during its window
    mujoco.mj_step(model, data)

final_box_pos = data.qpos[0:3]
final_box_speed = float(np.linalg.norm(data.qvel[0:3]))
lateral_drift = float(abs(final_box_pos[1] - box_pos_at_push[1]))

metrics = {
    "steps": num_steps,
    "seed": args.seed,
    "perturb": bool(args.perturb),
    "box_final_pos": [round(float(v), 6) for v in final_box_pos],
    "box_final_speed": round(final_box_speed, 6),
    "pusher_final_x": round(float(data.qpos[7]), 6),
    "lateral_drift": round(lateral_drift, 6),
    "shove_moved_box": bool(lateral_drift > 0.01),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

print(f"box final position {np.round(final_box_pos, 3)}, lateral drift {lateral_drift:.3f} m")
print(f"metrics: {args.out / 'metrics.json'}")
