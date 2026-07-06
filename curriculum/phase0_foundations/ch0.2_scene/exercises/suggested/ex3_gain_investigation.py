"""SUGGESTED exercise candidate (humans promote) — hyperparameter-investigation, ch0.2.

Objective tested: the actuator is part of the scene you author, and its gain is
a knob with physical consequences. The chapter's pusher runs VELOCITY servos
with `kv="20"`: ctrl is a target speed, kv is how hard the servo works to hit
that speed. Too low and the servo is mushy — it never reaches the commanded
speed against the block's friction; too high and it slams to speed instantly.

THE QUESTION (measurable): sweep the pusher's velocity gain kv over [5, 20, 80]
and, with the SAME commanded push (vy = 1.0 m/s for a fixed number of steps),
measure how far north the block travels. The walls are removed here so the full
travel is visible rather than clipped at the workspace edge.

PREDICT before running: the ORDERING of block travel across the three gains,
and roughly how big the spread is (1.5x? 3x? 10x?). Write it in PREDICTION,
then run this file and compare.
Estimated learner time: 15 minutes.
"""

import mujoco
import numpy as np

PREDICTION = None  # <- e.g. "kv=80 pushes ~2x farther than kv=5; kv=20 is in between"

METADATA = {
    "type": "hyperparameter-investigation",
    "chapter": "ch0.2-scene",
    "sweep": {"kv": [5, 20, 80]},
}

# The chapter block + pusher with the actuator gain spliced in as {kv}. No walls,
# so the block is free to travel as far as the push carries it.
SCENE = """
<mujoco model="ch0.2-ex3">
  <option timestep="0.01" integrator="implicitfast"/>
  <worldbody>
    <geom name="table" type="plane" size="0.9 0.9 0.1" contype="0" conaffinity="0"/>
    <body name="tee" pos="0 0 0.0152">
      <joint name="tee_x"   type="slide" axis="1 0 0" damping="4"    frictionloss="1.2"/>
      <joint name="tee_y"   type="slide" axis="0 1 0" damping="4"    frictionloss="1.2"/>
      <joint name="tee_yaw" type="hinge" axis="0 0 1" damping="0.02" frictionloss="0.006"/>
      <geom name="tee_bar"  type="box" size="0.06 0.015 0.015" pos="0  0.00 0" mass="0.06"/>
      <geom name="tee_stem" type="box" size="0.015 0.045 0.015" pos="0 -0.06 0" mass="0.045"/>
    </body>
    <body name="pusher" pos="0 0 0.02">
      <joint name="pusher_x" type="slide" axis="1 0 0" damping="0.5"/>
      <joint name="pusher_y" type="slide" axis="0 1 0" damping="0.5"/>
      <geom name="pusher_tip" type="cylinder" size="0.015 0.02" mass="0.2"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="pusher_vx" joint="pusher_x" kv="{kv}" ctrlrange="-1 1" forcerange="-30 30"/>
    <velocity name="pusher_vy" joint="pusher_y" kv="{kv}" ctrlrange="-1 1" forcerange="-30 30"/>
  </actuator>
</mujoco>
"""


def measure_travel(kv: int, seed: int = 0) -> float:
    """Settle the seeded block, then push north for 120 steps; return final y."""
    model = mujoco.MjModel.from_xml_string(SCENE.format(kv=kv))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)  # same draw order as scene.py -> same start pose
    tee_x_adr = int(model.joint("tee_x").qposadr[0])
    tee_y_adr = int(model.joint("tee_y").qposadr[0])
    tee_yaw_adr = int(model.joint("tee_yaw").qposadr[0])
    data.qpos[tee_x_adr] = float(rng.uniform(-0.03, 0.03))
    data.qpos[tee_y_adr] = float(rng.uniform(0.02, 0.06))
    data.qpos[tee_yaw_adr] = float(rng.uniform(-0.2, 0.2))
    data.qpos[int(model.joint("pusher_y").qposadr[0])] = -0.25
    mujoco.mj_forward(model, data)
    for _ in range(40):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
    for _ in range(120):
        data.ctrl[0], data.ctrl[1] = 0.0, 1.0
        mujoco.mj_step(model, data)
    return float(data.qpos[tee_y_adr])


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("Write your predicted ordering (and rough spread) into PREDICTION first.")
    print(f"prediction: {PREDICTION}")
    for kv in METADATA["sweep"]["kv"]:
        print(f"kv={kv:>2}: block travelled to y = {measure_travel(kv):.4f} m")
    print("Now: a policy trained with kv=20 — would its learned pushes still work if you shipped kv=5 hardware?")
