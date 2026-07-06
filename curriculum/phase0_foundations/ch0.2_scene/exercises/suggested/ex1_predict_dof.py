"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch0.2.

Objective tested: a joint's TYPE decides a body's degrees of freedom, and the
degrees of freedom decide what the physics is even allowed to do. The chapter's
T-block hangs from a planar joint set (slide-x, slide-y, hinge-yaw) — three
DOF, all in the table plane, no way to leave it. Here you swap that for a single
`<freejoint/>`: six DOF, the block floats free in 3D.

THE DIFF UNDER STUDY (same push in both; the ONLY change is the block's joint):

    - <joint tee_x slide/> <joint tee_y slide/> <joint tee_yaw hinge/>   # 3 DOF, planar
    + <freejoint/>                                                        # 6 DOF, full 3D

PREDICT before you run: the pusher drives into the block from the corner. With
the freejoint instead of the planar joints, the block...
  A) behaves the same — a T on a table is a T on a table
  B) slides AND now rises off the table plane (its z stops being pinned)
  C) immediately falls through the floor

Record your answer in PREDICTION below, then run this file.
Estimated learner time: 10 minutes.
"""

import mujoco

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch0.2-scene",
    "choices": ["A", "B", "C"],
    "gate_before_run": True,
}

REST_Z = 0.0152  # the stem geom's height when the block lies flat on the table

# The chapter scene, trimmed to just the block + pusher, with the block's joint
# set spliced in as {joint}. Everything else is held fixed so the joint is the
# only variable.
SCENE = """
<mujoco model="ch0.2-ex1">
  <option timestep="0.01" integrator="implicitfast"/>
  <worldbody>
    <geom name="table" type="plane" size="0.45 0.45 0.1"/>
    <body name="tee" pos="0 0 0.0152">
      {joint}
      <geom name="tee_bar"  type="box" size="0.06 0.015 0.015" pos="0  0.00 0" mass="0.06"/>
      <geom name="tee_stem" type="box" size="0.015 0.045 0.015" pos="0 -0.06 0" mass="0.045"/>
    </body>
    <body name="pusher" pos="-0.2 -0.12 0.02">
      <joint name="pusher_x" type="slide" axis="1 0 0" damping="0.5"/>
      <joint name="pusher_y" type="slide" axis="0 1 0" damping="0.5"/>
      <geom name="pusher_tip" type="cylinder" size="0.015 0.02" mass="0.2"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="vx" joint="pusher_x" kv="20" ctrlrange="-1 1" forcerange="-30 30"/>
    <velocity name="vy" joint="pusher_y" kv="20" ctrlrange="-1 1" forcerange="-30 30"/>
  </actuator>
</mujoco>
"""

PLANAR_JOINT = (
    '<joint name="tee_x" type="slide" axis="1 0 0" damping="4" frictionloss="1.2"/>'
    '<joint name="tee_y" type="slide" axis="0 1 0" damping="4" frictionloss="1.2"/>'
    '<joint name="tee_yaw" type="hinge" axis="0 0 1" damping="0.02" frictionloss="0.006"/>'
)
FREE_JOINT = "<freejoint/>"


def run(joint_xml: str) -> dict:
    """Drive the pusher diagonally into the block; report DOF count and max height."""
    model = mujoco.MjModel.from_xml_string(SCENE.format(joint=joint_xml))
    data = mujoco.MjData(model)
    stem_gid = model.geom("tee_stem").id
    max_stem_z = 0.0
    for _ in range(200):
        data.ctrl[0], data.ctrl[1] = 1.0, 0.5  # push toward the block from the corner
        mujoco.mj_step(model, data)
        max_stem_z = max(max_stem_z, float(data.geom_xpos[stem_gid][2]))
    return {"nq": int(model.nq), "nv": int(model.nv), "max_stem_z": max_stem_z}


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    for label, joint in (("planar (3 DOF)", PLANAR_JOINT), ("freejoint (6 DOF)", FREE_JOINT)):
        r = run(joint)
        rose = "ROSE off the plane" if r["max_stem_z"] > REST_Z + 0.002 else "stayed pinned flat"
        print(f"{label}: nq={r['nq']} nv={r['nv']}, max stem z = {r['max_stem_z']:.4f} ({rose})")
    print(f"your prediction: {PREDICTION} — now explain the DOF count to yourself before checking the key.")
