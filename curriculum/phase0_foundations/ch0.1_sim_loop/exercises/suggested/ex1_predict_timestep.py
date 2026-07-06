"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch0.1.

Objective tested: the timestep/stability trade. mj_step extrapolates further
per call as the timestep grows, and contact is where that goes wrong first.

THE DIFF UNDER STUDY (same 4.0 s of simulated time in both runs; the shove is
disabled so the simulator is the only suspect):

    - python sim_loop.py --no-perturb --timestep 0.002 --steps 2000
    + python sim_loop.py --no-perturb --timestep 0.05  --steps 80

PREDICT before you run: with the 25x larger timestep, the box...
  A) follows the same trajectory, sampled more coarsely
  B) ends up in roughly the same place, but contact looks springier
  C) leaves the floor during a flat horizontal push

Record your answer in PREDICTION below, then run this file.
Estimated learner time: 10 minutes.
"""

import re
from pathlib import Path

import mujoco

PREDICTION = None  # <- set to "A", "B", or "C" BEFORE running

# Site metadata (the site gates the run cell on a recorded choice; the answer
# key lives in checks.py, not here).
METADATA = {
    "type": "predict-then-run",
    "chapter": "ch0.1-sim-loop",
    "choices": ["A", "B", "C"],
    "gate_before_run": True,
}

ARTIFACT = Path(__file__).resolve().parents[2] / "sim_loop.py"
AIRBORNE_Z = 0.08  # the box rests at z=0.05; anything above this is a hop, not jitter


def scene_xml() -> str:
    """The exact scene from the chapter artifact — the diff is ONLY the timestep."""
    return re.search(r'SCENE_XML = """(.*?)"""', ARTIFACT.read_text(), re.S).group(1)


def run_flat_push(timestep: float, sim_seconds: float = 4.0) -> dict:
    """Push the box for a fixed amount of SIM time (not steps!) and watch its z."""
    model = mujoco.MjModel.from_xml_string(scene_xml())
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    max_box_z = 0.0
    for _ in range(int(sim_seconds / timestep)):
        data.ctrl[0] = 1.0  # same flat, horizontal, full-throttle push as the chapter
        mujoco.mj_step(model, data)
        max_box_z = max(max_box_z, float(data.qpos[2]))
    return {"timestep": timestep, "max_box_z": max_box_z, "final_pos": list(data.qpos[0:3])}


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Set PREDICTION to 'A', 'B', or 'C' first. Predicting after running teaches nothing.")
    for result in (run_flat_push(0.002), run_flat_push(0.05)):
        airborne = "AIRBORNE" if result["max_box_z"] > AIRBORNE_Z else "on the floor"
        print(f"dt={result['timestep']}: max box z = {result['max_box_z']:.4f} ({airborne}), final pos = {result['final_pos']}")
    print(f"your prediction: {PREDICTION} — now explain the measurement to yourself before checking the key.")
