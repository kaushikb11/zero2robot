"""SUGGESTED exercise candidate (humans promote) — hyperparameter-investigation, ch0.1.

Objective tested: reading a perturbation's effect quantitatively, and the idea
that scene parameters (here friction) are mjModel territory you can sweep.

THE QUESTION (measurable): the chapter's shove wins against sliding friction —
at the default coefficient 1.0, friction resists the 0.5 kg box with ~5 N.
Sweep the friction coefficient over [0.4, 1.0, 2.0] on both the box and the
floor. The SAME shove (seed 0: +9.8 N in y for 0.1 s) is applied each time.

PREDICT before running: the ordering of lateral drift across the three runs,
and roughly how big the spread is (2x? 10x? 100x?). Write it in PREDICTION,
then run this file and compare.
Estimated learner time: 15 minutes.
"""

import re
from pathlib import Path

import mujoco
import numpy as np

PREDICTION = None  # <- e.g. "mu=0.4 slides ~3x farther than 1.0; mu=2.0 barely moves"

METADATA = {
    "type": "hyperparameter-investigation",
    "chapter": "ch0.1-sim-loop",
    "sweep": {"friction": [0.4, 1.0, 2.0]},
}

ARTIFACT = Path(__file__).resolve().parents[2] / "sim_loop.py"


def scene_xml_with_friction(mu: float) -> str:
    """The chapter scene with an explicit friction coefficient on floor AND box.

    Both geoms, because MuJoCo combines the two surfaces' friction at each
    contact — patching only one side would let the other dominate.
    """
    xml = re.search(r'SCENE_XML = """(.*?)"""', ARTIFACT.read_text(), re.S).group(1)
    friction = f'friction="{mu} 0.005 0.0001"'
    xml = xml.replace('type="plane" size="2 2 0.1"', f'type="plane" size="2 2 0.1" {friction}')
    return xml.replace('type="box" size="0.05 0.05 0.05" mass="0.5"', f'type="box" size="0.05 0.05 0.05" mass="0.5" {friction}')


def measure_lateral_drift(mu: float, seed: int = 0) -> float:
    """Replay the chapter's smoke run (300 steps, shove at 150) at friction mu."""
    model = mujoco.MjModel.from_xml_string(scene_xml_with_friction(mu))
    data = mujoco.MjData(model)
    box = model.body("box")
    rng = np.random.default_rng(seed)  # same draw order as sim_loop.py -> same shove
    push_newtons = rng.uniform(6.0, 12.0)
    push_sign = rng.choice([-1.0, 1.0])
    push_force = np.array([0.0, push_sign * push_newtons, 0.0])
    y_at_push = 0.0
    for step in range(300):
        data.ctrl[0] = 1.0
        data.xfrc_applied[box.id, :3] = push_force if 150 <= step < 200 else 0.0
        if step == 150:
            y_at_push = float(data.qpos[1])
        mujoco.mj_step(model, data)
    return abs(float(data.qpos[1]) - y_at_push)


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("Write your predicted ordering (and rough spread) into PREDICTION first.")
    print(f"prediction: {PREDICTION}")
    for mu in METADATA["sweep"]["friction"]:
        print(f"friction mu={mu}: lateral drift = {measure_lateral_drift(mu):.4f} m")
    print("Now: does the SPREAD surprise you? Would a controller tuned at mu=1.0 survive at mu=0.4?")
