"""SUGGESTED exercise candidate (humans promote) — code-completion, ch5.8.

Objective tested: the ONE joint that must depend on where the box is. The scripted expert that
drives the SO-101 to the box is mostly a FIXED "lower-to-the-table" pose — shoulder_lift,
elbow_flex, wrist_flex held at constants — steered by exactly one joint: shoulder_pan, which
must aim the arm at the box's azimuth. Get that one number right and the recorded demos teach a
reach; get it wrong (or constant) and every demo drives to the same spot and the clone can't
reach a box that moves.

Your job: implement `expert_action(box_xyz)` — return the 6-D joint-position target
[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper] that reaches a box at
`box_xyz`. Five of the six are the fixed reach constants below. The sixth, shoulder_pan, is the
azimuth of the box in the base frame, mapped through PAN_GAIN (a one-number fit of pan -> tip
azimuth measured from the model): pan = -atan2(box_y, box_x) / PAN_GAIN.

Implement it below (pure numpy, no MuJoCo needed), then run the checks:
    pytest curriculum/phase5_practitioner/ch5.8_real_loop/exercises/suggested/checks.py -k ex3
Estimated learner time: 10 minutes.
"""

import numpy as np

# The fixed part of the reach: lower the gripper to the table, wrist_roll neutral, gripper ajar.
REACH_POSE = np.array([0.2, 0.3, 0.7])  # shoulder_lift, elbow_flex, wrist_flex
PAN_GAIN = 0.87                          # pan -> tip-azimuth calibration (fit from the SO-101 model)


def expert_action(box_xyz: np.ndarray) -> np.ndarray:
    """(3,) box position -> (6,) joint-position target that reaches it.

    Only shoulder_pan depends on the box. Remove the NotImplementedError and write it.
    HINT: pan = -arctan2(box_y, box_x) / PAN_GAIN; the target is
          [pan, REACH_POSE[0], REACH_POSE[1], REACH_POSE[2], 0.0, 0.3] as float32.
    """
    raise NotImplementedError("write expert_action: aim shoulder_pan at the box's azimuth, hold the reach pose")
