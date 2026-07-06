// PushT observation + task constants — the SINGLE SOURCE OF TRUTH for the
// browser's obs[10]/action[2] contract. Mirrors curriculum/common/envs/pusht/
// pusht_env.py (OBS/ACT layout, physics cadence, success tolerances) and
// gen_demos.py (STATE_NAMES / feature names). Phase-3 policy inference MUST feed
// a trained policy the obs this module builds, or the browser obs will not equal
// the training obs and the policy-return loop breaks silently.
//
// Obs-parity is asserted by scripts/obs_parity_check.mjs: assembleObs() output
// for a known state == PushTEnv._obs() for the same qpos (cross-checked against
// .venv python, within f32 tolerance).

import type { Sim } from '../sim/mujoco_sim';

// ------------------------------------------------------------------ dimensions
export const OBS_DIM = 10; // PushTEnv.OBS_DIM
export const ACT_DIM = 2; // PushTEnv.ACT_DIM
export const IMG_HW = 96; // gen_demos.IMG_HW (observation.image is 96x96x3)

// ------------------------------------------------------------- physics cadence
export const CONTROL_HZ = 10; // PushTEnv.CONTROL_HZ (== interchange fps)
export const FRAME_SKIP = 10; // PushTEnv.FRAME_SKIP (100 Hz physics / 10 Hz ctrl)
export const CONTROL_DT = 1 / CONTROL_HZ; // 0.1 s of sim per control step
export const MAX_STEPS = 300; // PushTEnv.MAX_STEPS (30 s at 10 Hz)

// ------------------------------------------------------------ success criteria
export const POS_TOL = 0.03; // m   — PushTEnv.POS_TOL
export const ANG_TOL = 0.2; // rad — PushTEnv.ANG_TOL (~11.5 deg)
export const SUCCESS_HOLD = 5; // consecutive in-tolerance control steps

// The target pose is fixed at the origin (PushTEnv.TARGET_POSE = [0, 0, 0]).
export const TARGET_X = 0.0;
export const TARGET_Y = 0.0;
export const TARGET_YAW = 0.0;

// -------------------------------------------------------- dataset/format labels
export const TASK = 'Push the T-shaped block to the target pose.';
export const ROBOT_TYPE = 'pusher_2d';
export const REPO_ID = 'zero2robot/pusht_teleop';

// Observation field names, in order — byte-identical to gen_demos.STATE_NAMES.
export const STATE_NAMES = [
  'pusher_x',
  'pusher_y',
  'tee_x',
  'tee_y',
  'sin_tee_yaw',
  'cos_tee_yaw',
  'target_x',
  'target_y',
  'sin_target_yaw',
  'cos_target_yaw',
] as const;

export const ACTION_NAMES = ['pusher_vx', 'pusher_vy'] as const;

// MuJoCo joints whose qpos we read (tee) or read/write (all) per reset/step.
export const TEE_JOINTS = ['tee_x', 'tee_y', 'tee_yaw'] as const;
export const PUSHER_JOINTS = ['pusher_x', 'pusher_y'] as const;

/** Wrap an angle to [-pi, pi). Mirrors pusht_env.wrap_angle. */
export function wrapAngle(a: number): number {
  return ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
}

/**
 * Assemble the float32[10] observation from raw scalar state. Pure — no sim, no
 * I/O — so it can be parity-checked against pusht_env._obs() in isolation.
 * Field order and the sin/cos yaw encoding are byte-identical to pusht_env._obs.
 * (sin/cos are 2pi-periodic, so wrapping tyaw is unnecessary here.)
 */
export function assembleObs(
  pusherX: number,
  pusherY: number,
  teeX: number,
  teeY: number,
  teeYaw: number,
  targetX: number,
  targetY: number,
  targetYaw: number,
): Float32Array {
  const o = new Float32Array(OBS_DIM);
  o[0] = pusherX;
  o[1] = pusherY;
  o[2] = teeX;
  o[3] = teeY;
  o[4] = Math.sin(teeYaw);
  o[5] = Math.cos(teeYaw);
  o[6] = targetX;
  o[7] = targetY;
  o[8] = Math.sin(targetYaw);
  o[9] = Math.cos(targetYaw);
  return o; // Float32Array assignment rounds to f32, matching np.float32
}

/**
 * Build the observation from live sim state. The obs a trained policy consumes
 * in Phase 3 comes from HERE — identical to what the teleop recorder writes.
 */
export function buildObs(sim: Sim): Float32Array {
  const px = sim.jointQpos('pusher_x');
  const py = sim.jointQpos('pusher_y');
  const tx = sim.jointQpos('tee_x');
  const ty = sim.jointQpos('tee_y');
  const tyaw = wrapAngle(sim.jointQpos('tee_yaw'));
  return assembleObs(px, py, tx, ty, tyaw, TARGET_X, TARGET_Y, TARGET_YAW);
}

/** Position/angle error of the tee vs the fixed target (pusht_env._errors). */
export function teeErrors(sim: Sim): { posErr: number; angErr: number } {
  const tx = sim.jointQpos('tee_x');
  const ty = sim.jointQpos('tee_y');
  const tyaw = wrapAngle(sim.jointQpos('tee_yaw'));
  const posErr = Math.hypot(tx - TARGET_X, ty - TARGET_Y);
  const angErr = Math.abs(wrapAngle(tyaw - TARGET_YAW));
  return { posErr, angErr };
}
