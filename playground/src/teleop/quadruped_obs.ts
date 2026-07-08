// Quadruped observation + task constants — the SINGLE SOURCE OF TRUTH for the
// browser's obs[23]/action[8] contract. Mirrors curriculum/common/envs/quadruped/
// quadruped_env.py (OBS/ACT layout, physics cadence, residual-pose action) so a
// ch2.5 walk (or ch2.4 rewards / ch2.7 DR) policy trained on quadruped_env is fed
// byte-identical observations in the browser. If the browser obs does not equal
// the training obs, the policy-drive loop breaks silently.
//
// Obs-parity is asserted by scripts/quadruped_obs_parity_check.mjs, which — unlike
// the pure-function pusher/cartpole checks — must run the MuJoCo-WASM sim (the
// nontrivial reads are the free-joint torso height/linvel and the torso UP-VECTOR
// from the body world rotation matrix xmat), so it loads the same WASM binding the
// browser uses and compares buildObs(sim) against quadruped_env._obs() for the
// SAME qpos/qvel, within f32 tolerance.

import type { Sim } from '../sim/mujoco_sim';

// ------------------------------------------------------------------ dimensions
export const OBS_DIM = 23; // QuadrupedEnv.OBS_DIM
export const ACT_DIM = 8; // QuadrupedEnv.ACT_DIM

// ------------------------------------------------------------- physics cadence
export const CONTROL_HZ = 50; // QuadrupedEnv.CONTROL_HZ
export const FRAME_SKIP = 4; // QuadrupedEnv.FRAME_SKIP (200 Hz physics / 50 Hz ctrl, timestep 0.005)
export const CONTROL_DT = 1 / CONTROL_HZ; // 0.02 s of sim per control step
export const MAX_STEPS = 500; // QuadrupedEnv.MAX_STEPS (10 s at 50 Hz)

// ----------------------------------------------------------------- task shape
// Joint order used everywhere (obs, action, DEFAULT_POSE) — MUST match
// quadruped_env.JOINT_NAMES exactly.
export const JOINT_NAMES = [
  'FL_hip', 'FL_knee', 'FR_hip', 'FR_knee',
  'HL_hip', 'HL_knee', 'HR_hip', 'HR_knee',
] as const;
export const FREE_JOINT = 'root'; // the 6-DOF floating base
export const TORSO_BODY = 'torso';

export const DEFAULT_HIP = 0.6; // QuadrupedEnv nominal crouch
export const DEFAULT_KNEE = -1.2;
export const DEFAULT_POSE = JOINT_NAMES.map((n) => (n.includes('hip') ? DEFAULT_HIP : DEFAULT_KNEE));
export const STAND_HEIGHT = 0.257; // torso-center z at which the crouch's feet touch z=0
export const ACTION_SCALE = 0.5; // action in [-1,1] -> +-0.5 rad target offset per joint
export const RESET_NOISE = 0.05; // rad; reset joint-angle jitter uniform[-b, b]
export const FALL_HEIGHT = 0.14; // m; torso this low => fallen
export const UPRIGHT_MIN = 0.4; // up_z below this => tipped => fallen

/**
 * Assemble the float32[23] observation from raw scalar state. Pure — no sim, no
 * I/O. The quadruped obs is a DIRECT concatenation (no cos/sin encoding), so this
 * is just the field-order contract; the nontrivial reads live in buildObs (the
 * xmat up-vector + free-joint slots), which the WASM parity check verifies.
 *   [ jointAngles(8), jointVels(8), torsoHeight(1), torsoUp(3), torsoLinVel(3) ]
 * where torsoUp is the torso body's world z-axis ((0,0,1) = upright).
 */
export function assembleObs(
  jointAngles: ArrayLike<number>,
  jointVels: ArrayLike<number>,
  torsoHeight: number,
  torsoUp: ArrayLike<number>,
  torsoLinVel: ArrayLike<number>,
): Float32Array {
  const o = new Float32Array(OBS_DIM);
  for (let i = 0; i < 8; i++) o[i] = jointAngles[i];
  for (let i = 0; i < 8; i++) o[8 + i] = jointVels[i];
  o[16] = torsoHeight;
  o[17] = torsoUp[0]; o[18] = torsoUp[1]; o[19] = torsoUp[2];
  o[20] = torsoLinVel[0]; o[21] = torsoLinVel[1]; o[22] = torsoLinVel[2];
  return o;
}

/** The torso "up" vector: the 3rd column of the torso body's world rotation
 *  matrix (xmat is row-major, so column 2 is entries [2], [5], [8]). Mirrors
 *  quadruped_env.torso_up (xmat.reshape(3,3)[:, 2]). */
export function torsoUp(sim: Sim): [number, number, number] {
  const m = sim.bodyXmat(TORSO_BODY);
  return [m[2], m[5], m[8]];
}

/**
 * Build the observation from live sim state — identical to quadruped_env._obs().
 * The 8 leg hinges are single-dof named joints; the torso height, up-vector, and
 * linear velocity come from the free joint's qpos/qvel block + the torso xmat.
 */
export function buildObs(sim: Sim): Float32Array {
  const angles: number[] = [];
  const vels: number[] = [];
  for (const n of JOINT_NAMES) { angles.push(sim.jointQpos(n)); vels.push(sim.jointQvel(n)); }
  const rootQadr = sim.jointQposAdr(FREE_JOINT); // free joint: qpos[rootQadr .. +7] = [x,y,z, qw,qx,qy,qz]
  const rootVadr = sim.jointDofAdr(FREE_JOINT); // free joint: qvel[rootVadr .. +6] = [vx,vy,vz, wx,wy,wz]
  const height = sim.qposAt(rootQadr + 2);
  const up = torsoUp(sim);
  const linvel: [number, number, number] = [
    sim.qvelAt(rootVadr), sim.qvelAt(rootVadr + 1), sim.qvelAt(rootVadr + 2),
  ];
  return assembleObs(angles, vels, height, up, linvel);
}

/** Torso height (m) and forward velocity (m/s) — HUD conveniences. */
export function torsoHeight(sim: Sim): number {
  return sim.qposAt(sim.jointQposAdr(FREE_JOINT) + 2);
}
export function forwardVel(sim: Sim): number {
  return sim.qvelAt(sim.jointDofAdr(FREE_JOINT)); // world +x
}
