// Pusher-reach observation + task constants — the SINGLE SOURCE OF TRUTH for the
// browser's obs[8]/action[2] contract. Mirrors curriculum/common/envs/pusher_reach/
// pusher_reach_env.py (OBS/ACT layout, physics cadence, link geometry) so a
// ch2.2 SAC (and ch4 offline/serl) policy trained on pusher_reach_env is fed
// byte-identical observations in the browser. If the browser obs does not equal
// the training obs, the policy-drive loop breaks silently.
//
// Obs-parity is asserted by scripts/pusher_reach_obs_parity_check.mjs: assembleObs()
// output for a known state == PusherReachEnv._obs() for the same qpos/qvel/target
// (cross-checked against .venv python, within f32 tolerance). The nontrivial part
// the check pins is the analytic forward-kinematics for the fingertip (below) —
// it must equal MuJoCo's mj_forward site_xpos to floating-point.

import type { Sim } from '../sim/mujoco_sim';

// ------------------------------------------------------------------ dimensions
export const OBS_DIM = 8; // PusherReachEnv.OBS_DIM
export const ACT_DIM = 2; // PusherReachEnv.ACT_DIM

// ------------------------------------------------------------- physics cadence
export const CONTROL_HZ = 50; // PusherReachEnv.CONTROL_HZ
export const FRAME_SKIP = 2; // PusherReachEnv.FRAME_SKIP (100 Hz physics / 50 Hz ctrl, timestep 0.01)
export const CONTROL_DT = 1 / CONTROL_HZ; // 0.02 s of sim per control step
export const MAX_STEPS = 100; // PusherReachEnv.MAX_STEPS (2 s at 50 Hz)

// -------------------------------------------------------------------- geometry
export const LINK_LEN = 0.1; // m — each of the two links (total reach = 0.2 m)
export const SUCCESS_TOL = 0.02; // m — fingertip within this of target => success
// reset() target sampling: uniform annulus around the base, strictly inside reach
export const TARGET_R_MIN = 0.05; // m  (PusherReachEnv._TARGET_R[0])
export const TARGET_R_MAX = 0.19; // m  (PusherReachEnv._TARGET_R[1]; max < 2*LINK_LEN)

// The pusher-reach MuJoCo joints whose qpos/qvel we read (and write on reset).
export const SHOULDER_JOINT = 'shoulder'; // 1-dof hinge about +z
export const ELBOW_JOINT = 'elbow'; // 1-dof hinge about +z, relative to link 1

/** Wrap an angle to [-pi, pi). Mirrors pusher_reach_env.wrap_angle. */
export function wrapAngle(a: number): number {
  return ((((a + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
}

/**
 * Analytic forward kinematics for the planar 2-link arm's fingertip (world x, y).
 * The MJCF is: link1 hinge (shoulder) at the origin; link2 body at (LINK_LEN,0,0)
 * in link1's frame with hinge (elbow) relative to link1; the `fingertip` site at
 * (LINK_LEN,0,0) in link2's frame. So, with the elbow measured relative to link1:
 *     fingertip = R(shoulder)·(L,0) + R(shoulder+elbow)·(L,0)
 * This equals MuJoCo's mj_forward site_xpos to floating point (parity-checked) —
 * it is what lets the browser reconstruct dx/dy WITHOUT a WASM kinematics read.
 */
export function fingertipXY(shoulder: number, elbow: number): [number, number] {
  const x = LINK_LEN * (Math.cos(shoulder) + Math.cos(shoulder + elbow));
  const y = LINK_LEN * (Math.sin(shoulder) + Math.sin(shoulder + elbow));
  return [x, y];
}

/**
 * Assemble the float32[8] observation from raw scalar state. Pure — no sim, no
 * I/O — so it can be parity-checked against pusher_reach_env._obs() in isolation.
 * Field order and the cos/sin joint encoding are byte-identical to
 * pusher_reach_env._obs:
 *   [cos(shoulder), sin(shoulder), cos(elbow), sin(elbow),
 *    shoulder_angvel, elbow_angvel, dx, dy]
 * where (dx, dy) = target_xy − fingertip_xy (the dense reward signal). The joint
 * angles are wrapped to [-pi, pi) exactly as the Python env does before cos/sin
 * (cos/sin are 2pi-periodic so the wrap does not change them — kept for exact
 * parity); the fingertip FK uses the raw angles (also 2pi-periodic).
 */
export function assembleObs(
  shoulder: number,
  elbow: number,
  shoulderVel: number,
  elbowVel: number,
  targetX: number,
  targetY: number,
): Float32Array {
  const sh = wrapAngle(shoulder);
  const el = wrapAngle(elbow);
  const [fx, fy] = fingertipXY(shoulder, elbow);
  const o = new Float32Array(OBS_DIM);
  o[0] = Math.cos(sh);
  o[1] = Math.sin(sh);
  o[2] = Math.cos(el);
  o[3] = Math.sin(el);
  o[4] = shoulderVel;
  o[5] = elbowVel;
  o[6] = targetX - fx;
  o[7] = targetY - fy;
  return o; // Float32Array assignment rounds to f32, matching np.float32
}

/**
 * Build the observation from live sim state + the env's stored target. The obs a
 * trained SAC/offline/serl policy consumes in the browser comes from HERE —
 * identical to the training env's obs. The target is a mocap body with no
 * dynamics, so the browser env owns it (targetX/targetY) and passes it in.
 */
export function buildObs(sim: Sim, targetX: number, targetY: number): Float32Array {
  const shoulder = sim.jointQpos(SHOULDER_JOINT);
  const elbow = sim.jointQpos(ELBOW_JOINT);
  const shoulderVel = sim.jointQvel(SHOULDER_JOINT);
  const elbowVel = sim.jointQvel(ELBOW_JOINT);
  return assembleObs(shoulder, elbow, shoulderVel, elbowVel, targetX, targetY);
}

/** Live fingertip (world x, y) from the sim's joint angles — for the renderer. */
export function fingertip(sim: Sim): [number, number] {
  return fingertipXY(sim.jointQpos(SHOULDER_JOINT), sim.jointQpos(ELBOW_JOINT));
}

/** Live elbow-joint world position (world x, y) — for drawing the arm polyline. */
export function elbowXY(sim: Sim): [number, number] {
  const shoulder = sim.jointQpos(SHOULDER_JOINT);
  return [LINK_LEN * Math.cos(shoulder), LINK_LEN * Math.sin(shoulder)];
}

/** Fingertip→target distance (m), the quantity the dense reward penalizes. */
export function distToTarget(sim: Sim, targetX: number, targetY: number): number {
  const [fx, fy] = fingertip(sim);
  return Math.hypot(targetX - fx, targetY - fy);
}
