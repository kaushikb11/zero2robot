// Cartpole observation + task constants — the SINGLE SOURCE OF TRUTH for the
// browser's obs[5]/action[1] contract. Mirrors curriculum/common/envs/cartpole/
// cartpole_env.py (OBS/ACT layout, physics cadence, termination limits) so a
// ch2.1 PPO policy trained on cartpole_env is fed byte-identical observations in
// the browser. If the browser obs does not equal the training obs, the
// policy-drive loop breaks silently.
//
// Obs-parity is asserted by scripts/cartpole_obs_parity_check.mjs: assembleObs()
// output for a known state == CartpoleEnv._obs() for the same qpos/qvel
// (cross-checked against .venv python, within f32 tolerance).

import type { Sim } from '../sim/mujoco_sim';

// ------------------------------------------------------------------ dimensions
export const OBS_DIM = 5; // CartpoleEnv.OBS_DIM
export const ACT_DIM = 1; // CartpoleEnv.ACT_DIM

// ------------------------------------------------------------- physics cadence
export const CONTROL_HZ = 50; // CartpoleEnv.CONTROL_HZ
export const FRAME_SKIP = 2; // CartpoleEnv.FRAME_SKIP (100 Hz physics / 50 Hz ctrl, timestep 0.01)
export const CONTROL_DT = 1 / CONTROL_HZ; // 0.02 s of sim per control step
export const MAX_STEPS = 500; // CartpoleEnv.MAX_STEPS (10 s at 50 Hz)

// ----------------------------------------------------------- termination limits
export const ANGLE_LIMIT = 0.2095; // rad (~12 deg) — pole-fall threshold (CartpoleEnv.ANGLE_LIMIT)
export const CART_LIMIT = 2.4; // m — cart-off-rail threshold (CartpoleEnv.CART_LIMIT)
export const RESET_BOUND = 0.05; // reset draws each state var from uniform[-b, b]

// The cartpole MuJoCo joints whose qpos/qvel we read (and write on reset/nudge).
export const CART_JOINT = 'slider'; // 1-dof slide: cart position along the rail
export const POLE_JOINT = 'hinge'; // 1-dof hinge: pole angle (0 = upright)

/** Wrap an angle to [-pi, pi). Mirrors cartpole_env.wrap_angle. */
export function wrapAngle(a: number): number {
  return ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
}

/**
 * Assemble the float32[5] observation from raw scalar state. Pure — no sim, no
 * I/O — so it can be parity-checked against cartpole_env._obs() in isolation.
 * Field order and the cos/sin(angle-from-upright) encoding are byte-identical to
 * cartpole_env._obs: [cart_pos, cart_vel, cos(theta), sin(theta), pole_angvel].
 * `poleAngle` is the raw hinge angle; it is wrapped to [-pi, pi) here exactly as
 * the Python env's pole_angle property does before cos/sin (cos/sin are
 * 2pi-periodic so the wrap does not change them — kept for exact parity).
 */
export function assembleObs(
  cartPos: number,
  cartVel: number,
  poleAngle: number,
  poleAngvel: number,
): Float32Array {
  const theta = wrapAngle(poleAngle);
  const o = new Float32Array(OBS_DIM);
  o[0] = cartPos;
  o[1] = cartVel;
  o[2] = Math.cos(theta);
  o[3] = Math.sin(theta);
  o[4] = poleAngvel;
  return o; // Float32Array assignment rounds to f32, matching np.float32
}

/**
 * Build the observation from live sim state. The obs a trained PPO policy
 * consumes in the browser comes from HERE — identical to the training env's obs.
 */
export function buildObs(sim: Sim): Float32Array {
  const cartPos = sim.jointQpos(CART_JOINT);
  const cartVel = sim.jointQvel(CART_JOINT);
  const poleAngle = sim.jointQpos(POLE_JOINT);
  const poleAngvel = sim.jointQvel(POLE_JOINT);
  return assembleObs(cartPos, cartVel, poleAngle, poleAngvel);
}

/** The wrapped pole angle (rad from upright), for the HUD/termination check. */
export function poleAngle(sim: Sim): number {
  return wrapAngle(sim.jointQpos(POLE_JOINT));
}

/** cartpole_env._fallen: pole past the angle limit OR cart off the rail. */
export function fallen(sim: Sim): boolean {
  return Math.abs(poleAngle(sim)) > ANGLE_LIMIT
    || Math.abs(sim.jointQpos(CART_JOINT)) > CART_LIMIT;
}
