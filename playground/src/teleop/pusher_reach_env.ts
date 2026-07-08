// Browser mirror of curriculum/common/envs/pusher_reach/pusher_reach_env.py — the
// reset/step episode lifecycle over MuJoCo-WASM. Keeps the SAME cadence
// (50 Hz control / 100 Hz physics via FRAME_SKIP), the SAME action semantics
// (torque clipped to +-1, held FRAME_SKIP physics steps), the SAME dense reward
// (−fingertip→target distance) and success rule (dist < SUCCESS_TOL). obs
// construction is delegated to pusher_reach_obs so a SAC/offline/serl policy is
// fed byte-identical observations (obs-parity asserted by
// scripts/pusher_reach_obs_parity_check.mjs).
//
// The target is a MOCAP body in the MJCF (no dynamics, no collision), so — exactly
// as the Python env sets data.mocap_pos each reset — the browser env OWNS the
// target position (targetX/targetY) and never lets it perturb the arm. obs and the
// renderer both read the env's stored target; the WASM sim only integrates the
// two-link arm. Dragging the target (the "re-reach" hero) just moves that stored
// point, so the policy — which trained on exactly such fingertip→target vectors —
// chases it.
import type { Sim } from '../sim/mujoco_sim';
import { Rng } from './rng';
import {
  ACT_DIM,
  buildObs,
  FRAME_SKIP,
  LINK_LEN,
  MAX_STEPS,
  SHOULDER_JOINT,
  ELBOW_JOINT,
  SUCCESS_TOL,
  TARGET_R_MIN,
  TARGET_R_MAX,
  distToTarget,
} from './pusher_reach_obs';

export interface StepResult {
  obs: Float32Array; // observation AFTER the step
  reward: number; // dense: −distance (+ one-time SUCCESS_BONUS on first reach)
  done: boolean; // truncated at MAX_STEPS (no early termination by default)
  success: boolean; // latched: dist < SUCCESS_TOL was reached this episode
  dist: number; // fingertip→target distance (m)
  stepCount: number;
}

const SUCCESS_BONUS = 1.0; // PusherReachEnv.SUCCESS_BONUS (one-time, on first reach)

export class BrowserPusherReachEnv {
  private stepCount = 0;
  private _lastSeed = 0;
  private _success = false;
  private _targetX = 0.15;
  private _targetY = 0.0;

  constructor(private sim: Sim) {}

  get steps(): number {
    return this.stepCount;
  }
  get lastSeed(): number {
    return this._lastSeed;
  }
  /** The env's stored target (world x, y) — obs and the renderer read this. */
  get target(): [number, number] {
    return [this._targetX, this._targetY];
  }

  /** Current observation (what a policy consumes). */
  obs(): Float32Array {
    return buildObs(this.sim, this._targetX, this._targetY);
  }

  /** Fingertip→target distance (m) — the dense-reward magnitude, for the HUD. */
  dist(): number {
    return distToTarget(this.sim, this._targetX, this._targetY);
  }

  /**
   * Reseed like pusher_reach_env.reset: both joint angles uniform in [-pi, pi),
   * velocities at rest, and the target on a uniform annulus strictly inside the
   * arm's reach (so a reach solution always exists). See rng.ts for the
   * PCG64-parity caveat — the browser start differs bit-for-bit from Python (fine;
   * only obs CONSTRUCTION must match, asserted separately).
   */
  reset(seed: number): Float32Array {
    this._lastSeed = seed;
    const rng = new Rng(seed);
    this.sim.resetData();

    this.sim.setJointQpos(SHOULDER_JOINT, rng.uniform(-Math.PI, Math.PI));
    this.sim.setJointQpos(ELBOW_JOINT, rng.uniform(-Math.PI, Math.PI));
    this.sim.setJointQvel(SHOULDER_JOINT, 0);
    this.sim.setJointQvel(ELBOW_JOINT, 0);

    const r = rng.uniform(TARGET_R_MIN, TARGET_R_MAX);
    const phi = rng.uniform(0, 2 * Math.PI);
    this._targetX = r * Math.cos(phi);
    this._targetY = r * Math.sin(phi);

    this.sim.setCtrl([0, 0]);
    this.sim.forward();

    this.stepCount = 0;
    this._success = false;
    return this.obs();
  }

  /**
   * Move the target WITHOUT resetting the episode — the "watch it re-reach" hero.
   * The visitor drags the green dot; the running policy, which trained on
   * fingertip→target vectors, closes on the new goal. The point is clamped to the
   * reachable annulus so a solution always exists (honest: a fair re-reach, not an
   * unreachable trick). Radius is bounded by TARGET_R_MAX (< arm length).
   */
  setTarget(x: number, y: number): void {
    const r = Math.hypot(x, y);
    if (r < 1e-6) {
      this._targetX = TARGET_R_MIN;
      this._targetY = 0;
      return;
    }
    const clamped = Math.max(TARGET_R_MIN, Math.min(TARGET_R_MAX, r));
    const s = clamped / r;
    this._targetX = x * s;
    this._targetY = y * s;
  }

  /**
   * One control step: clip the torque action to +-1, hold it for FRAME_SKIP
   * physics steps, then evaluate the dense reward + success exactly as
   * pusher_reach_env.step. Default (Reacher-style) has NO early termination, so
   * done latches only at the MAX_STEPS budget.
   */
  step(action: ArrayLike<number>): StepResult {
    const ctrl = new Float64Array(ACT_DIM);
    for (let i = 0; i < ACT_DIM; i++) ctrl[i] = Math.max(-1, Math.min(1, action[i] ?? 0));
    this.sim.step(ctrl, FRAME_SKIP);
    this.stepCount += 1;

    const dist = this.dist();
    let reward = -dist;
    if (!this._success && dist < SUCCESS_TOL) {
      this._success = true;
      reward += SUCCESS_BONUS;
    }
    const truncated = this.stepCount >= MAX_STEPS;
    return {
      obs: this.obs(),
      reward,
      done: truncated,
      success: this._success,
      dist,
      stepCount: this.stepCount,
    };
  }
}

// The arm's total reach (both links extended) — a UI convenience (the max target
// radius the drag handle is clamped to is TARGET_R_MAX < this).
export const ARM_REACH = 2 * LINK_LEN;
