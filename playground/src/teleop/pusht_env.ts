// Browser mirror of curriculum/common/envs/pusht/pusht_env.py — the reset/step
// episode lifecycle over MuJoCo-WASM. Keeps the SAME cadence (10 Hz control /
// 100 Hz physics via FRAME_SKIP), the SAME action semantics (target velocity,
// clipped +-1), and the SAME success criterion (pos+ang tolerance held for
// SUCCESS_HOLD control steps). obs construction is delegated to pusht_obs so a
// Phase-3 policy is fed byte-identical observations.
import type { Sim } from '../sim/mujoco_sim';
import { Rng } from './rng';
import {
  ACT_DIM,
  ANG_TOL,
  buildObs,
  FRAME_SKIP,
  MAX_STEPS,
  POS_TOL,
  SUCCESS_HOLD,
  teeErrors,
} from './pusht_obs';

// reset() sampling ranges — mirror PushTEnv._SPAWN_R / _PUSHER_BOUND / _PUSHER_CLEAR.
const SPAWN_R: [number, number] = [0.1, 0.24]; // block distance from target
const PUSHER_BOUND = 0.32; // |x|,|y| bound for the pusher spawn
const PUSHER_CLEAR = 0.13; // min pusher distance from block center

export interface StepResult {
  obs: Float32Array; // observation AFTER the step
  reward: number;
  done: boolean;
  success: boolean;
  posErr: number;
  angErr: number;
  stepCount: number;
}

export class BrowserPushTEnv {
  private stepCount = 0;
  private successStreak = 0;
  private _success = false;
  private _lastSeed = 0;

  constructor(private sim: Sim) {}

  get success(): boolean {
    return this._success;
  }
  get steps(): number {
    return this.stepCount;
  }
  get lastSeed(): number {
    return this._lastSeed;
  }

  /** Current observation (obs a policy would consume). */
  obs(): Float32Array {
    return buildObs(this.sim);
  }

  errors(): { posErr: number; angErr: number } {
    return teeErrors(this.sim);
  }

  /**
   * Reseed the scene like pusht_env.reset: block on a uniform annulus around the
   * target with uniform yaw, pusher rejection-sampled clear of the block. See
   * rng.ts for the PCG64-parity caveat.
   */
  reset(seed: number): Float32Array {
    this._lastSeed = seed;
    const rng = new Rng(seed);
    this.sim.resetData();

    const r = rng.uniform(SPAWN_R[0], SPAWN_R[1]);
    const phi = rng.uniform(0, 2 * Math.PI);
    const teeX = r * Math.cos(phi);
    const teeY = r * Math.sin(phi);
    const teeYaw = rng.uniform(-Math.PI, Math.PI);

    let pusherX = 0;
    let pusherY = 0;
    // Rejection-sample the pusher clear of the block (deterministic given seed).
    for (;;) {
      pusherX = rng.uniform(-PUSHER_BOUND, PUSHER_BOUND);
      pusherY = rng.uniform(-PUSHER_BOUND, PUSHER_BOUND);
      if (Math.hypot(pusherX - teeX, pusherY - teeY) > PUSHER_CLEAR) break;
    }

    this.sim.setJointQpos('tee_x', teeX);
    this.sim.setJointQpos('tee_y', teeY);
    this.sim.setJointQpos('tee_yaw', teeYaw);
    this.sim.setJointQpos('pusher_x', pusherX);
    this.sim.setJointQpos('pusher_y', pusherY);
    this.sim.setCtrl([0, 0]);
    this.sim.forward();

    this.stepCount = 0;
    this.successStreak = 0;
    this._success = false;
    return this.obs();
  }

  /**
   * The "watch it recover" hero: teleport the T-block to a (hard) pose without
   * touching the pusher, then re-arm the episode clock so the loaded policy's
   * recovery attempt is scored from scratch. Honest by construction — a ~62%
   * policy will sometimes push it home and sometimes not. Unlike reset() this
   * keeps the pusher where it is, so it reads as "the block got knocked askew,
   * now fix it" rather than a fresh scene.
   */
  perturbBlock(teeX: number, teeY: number, teeYaw: number): Float32Array {
    this.sim.setJointQpos('tee_x', teeX);
    this.sim.setJointQpos('tee_y', teeY);
    this.sim.setJointQpos('tee_yaw', teeYaw);
    this.sim.setCtrl([0, 0]);
    this.sim.forward();
    this.stepCount = 0;
    this.successStreak = 0;
    this._success = false;
    return this.obs();
  }

  /**
   * One control step: clip the action to +-1, hold it for FRAME_SKIP physics
   * steps, then evaluate reward/success exactly as pusht_env.step.
   */
  step(action: ArrayLike<number>): StepResult {
    const clipped = new Float64Array(ACT_DIM);
    for (let i = 0; i < ACT_DIM; i++) {
      clipped[i] = Math.max(-1, Math.min(1, action[i] ?? 0));
    }
    this.sim.step(clipped, FRAME_SKIP);
    this.stepCount += 1;

    const { posErr, angErr } = this.errors();
    const inTol = posErr < POS_TOL && angErr < ANG_TOL;
    this.successStreak = inTol ? this.successStreak + 1 : 0;

    let reward = -0.5 * (posErr / 0.5 + angErr / Math.PI);
    if (!this._success && this.successStreak >= SUCCESS_HOLD) {
      this._success = true;
      reward += 1.0;
    }
    const done = this._success || this.stepCount >= MAX_STEPS;
    return {
      obs: this.obs(),
      reward,
      done,
      success: this._success,
      posErr,
      angErr,
      stepCount: this.stepCount,
    };
  }
}
