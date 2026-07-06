// Browser mirror of curriculum/common/envs/cartpole/cartpole_env.py — the
// reset/step episode lifecycle over MuJoCo-WASM. Keeps the SAME cadence
// (50 Hz control / 100 Hz physics via FRAME_SKIP), the SAME action semantics
// (force clipped to +-1, held FRAME_SKIP physics steps), the SAME termination
// (pole past ANGLE_LIMIT or cart past CART_LIMIT), and the SAME +1/step alive
// reward. obs construction is delegated to cartpole_obs so the PPO policy is fed
// byte-identical observations (obs-parity asserted by cartpole_obs_parity_check).
import type { Sim } from '../sim/mujoco_sim';
import { Rng } from './rng';
import {
  ACT_DIM,
  buildObs,
  CART_JOINT,
  CART_LIMIT,
  FRAME_SKIP,
  MAX_STEPS,
  poleAngle,
  POLE_JOINT,
  RESET_BOUND,
  ANGLE_LIMIT,
} from './cartpole_obs';

export interface StepResult {
  obs: Float32Array; // observation AFTER the step
  reward: number; // +1 alive bonus
  done: boolean; // terminated OR truncated
  terminated: boolean; // the pole fell / the cart ran off the rail
  truncated: boolean; // the 500-step budget ran out with the pole still up
  poleAngle: number; // wrapped pole angle (rad from upright)
  stepCount: number;
}

export class BrowserCartpoleEnv {
  private stepCount = 0;
  private _lastSeed = 0;

  constructor(private sim: Sim) {}

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

  /** Wrapped pole angle (rad from upright); 0 = balanced. */
  angle(): number {
    return poleAngle(this.sim);
  }

  /**
   * Reseed the scene like cartpole_env.reset: every state var drawn uniform in
   * [-RESET_BOUND, RESET_BOUND] so the pole starts near-upright. See rng.ts for
   * the PCG64-parity caveat — the browser start differs bit-for-bit from Python
   * (fine; only obs CONSTRUCTION must match, asserted separately).
   */
  reset(seed: number): Float32Array {
    this._lastSeed = seed;
    const rng = new Rng(seed);
    this.sim.resetData();

    this.sim.setJointQpos(CART_JOINT, rng.uniform(-RESET_BOUND, RESET_BOUND));
    this.sim.setJointQpos(POLE_JOINT, rng.uniform(-RESET_BOUND, RESET_BOUND));
    this.sim.setJointQvel(CART_JOINT, rng.uniform(-RESET_BOUND, RESET_BOUND));
    this.sim.setJointQvel(POLE_JOINT, rng.uniform(-RESET_BOUND, RESET_BOUND));
    this.sim.setCtrl([0]);
    this.sim.forward();

    this.stepCount = 0;
    return this.obs();
  }

  /**
   * The "watch it recover" hero: add a velocity impulse to the cart WITHOUT
   * resetting the episode, so the trained policy has to catch the pole from the
   * disturbance the visitor created. Writes to the slider qvel (a shove), then
   * re-derives dependent quantities. Honest by construction — a well-trained PPO
   * policy recovers because it visited exactly such states during training.
   */
  nudgeCart(deltaVel: number): void {
    this.sim.setJointQvel(CART_JOINT, this.sim.jointQvel(CART_JOINT) + deltaVel);
    this.sim.forward();
  }

  /**
   * One control step: clip the action to +-1, hold it for FRAME_SKIP physics
   * steps, then evaluate termination/truncation exactly as cartpole_env.step.
   * Reward is the classic +1 alive bonus (episode return == steps survived).
   */
  step(action: ArrayLike<number>): StepResult {
    const force = Math.max(-1, Math.min(1, action[0] ?? 0));
    const ctrl = new Float64Array(ACT_DIM);
    ctrl[0] = force;
    this.sim.step(ctrl, FRAME_SKIP);
    this.stepCount += 1;

    const angle = poleAngle(this.sim);
    const cartPos = this.sim.jointQpos(CART_JOINT);
    const terminated = Math.abs(angle) > ANGLE_LIMIT || Math.abs(cartPos) > CART_LIMIT;
    const truncated = this.stepCount >= MAX_STEPS;
    return {
      obs: this.obs(),
      reward: 1.0,
      done: terminated || truncated,
      terminated,
      truncated,
      poleAngle: angle,
      stepCount: this.stepCount,
    };
  }
}
