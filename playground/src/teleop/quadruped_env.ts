// Browser mirror of curriculum/common/envs/quadruped/quadruped_env.py — the
// reset/step episode lifecycle over MuJoCo-WASM for the free-floating quadruped.
// Keeps the SAME cadence (50 Hz control / 200 Hz physics via FRAME_SKIP=4,
// timestep 0.005), the SAME residual-position action semantics (ctrl = DEFAULT_POSE
// + ACTION_SCALE * clip(action, -1, 1), held FRAME_SKIP physics steps), the SAME
// five-term reward, and the SAME termination (fall: height < FALL_HEIGHT OR
// up_z < UPRIGHT_MIN) vs truncation (step budget). obs construction is delegated to
// quadruped_obs (buildObs) so a ch2.5 walk / ch2.4 rewards / ch2.7 DR policy is fed
// byte-identical observations in the browser — obs-parity over a real contact
// rollout is asserted by scripts/quadruped_obs_parity_check.mjs.
//
// The base is a 6-DOF FREE JOINT ("root"): its 7-slot qpos block is
// [x, y, z, qw, qx, qy, qz] and its 6-dof qvel block is [vx, vy, vz, wx, wy, wz]
// (world frame). reset() writes the torso to STAND_HEIGHT, upright (identity
// quaternion), at rest, with the 8 leg hinges at the nominal crouch + seeded
// jitter — exactly quadruped_env.reset (the reset RNG differs from numpy PCG64,
// which is fine: only obs CONSTRUCTION must match, asserted separately; see rng.ts).
import type { Sim } from '../sim/mujoco_sim';
import { Rng } from './rng';
import {
  ACT_DIM,
  ACTION_SCALE,
  buildObs,
  DEFAULT_POSE,
  FALL_HEIGHT,
  FRAME_SKIP,
  FREE_JOINT,
  JOINT_NAMES,
  MAX_STEPS,
  RESET_NOISE,
  STAND_HEIGHT,
  TORSO_BODY,
  UPRIGHT_MIN,
  forwardVel,
  torsoHeight,
  torsoUp,
} from './quadruped_obs';

// ------------------------------------------------------------------ reward terms
// Mirrors QuadrupedEnv._reward — five named, weighted terms. The toys surface a
// subset (forward distance, height, up_z, and the height term the ch2.4 hack
// climbs) so the reward-vs-behaviour story is visible.
const TARGET_HEIGHT = 0.25; // m; height the height-penalty is measured against
const MAX_VX = 1.0; // m/s; forward-velocity reward is clipped to +-this
const W_FORWARD = 1.0;
const W_UPRIGHT = 0.2;
const W_HEIGHT = 5.0;
const W_ALIVE = 0.2;
const W_CTRL = 0.001;

export interface RewardTerms {
  forward: number;
  upright: number;
  height: number;
  alive: number;
  ctrl: number;
}

export interface QuadStepResult {
  obs: Float32Array; // observation AFTER the step
  reward: number; // the env's default summed reward (five terms)
  terms: RewardTerms; // per-term breakdown (info["reward_terms"])
  done: boolean; // terminated (fell) OR truncated (step budget)
  terminated: boolean; // fell: height < FALL_HEIGHT or up_z < UPRIGHT_MIN
  truncated: boolean; // reached MAX_STEPS still upright
  height: number; // torso-center world z (m)
  upZ: number; // torso up-vector z-component (uprightness)
  forwardVel: number; // world +x velocity (m/s)
  forwardDist: number; // torso x travelled since reset (m)
  stepCount: number;
}

export class BrowserQuadrupedEnv {
  private stepCount = 0;
  private _lastSeed = 0;
  private _x0 = 0; // torso x at reset (forward-distance origin)
  private _rootQadr = 0;
  private _rootVadr = 0;

  constructor(private sim: Sim) {
    this._rootQadr = sim.jointQposAdr(FREE_JOINT);
    this._rootVadr = sim.jointDofAdr(FREE_JOINT);
  }

  get steps(): number {
    return this.stepCount;
  }
  get lastSeed(): number {
    return this._lastSeed;
  }

  /** Current observation (what a policy consumes) — obs[23], verbatim buildObs. */
  obs(): Float32Array {
    return buildObs(this.sim);
  }

  /** Torso-center world z (m). */
  height(): number {
    return torsoHeight(this.sim);
  }
  /** Torso up-vector z-component — 1.0 = perfectly upright. */
  upZ(): number {
    return torsoUp(this.sim)[2];
  }
  /** World +x velocity (m/s) — the reward's forward term reads this. */
  forwardVel(): number {
    return forwardVel(this.sim);
  }
  /** Torso x travelled since the last reset (m). */
  forwardDist(): number {
    return this.sim.qposAt(this._rootQadr) - this._x0;
  }
  /** True once the robot has fallen (terminated) at the current state. */
  fallen(): boolean {
    return this.height() < FALL_HEIGHT || this.upZ() < UPRIGHT_MIN;
  }

  /**
   * Reseed like quadruped_env.reset: torso at STAND_HEIGHT, upright (identity
   * quaternion), at rest; the 8 leg hinges at DEFAULT_POSE + small seeded jitter;
   * the PD servos commanded to the noiseless nominal pose (so action 0 = stand).
   */
  reset(seed: number): Float32Array {
    this._lastSeed = seed;
    const rng = new Rng(seed);
    this.sim.resetData();

    const q = this._rootQadr;
    this.sim.setQposAt(q + 0, 0.0);
    this.sim.setQposAt(q + 1, 0.0);
    this.sim.setQposAt(q + 2, STAND_HEIGHT);
    this.sim.setQposAt(q + 3, 1.0); // qw
    this.sim.setQposAt(q + 4, 0.0); // qx
    this.sim.setQposAt(q + 5, 0.0); // qy
    this.sim.setQposAt(q + 6, 0.0); // qz

    for (let i = 0; i < ACT_DIM; i++) {
      const jitter = rng.uniform(-RESET_NOISE, RESET_NOISE);
      this.sim.setJointQpos(JOINT_NAMES[i], DEFAULT_POSE[i] + jitter);
      this.sim.setJointQvel(JOINT_NAMES[i], 0);
    }
    // free-joint velocities to rest
    for (let i = 0; i < 6; i++) this.sim.setQvelAt(this._rootVadr + i, 0);

    // PD servos hold the (noiseless) nominal pose at reset -> action 0 = stand
    this.sim.setCtrl(DEFAULT_POSE);
    this.sim.forward();

    this.stepCount = 0;
    this._x0 = this.sim.qposAt(q); // forward-distance origin
    return this.obs();
  }

  /**
   * One control step: clip the residual action to [-1, 1], command
   * DEFAULT_POSE + ACTION_SCALE * action to the 8 PD position servos, hold it for
   * FRAME_SKIP physics steps, then evaluate the five-term reward + fall/timeout
   * exactly as quadruped_env.step. Episodes END on a fall (terminated) or the
   * MAX_STEPS budget (truncated) — the caller decides whether to re-reset.
   */
  step(action: ArrayLike<number>): QuadStepResult {
    const ctrl = new Float64Array(ACT_DIM);
    let sumSq = 0;
    for (let i = 0; i < ACT_DIM; i++) {
      const a = Math.max(-1, Math.min(1, action[i] ?? 0));
      ctrl[i] = DEFAULT_POSE[i] + ACTION_SCALE * a;
      sumSq += a * a;
    }
    this.sim.step(ctrl, FRAME_SKIP);
    this.stepCount += 1;

    const height = this.height();
    const upZ = this.upZ();
    const fwdVel = this.forwardVel();
    const vx = Math.max(-MAX_VX, Math.min(MAX_VX, fwdVel));
    const terms: RewardTerms = {
      forward: W_FORWARD * vx,
      upright: W_UPRIGHT * upZ,
      height: -W_HEIGHT * (height - TARGET_HEIGHT) ** 2,
      alive: W_ALIVE,
      ctrl: -W_CTRL * sumSq,
    };
    const reward = terms.forward + terms.upright + terms.height + terms.alive + terms.ctrl;
    const terminated = height < FALL_HEIGHT || upZ < UPRIGHT_MIN;
    const truncated = this.stepCount >= MAX_STEPS;

    return {
      obs: this.obs(),
      reward,
      terms,
      done: terminated || truncated,
      terminated,
      truncated,
      height,
      upZ,
      forwardVel: fwdVel,
      forwardDist: this.forwardDist(),
      stepCount: this.stepCount,
    };
  }
}

// Convenience re-exports for the toys (torso body name + the raw height reward the
// ch2.4 height-hack maximizes: -W_HEIGHT*(h-h*)^2 is the shaped term; the hack's
// program is HACK_HEIGHT_W * raw height — see rewards.py — so a toy that wants the
// "its reward climbs" readout reads torso height directly).
export { TORSO_BODY, JOINT_NAMES, MAX_STEPS };
