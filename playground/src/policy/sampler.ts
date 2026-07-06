// The sampler-aware (contract-v2) runtime CORE — the iterative ODE integrator a
// generative policy needs, factored OUT of any onnxruntime dependency so it is
// unit-testable in plain Node (scripts/flow_sampler_parity_check.mjs transpiles
// this file and drives it with the real ONNX velocity net, cross-checked against
// the Python sampler in curriculum/.../ch1.5_flow/flow.py).
//
// Why this exists: tensor contract v1 is a single stateless step,
// model(obs[1,obs]) -> action[1,act]. A GENERATIVE policy (ch1.4 diffusion,
// ch1.5 flow) does not fit it: one action is produced by evaluating a
// velocity/denoiser net `num_steps` times, chained through an integrator the
// RUNTIME must own. contract v2 (see contracts.ts) ships the net + the sampler
// metadata; this file is the integrator. infer.ts wraps it with the ORT session.
//
// Determinism note (mirrors teleop/rng.ts's honest limitation): GaussianRng is a
// compact SEEDED standard-normal source so a browser rollout is reproducible
// (same seed -> same actions, every run). It is NOT torch's Mersenne-Twister, so
// the exact per-seed noise draw differs from flow.py's Python env — exactly as
// rng.ts's block spawns differ from pusht_env's. What IS proven identical (the
// load-bearing property) is the SAMPLER: given the SAME noise + obs, eulerSample
// reproduces flow.py's ode_sample_loop + un-standardization bit-for-bit within
// f32 tolerance (the parity check feeds both sides the same noise).

/** A seeded standard-normal generator: mulberry32 uniform core (identical to
 *  teleop/rng.ts) + Box–Muller with a cached spare. Seeding it makes the flow
 *  sampler's noise init reproducible so a live rollout replays deterministically. */
export class GaussianRng {
  private s: number;
  private spare: number | null = null;

  constructor(seed: number) {
    // Mix the seed so small seeds (0, 1, 2) produce well-separated streams —
    // same mixing as teleop/rng.ts.
    this.s = (seed ^ 0x9e3779b9) >>> 0;
  }

  /** Reseed the stream in place (drops any cached Box–Muller spare). */
  reseed(seed: number): void {
    this.s = (seed ^ 0x9e3779b9) >>> 0;
    this.spare = null;
  }

  /** Uniform float in [0, 1) — mulberry32. */
  private uniform(): number {
    let t = (this.s += 0x6d2b79f5) >>> 0;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** One standard-normal draw (mean 0, std 1). */
  normal(): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return v;
    }
    // Box–Muller: two uniforms -> two normals; keep the sine one for next call.
    const u1 = Math.max(this.uniform(), 1e-12); // guard log(0)
    const u2 = this.uniform();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    this.spare = r * Math.sin(theta);
    return r * Math.cos(theta);
  }

  /** Fill `out` with standard-normal draws (in place) and return it. */
  fill(out: Float32Array): Float32Array {
    for (let i = 0; i < out.length; i++) out[i] = this.normal();
    return out;
  }
}

/** Evaluate the velocity net at (point, t). Returns the velocity in the same
 *  standardized action space as `point`. Async because the ONNX session.run is
 *  async in the browser; the parity check supplies a synchronous-resolving one. */
export type VelocityFn = (point: Float32Array, t: number) => Promise<Float32Array>;

/**
 * Forward-Euler ODE sampler — a bit-faithful JS port of flow.py's
 * `ode_sample_loop` + the rollout's un-standardization:
 *
 *   x = noise                       # start at pure noise, flow time t = 0
 *   dt = 1 / num_steps
 *   for i in range(num_steps):
 *       t = i * dt
 *       x = x + dt * velocity(x, t) # forward Euler along the velocity field
 *   action = clamp(x * act_std + act_mean, -1, 1)   # denormalize
 *
 * All arithmetic is kept in f32 stepwise (Math.fround on the dt*v product and
 * the denorm product) so it matches the Python float32 tensor math term-for-term
 * — over 100 steps a naive f64 accumulation would drift out of parity tolerance.
 *
 * @param noise    initial point in STANDARDIZED action space, length act_dim.
 * @param numSteps Euler steps (flow_steps). Fewer => curved-path error (the
 *                 chapter's Break-It); the runtime exposes it as a live control.
 * @param velocity the (ONNX) velocity net, conditioned on the obs by the caller.
 * @param actMean  per-dim action mean used to un-standardize (from the ckpt).
 * @param actStd   per-dim action std used to un-standardize (from the ckpt).
 */
export async function eulerSample(
  noise: Float32Array,
  numSteps: number,
  velocity: VelocityFn,
  actMean: ArrayLike<number>,
  actStd: ArrayLike<number>,
): Promise<Float32Array> {
  if (!Number.isInteger(numSteps) || numSteps < 1) {
    throw new Error(`flow sampler needs a positive integer num_steps, got ${numSteps}`);
  }
  const act = noise.length;
  if (actMean.length !== act || actStd.length !== act) {
    throw new Error(
      `act_mean/act_std length (${actMean.length}/${actStd.length}) != action dim ${act}`,
    );
  }
  let x = Float32Array.from(noise);
  const dt = 1.0 / numSteps;
  for (let i = 0; i < numSteps; i++) {
    const t = Math.fround(i * dt); // t is a float32 in flow.py (torch.full default dtype)
    const v = await velocity(x, t);
    if (v.length !== act) {
      throw new Error(`velocity net returned dim ${v.length}, expected action dim ${act}`);
    }
    const nx = new Float32Array(act); // storing into a Float32Array rounds to f32
    for (let k = 0; k < act; k++) nx[k] = x[k] + Math.fround(dt * v[k]);
    x = nx;
  }
  // Un-standardize then clamp to the actuator range, exactly like flow.py's
  // rollout: (sample * act_std + act_mean).clamp(-1, 1).
  const out = new Float32Array(act);
  for (let k = 0; k < act; k++) {
    const denorm = Math.fround(x[k] * actStd[k]) + actMean[k];
    out[k] = Math.min(1, Math.max(-1, Math.fround(denorm)));
  }
  return out;
}

// ===========================================================================
// DDPM ancestral sampler — the ch1.4 diffusion sibling of eulerSample. Where
// flow integrates a schedule-free ODE, diffusion runs the REVERSE noise process:
// start from pure noise x_T and walk back to a sample one denoising step at a
// time, each step subtracting the net's predicted noise (eps), rescaling by the
// schedule, and re-injecting a little fresh noise (except the last step). This is
// a bit-faithful JS port of diffusion.py's make_schedule (cosine branch) +
// p_sample_loop + the rollout un-standardization; the JS-vs-Python parity check
// (scripts/ddpm_sampler_parity_check.mjs) proves it reproduces the chapter within
// f32 tol. All arithmetic is kept in f32 stepwise (Math.fround) so it matches the
// Python float32 tensor math term-for-term.
// ===========================================================================

/** The precomputed forward-process constants for `steps` denoising steps, all
 *  length `steps` (f32). Mirrors the dict make_schedule returns in diffusion.py. */
export interface DdpmSchedule {
  steps: number;
  betas: Float32Array;
  alphas: Float32Array;
  acp: Float32Array;            // alpha-bar (cumprod of alphas) — signal fraction left at step t
  acpPrev: Float32Array;        // acp shifted one step (acp_prev[0] = 1)
  sqrtAcp: Float32Array;
  sqrtOneMinusAcp: Float32Array;
  postSigma: Float32Array;      // per-step reverse noise std (posterior beta-tilde, sqrt)
}

/** f32 helpers so the schedule/loop match torch's float32 tensor ops term-for-term. */
const f = Math.fround;
const fsqrt = (x: number) => f(Math.sqrt(x));

/**
 * Rebuild the COSINE noise schedule for `steps` (Nichol & Dhariwal), a
 * bit-faithful port of diffusion.py's `make_schedule` non-broken branch:
 *
 *   u        = linspace(0, steps, steps+1) / steps
 *   acp_full = cos((u + 0.008) / 1.008 * pi/2) ** 2
 *   acp_full = acp_full / acp_full[0]
 *   betas    = (1 - acp_full[1:] / acp_full[:-1]).clamp(1e-8, 0.999)
 *
 * The runtime rebuilds the schedule per step-count so the live `denoising_steps`
 * control re-derives the schedule EXACTLY as `diffusion.py --denoising_steps n`
 * does (changing steps rebuilds the whole schedule — this is not strided/DDIM
 * sub-sampling). All ops are f32 to track torch's float32 make_schedule.
 */
export function cosineBetas(steps: number): Float32Array {
  if (!Number.isInteger(steps) || steps < 1) {
    throw new Error(`ddpm schedule needs a positive integer num_steps, got ${steps}`);
  }
  const HALF_PI = f(Math.PI / 2);
  const acpFull = new Float32Array(steps + 1);
  for (let i = 0; i <= steps; i++) {
    const u = f(f(i) / steps);              // linspace(0, steps, steps+1)/steps
    const inner = f(f(f(u + 0.008) / 1.008) * HALF_PI);
    const c = f(Math.cos(inner));
    acpFull[i] = f(c * c);                   // cos(...)**2
  }
  const a0 = acpFull[0];
  for (let i = 0; i <= steps; i++) acpFull[i] = f(acpFull[i] / a0);
  const betas = new Float32Array(steps);
  for (let i = 0; i < steps; i++) {
    let b = f(1 - f(acpFull[i + 1] / acpFull[i]));
    b = Math.min(0.999, Math.max(1e-8, b)); // .clamp(1e-8, 0.999)
    betas[i] = f(b);
  }
  return betas;
}

/** Derive the full DDPM schedule from a betas array, a bit-faithful port of the
 *  tail of diffusion.py's make_schedule (alphas, acp cumprod, acp_prev, the
 *  posterior variance beta-tilde). Pure f32 to match torch's float32 ops. */
export function makeDdpmSchedule(betas: ArrayLike<number>): DdpmSchedule {
  const steps = betas.length;
  if (steps < 1) throw new Error('ddpm schedule needs at least one beta');
  const b = new Float32Array(steps);
  const alphas = new Float32Array(steps);
  const acp = new Float32Array(steps);
  const acpPrev = new Float32Array(steps);
  const sqrtAcp = new Float32Array(steps);
  const sqrtOneMinusAcp = new Float32Array(steps);
  const postSigma = new Float32Array(steps);
  for (let i = 0; i < steps; i++) {
    b[i] = f(betas[i]);
    alphas[i] = f(1 - b[i]);
    acp[i] = i === 0 ? alphas[i] : f(acp[i - 1] * alphas[i]); // cumprod, f32
  }
  for (let i = 0; i < steps; i++) {
    acpPrev[i] = i === 0 ? 1 : acp[i - 1];                     // cat([ones(1), acp[:-1]])
    sqrtAcp[i] = fsqrt(acp[i]);
    sqrtOneMinusAcp[i] = fsqrt(f(1 - acp[i]));
    // post_var = betas * (1 - acp_prev) / (1 - acp); post_sigma = sqrt(clamp_min(post_var, 1e-20))
    const postVar = f(f(b[i] * f(1 - acpPrev[i])) / f(1 - acp[i]));
    postSigma[i] = fsqrt(Math.max(postVar, 1e-20));
  }
  return { steps, betas: b, alphas, acp, acpPrev, sqrtAcp, sqrtOneMinusAcp, postSigma };
}

/** Evaluate the denoiser net at (point, step). Returns the predicted NOISE (eps)
 *  in standardized action space. `step` is the integer reverse-diffusion step
 *  index, carried to the ONNX net as the float32 "flow_time" input (its embedding
 *  does t.float() internally, so an integer-valued float is identical). */
export type DenoiserFn = (point: Float32Array, step: number) => Promise<Float32Array>;

/**
 * Ancestral DDPM reverse sampler — a bit-faithful JS port of diffusion.py's
 * `p_sample_loop` + the rollout's un-standardization:
 *
 *   x = noise[0]                                  # x_T ~ N(0, I)
 *   for step in reversed(range(steps)):
 *       eps = denoiser(x, step)
 *       x0  = clamp((x - sqrt(1-acp)*eps)/sqrt(acp), -x0_clip, x0_clip)
 *       mean = betas*sqrt(acp_prev)/(1-acp)*x0 + (1-acp_prev)*sqrt(alpha)/(1-acp)*x
 *       x = mean + (post_sigma*noise[k] if step > 0 else 0)
 *   action = clamp(x * act_std + act_mean, -1, 1)
 *
 * Noise is fed EXPLICITLY as `noiseSeq` (length steps): index 0 is x_T, indices
 * 1..steps-1 are the per-reverse-step z draws in loop order (step steps-1, ...,
 * 1; the final step draws none). The parity check feeds the SAME sequence to both
 * sides; the browser policy draws it from its seeded GaussianRng.
 *
 * @param noiseSeq standardized-space noise: [x_T, z@(steps-1), ..., z@1], each act_dim.
 * @param schedule the DDPM schedule for THIS step count (makeDdpmSchedule).
 * @param denoiser the (ONNX) eps-prediction net, conditioned on the obs by the caller.
 * @param x0Clip   clamp on the predicted clean sample (diffusion.py X0_CLIP).
 * @param actMean/actStd per-dim un-standardization stats (from the ckpt).
 */
export async function ddpmSample(
  noiseSeq: ArrayLike<Float32Array>,
  schedule: DdpmSchedule,
  denoiser: DenoiserFn,
  x0Clip: number,
  actMean: ArrayLike<number>,
  actStd: ArrayLike<number>,
): Promise<Float32Array> {
  const steps = schedule.steps;
  if (noiseSeq.length !== steps) {
    throw new Error(`ddpm sampler needs ${steps} noise draws (x_T + one per step>0), got ${noiseSeq.length}`);
  }
  const act = noiseSeq[0].length;
  if (actMean.length !== act || actStd.length !== act) {
    throw new Error(
      `act_mean/act_std length (${actMean.length}/${actStd.length}) != action dim ${act}`,
    );
  }
  let x = Float32Array.from(noiseSeq[0]); // x_T (pure noise)
  let zi = 1;                             // index into the per-step z draws
  for (let step = steps - 1; step >= 0; step--) {
    const eps = await denoiser(x, step);
    if (eps.length !== act) {
      throw new Error(`denoiser net returned dim ${eps.length}, expected action dim ${act}`);
    }
    const acp = schedule.acp[step];
    const oneMinusAcp = f(1 - acp);
    const sqrtAcp = schedule.sqrtAcp[step];
    const sqrtOneMinusAcp = schedule.sqrtOneMinusAcp[step];
    // Scalar reverse-posterior coefficients (torch evaluates the 0-dim tensor
    // sub-expressions to a scalar BEFORE broadcasting over x0 / x).
    const coefX0 = f(f(schedule.betas[step] * fsqrt(schedule.acpPrev[step])) / oneMinusAcp);
    const coefX = f(f(f(1 - schedule.acpPrev[step]) * fsqrt(schedule.alphas[step])) / oneMinusAcp);
    const mean = new Float32Array(act);
    for (let k = 0; k < act; k++) {
      // x0 = clamp((x - sqrt(1-acp)*eps) / sqrt(acp), -x0_clip, x0_clip)
      let x0k = f(f(x[k] - f(sqrtOneMinusAcp * eps[k])) / sqrtAcp);
      x0k = Math.min(x0Clip, Math.max(-x0Clip, x0k));
      // mean = coefX0*x0 + coefX*x
      mean[k] = f(f(coefX0 * x0k) + f(coefX * x[k]));
    }
    if (step > 0) {
      const z = noiseSeq[zi++];
      const sigma = schedule.postSigma[step];
      const nx = new Float32Array(act);
      for (let k = 0; k < act; k++) nx[k] = f(mean[k] + f(sigma * z[k]));
      x = nx;
    } else {
      x = mean; // final step injects no noise (matches diffusion.py's `else 0.0`)
    }
  }
  // Un-standardize then clamp, exactly like diffusion.py's rollout:
  // (sample * act_std + act_mean).clamp(-1, 1).
  const out = new Float32Array(act);
  for (let k = 0; k < act; k++) {
    const denorm = Math.fround(x[k] * actStd[k]) + actMean[k];
    out[k] = Math.min(1, Math.max(-1, Math.fround(denorm)));
  }
  return out;
}
