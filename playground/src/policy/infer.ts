// ONNX policy inference via onnxruntime-web (wasm execution provider,
// single-threaded — no COOP/COEP requirement, matching the mujoco build).
// The /wasm subpath is the wasm-EP-only "bundle" build: the Emscripten loader
// .mjs is embedded, so only the .wasm binary needs an explicit URL.
import * as ort from 'onnxruntime-web/wasm';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import {
  INPUT_NAME,
  OUTPUT_NAME,
  POINT_NAME,
  TIME_NAME,
  SAMPLER_OBS_NAME,
  VELOCITY_NAME,
  GraphContractError,
  validateContract,
  validateGraphContract,
  validateSamplerContract,
  validateSamplerGraphContract,
  type GraphTensorMeta,
  type PolicyContractV1,
  type PolicyContractV2,
} from './contracts';
import { SAMPLER_DDPM } from './contracts';
import { readOnnxMetadata } from './onnx_metadata';
import {
  GaussianRng,
  eulerSample,
  ddpmSample,
  makeDdpmSchedule,
  cosineBetas,
  type VelocityFn,
} from './sampler';

/** Project an ORT session I/O metadata array onto our dependency-free
 *  GraphTensorMeta, picking the entry named `name`. */
function graphTensorMeta(
  list: readonly ort.InferenceSession.ValueMetadata[],
  name: string,
): GraphTensorMeta {
  const m = list.find((e) => e.name === name);
  if (!m) throw new Error(`ONNX graph exposes no tensor named "${name}"`);
  return m.isTensor
    ? { name: m.name, isTensor: true, type: m.type, shape: m.shape }
    : { name: m.name, isTensor: false };
}

// Hand ort the bundled runtime assets explicitly (its own import.meta.url
// resolution does not survive Vite bundling).
ort.env.wasm.wasmPaths = { wasm: ortWasmUrl };
ort.env.wasm.numThreads = 1;

export interface Policy {
  contract: PolicyContractV1;
  /** Run one forward pass; obs length must equal contract.obsDim. */
  act(obs: Float32Array): Promise<Float32Array>;
  /** Mean latency in ms over all act() calls so far. */
  meanLatencyMs(): number;
  readonly calls: number;
}

export async function loadPolicy(url: string): Promise<Policy> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`failed to fetch model ${url}: HTTP ${resp.status}`);
  return loadPolicyFromBytes(new Uint8Array(await resp.arrayBuffer()));
}

/** Load a policy directly from ONNX bytes — the path used when a learner drags
 *  or file-picks their own bc_policy.onnx (no URL to fetch). Runs the identical
 *  fail-closed contract gate as loadPolicy before the runtime sees the model. */
export async function loadPolicyFromBytes(bytes: Uint8Array): Promise<Policy> {
  // Contract gate BEFORE the runtime sees the model.
  const contract = validateContract(readOnnxMetadata(bytes));

  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
  });

  if (!session.inputNames.includes(INPUT_NAME) || !session.outputNames.includes(OUTPUT_NAME)) {
    throw new Error(
      `Model I/O names do not match contract v${contract.contractVersion}: ` +
      `expected input "${INPUT_NAME}" and output "${OUTPUT_NAME}", ` +
      `got inputs [${session.inputNames}] outputs [${session.outputNames}].`,
    );
  }

  // Graph gate: the ONNX graph's declared input/output dtype and dims must
  // agree with contract v1 AND with the model's own stamped z2r_obs_dim/
  // z2r_act_dim. Fails closed here, at LOAD time, with a human-readable error —
  // rather than surfacing a raw ORT internal error on the first act(), or (for
  // a non-float32 output) silently mis-casting the result tensor below.
  validateGraphContract(
    contract,
    graphTensorMeta(session.inputMetadata, INPUT_NAME),
    graphTensorMeta(session.outputMetadata, OUTPUT_NAME),
  );

  let totalMs = 0;
  let calls = 0;

  return {
    contract,
    get calls() {
      return calls;
    },
    async act(obs: Float32Array): Promise<Float32Array> {
      if (obs.length !== contract.obsDim) {
        throw new Error(`observation length ${obs.length} != obs_dim ${contract.obsDim}`);
      }
      const t0 = performance.now();
      const feeds = {
        [INPUT_NAME]: new ort.Tensor('float32', obs, [1, contract.obsDim]),
      };
      const results = await session.run(feeds);
      totalMs += performance.now() - t0;
      calls += 1;
      const action = results[OUTPUT_NAME];
      if (action.dims.length !== 2 || action.dims[0] !== 1 || action.dims[1] !== contract.actDim) {
        throw new Error(
          `action shape [${action.dims}] != contract [1, ${contract.actDim}]`,
        );
      }
      // Defense in depth: the load-time graph gate already rejects non-float32
      // outputs, but never let a non-float32 tensor reach the `as Float32Array`
      // cast — a float64/int64 buffer would masquerade as Float32Array (and an
      // int64 tensor's data is BigInt64Array, which crashes the caller's
      // toFixed()). Guard the cast with a human-readable contract error.
      if (action.type !== 'float32') {
        throw new GraphContractError(
          `Contract ${contract.contractVersion} requires output "${OUTPUT_NAME}" ` +
          `to be float32, but this run produced a ${action.type} tensor.`,
        );
      }
      return action.data as Float32Array;
    },
    meanLatencyMs: () => (calls > 0 ? totalMs / calls : 0),
  };
}

// ===========================================================================
// Contract v2 — the SAMPLER-AWARE runtime (generative policies). Additive: v1's
// loadPolicy/Policy above is untouched. A SamplerPolicy owns the iterative Euler
// ODE loop (sampler.ts) around the velocity net's ONNX session, so the browser
// can DRIVE a flow (ch1.5) or diffusion (ch1.4) policy that v1 cannot express.
// ===========================================================================

/** Options for a single sampled action. */
export interface SampleOptions {
  /** Override the contract's default num_steps (the flow_steps / denoising_steps
   *  live control). For ddpm this rebuilds the whole schedule for that step count
   *  (exactly as `diffusion.py --denoising_steps n` does). */
  numSteps?: number;
  /** flow ONLY: supply an explicit standardized-space noise init instead of
   *  drawing from the policy's seeded RNG. Used by the parity check to feed
   *  identical noise to both samplers; length must equal act_dim. */
  noise?: Float32Array;
  /** ddpm ONLY: supply the explicit noise SEQUENCE (length num_steps: x_T then one
   *  z per reverse step) instead of drawing from the seeded RNG. Used by the
   *  parity check; each entry length must equal act_dim. */
  noiseSeq?: Float32Array[];
}

export interface SamplerPolicy {
  contract: PolicyContractV2;
  /** Sample ONE action for `obs`: seeded noise -> num_steps net evals through the
   *  contract's sampler -> un-standardize. Mirrors flow.py's ode_sample_loop
   *  (flow-euler) or diffusion.py's p_sample_loop (ddpm). */
  sampleAction(obs: Float32Array, opts?: SampleOptions): Promise<Float32Array>;
  /** Reseed the noise stream so a rollout replays deterministically (flow.py
   *  seeds its sampler from the episode seed at reset; call this on env.reset). */
  seedNoise(seed: number): void;
  /** Mean latency in ms per sampleAction() (a full multi-step sample). */
  meanLatencyMs(): number;
  readonly calls: number;
}

export async function loadSamplerPolicy(url: string, noiseSeed = 0): Promise<SamplerPolicy> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`failed to fetch model ${url}: HTTP ${resp.status}`);
  return loadSamplerPolicyFromBytes(new Uint8Array(await resp.arrayBuffer()), noiseSeed);
}

/** Load a sampler (contract-v2) policy from ONNX bytes. Runs the identical
 *  fail-closed contract + graph gates as the v1 path before the runtime sees the
 *  model, then wraps the velocity net's session in the Euler ODE sampler. */
export async function loadSamplerPolicyFromBytes(
  bytes: Uint8Array,
  noiseSeed = 0,
): Promise<SamplerPolicy> {
  // Contract gate BEFORE the runtime sees the model.
  const contract = validateSamplerContract(readOnnxMetadata(bytes));

  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
  });

  const need = [POINT_NAME, TIME_NAME, SAMPLER_OBS_NAME];
  if (!need.every((n) => session.inputNames.includes(n)) || !session.outputNames.includes(VELOCITY_NAME)) {
    throw new Error(
      `Model I/O names do not match contract v2: expected inputs [${need}] and ` +
      `output "${VELOCITY_NAME}", got inputs [${session.inputNames}] outputs [${session.outputNames}].`,
    );
  }

  // Graph gate: fail closed at LOAD time (before the first sample) if the graph's
  // declared dtypes/dims disagree with the contract or the stamped dims.
  validateSamplerGraphContract(
    contract,
    graphTensorMeta(session.inputMetadata, POINT_NAME),
    graphTensorMeta(session.inputMetadata, TIME_NAME),
    graphTensorMeta(session.inputMetadata, SAMPLER_OBS_NAME),
    graphTensorMeta(session.outputMetadata, VELOCITY_NAME),
  );

  const rng = new GaussianRng(noiseSeed);
  let totalMs = 0;
  let calls = 0;

  return {
    contract,
    get calls() {
      return calls;
    },
    seedNoise(seed: number) {
      rng.reseed(seed);
    },
    async sampleAction(obs: Float32Array, opts?: SampleOptions): Promise<Float32Array> {
      if (obs.length !== contract.obsDim) {
        throw new Error(`observation length ${obs.length} != obs_dim ${contract.obsDim}`);
      }
      const numSteps = opts?.numSteps ?? contract.numSteps;
      if (!Number.isInteger(numSteps) || numSteps < 1) {
        throw new Error(`num_steps must be a positive integer, got ${numSteps}`);
      }
      // The net conditioned on THIS obs; the sampler chains it num_steps times.
      // The obs tensor is built once and reused across the inner steps. The
      // closure serves both roles: `t` is the flow ODE time (flow-euler) or the
      // integer reverse-diffusion step index (ddpm), carried to the ONNX net's
      // float32 "flow_time" input either way.
      const obsTensor = new ort.Tensor('float32', obs, [1, contract.obsDim]);
      const net: VelocityFn = async (point, t) => {
        const feeds = {
          [POINT_NAME]: new ort.Tensor('float32', point, [1, contract.actDim]),
          [TIME_NAME]: new ort.Tensor('float32', new Float32Array([t]), [1]),
          [SAMPLER_OBS_NAME]: obsTensor,
        };
        const res = await session.run(feeds);
        const v = res[VELOCITY_NAME];
        if (v.type !== 'float32') {
          throw new GraphContractError(
            `Contract v2 requires output "${VELOCITY_NAME}" to be float32, ` +
            `but this run produced a ${v.type} tensor.`,
          );
        }
        return v.data as Float32Array;
      };

      const t0 = performance.now();
      let action: Float32Array;
      if (contract.sampler === SAMPLER_DDPM) {
        // Rebuild the DDPM schedule for THIS step count (changing steps rebuilds
        // the whole schedule, exactly like diffusion.py --denoising_steps). The
        // shipped betas is used bit-exact at the default step count; other counts
        // re-derive the cosine schedule the export was built from.
        const betas = numSteps === contract.numSteps && contract.betas
          ? Float32Array.from(contract.betas)
          : cosineBetas(numSteps);
        const schedule = makeDdpmSchedule(betas);
        // Noise SEQUENCE: x_T + one z per reverse step>0 (numSteps draws total).
        const noiseSeq = opts?.noiseSeq
          ?? Array.from({ length: numSteps }, () => rng.fill(new Float32Array(contract.actDim)));
        if (noiseSeq.length !== numSteps || noiseSeq.some((n) => n.length !== contract.actDim)) {
          throw new Error(
            `ddpm noiseSeq must be ${numSteps} draws of length ${contract.actDim}`,
          );
        }
        action = await ddpmSample(
          noiseSeq, schedule, net, contract.x0Clip!, contract.actMean, contract.actStd,
        );
      } else {
        const noise = opts?.noise ?? rng.fill(new Float32Array(contract.actDim));
        if (noise.length !== contract.actDim) {
          throw new Error(`noise length ${noise.length} != act_dim ${contract.actDim}`);
        }
        action = await eulerSample(noise, numSteps, net, contract.actMean, contract.actStd);
      }
      totalMs += performance.now() - t0;
      calls += 1;
      return action;
    },
    meanLatencyMs: () => (calls > 0 ? totalMs / calls : 0),
  };
}
