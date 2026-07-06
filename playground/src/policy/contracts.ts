// v1 tensor contract between exported ONNX policies and the browser runtime.
//
// SINGLE SOURCE OF TRUTH shared with curriculum/common/export_onnx.py — change
// only in a coordinated cross-package PR (see 03-engineering/artifact-pipeline.md
// and playground/CLAUDE.md). playground/scripts/make_toy_onnx.py produces its
// toy model through export_onnx.export_policy, so both producers share one
// stamping path.
//
// Contract v1:
//   input : name "observation", float32, shape [1, obs_dim]
//   output: name "action",      float32, shape [1, act_dim]
//   ONNX ModelProto metadata_props must carry:
//     z2r_contract_version = "v1"
//     z2r_obs_dim, z2r_act_dim  (decimal strings, must match the graph shapes)
// The playground refuses to run a model whose z2r_contract_version does not
// match CONTRACT_VERSION, with a human-readable error naming both versions.

export const CONTRACT_VERSION = 'v1';

export const INPUT_NAME = 'observation';
export const OUTPUT_NAME = 'action';

/** Metadata keys required in the ONNX ModelProto (mirror export_onnx.py). */
export const METADATA_KEYS = {
  contractVersion: 'z2r_contract_version',
  obsDim: 'z2r_obs_dim',
  actDim: 'z2r_act_dim',
} as const;

export interface PolicyContractV1 {
  contractVersion: typeof CONTRACT_VERSION;
  obsDim: number;
  actDim: number;
}

/** Dims for the spike's placeholder PushT scene:
 *  observation = [pusher_x, pusher_y, box_x, box_y]
 *  action      = [force_x, force_y] on the pusher's slide actuators.
 *  The 4-dim toy still exercises the load + contract-gate path; it CANNOT drive
 *  the real PushT env (see PUSHT_POLICY / assertDrivesPushT). */
export const PUSHT_PLACEHOLDER = {
  obsDim: 4,
  actDim: 2,
} as const;

/** The REAL PushT policy contract — the layout a bc.py ONNX carries and the one
 *  the playground drives with. obs[10]/action[2] byte-identical to
 *  pusht_obs.ts (OBS_DIM / ACT_DIM) and curriculum/.../pusht_env.py. A policy
 *  must declare exactly these dims to DRIVE the scene; anything else (e.g. the
 *  4-dim toy) loads and contract-checks fine but is refused at drive time with
 *  a human-readable error naming what PushT needs. */
export const PUSHT_POLICY = {
  obsDim: 10,
  actDim: 2,
} as const;

/** Raised when a contract-valid policy has the wrong dims to DRIVE PushT.
 *  Accepts either a v1 or v2 contract shape (both carry contractVersion/obs/act). */
export class PolicyShapeMismatchError extends Error {
  constructor(found: { contractVersion: string; obsDim: number; actDim: number }) {
    super(
      `This policy is contract-${found.contractVersion} valid, but it declares ` +
      `obs_dim=${found.obsDim}, act_dim=${found.actDim} — driving the PushT scene ` +
      `needs obs_dim=${PUSHT_POLICY.obsDim}, act_dim=${PUSHT_POLICY.actDim} (the ` +
      `pusht_env observation/action layout from pusht_obs.ts). This looks like a ` +
      `different policy (the 4-dim toy, or a non-PushT task). Train a PushT policy ` +
      `with curriculum/phase1_imitation/ch1.1_bc/bc.py and drag its bc_policy.onnx in.`,
    );
    this.name = 'PolicyShapeMismatchError';
  }
}

/** Gate a loaded policy against the PushT drive contract (obs[10]/action[2]).
 *  Throws PolicyShapeMismatchError so the UI can refuse to drive with a
 *  human-readable reason — the toy path stays loadable, driving does not. */
export function assertDrivesPushT(contract: PolicyContractV1): void {
  if (contract.obsDim !== PUSHT_POLICY.obsDim || contract.actDim !== PUSHT_POLICY.actDim) {
    throw new PolicyShapeMismatchError(contract);
  }
}

/** The REAL cartpole (ch2.1 PPO) policy contract — obs[5]/action[1], the layout
 *  a ppo.py ONNX carries and the one the cartpole toy drives with. Byte-identical
 *  to cartpole_obs.ts (OBS_DIM / ACT_DIM) and cartpole_env.py. */
export const CARTPOLE_POLICY = {
  obsDim: 5,
  actDim: 1,
} as const;

/** Gate a loaded policy against the cartpole drive contract (obs[5]/action[1]).
 *  Mirrors assertDrivesPushT: a contract-v1 policy with the wrong dims loads and
 *  contract-checks fine but is refused at drive time with a human-readable
 *  reason (a PushT obs[10] policy cannot drive the 1-DOF cartpole, and vice
 *  versa). Throws PolicyShapeMismatchError so the UI can refuse with the found
 *  dims named. */
export function assertDrivesCartpole(contract: PolicyContractV1): void {
  if (contract.obsDim !== CARTPOLE_POLICY.obsDim || contract.actDim !== CARTPOLE_POLICY.actDim) {
    throw new PolicyShapeMismatchError(contract);
  }
}

export class ContractMismatchError extends Error {
  constructor(found: string | undefined, expected: string) {
    super(
      found === undefined
        ? `This model carries no ${METADATA_KEYS.contractVersion} in its ONNX ` +
          `metadata. The playground requires tensor contract "${expected}". ` +
          `Re-export it with curriculum/common/export_onnx.py.`
        : `This model was exported with tensor contract "${found}", but this ` +
          `playground build expects "${expected}". Re-export the policy with a ` +
          `matching export_onnx.py, or update the playground.`,
    );
    this.name = 'ContractMismatchError';
  }
}

/** Raised when the ONNX graph's declared I/O (dtype or dims) disagrees with the
 *  contract or with the model's own stamped z2r_obs_dim/z2r_act_dim metadata. */
export class GraphContractError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'GraphContractError';
  }
}

/** Minimal, dependency-free view of one tensor's graph metadata, mirroring
 *  onnxruntime-web's InferenceSession.ValueMetadata (name + isTensor, and for
 *  tensors a `type` string and a `shape` of numbers/symbolic-dim strings). Kept
 *  free of any ORT import so the checks below are unit-testable without a
 *  runtime session. */
export interface GraphTensorMeta {
  name: string;
  isTensor: boolean;
  /** ORT dtype string, e.g. "float32", "float64", "int64" (tensors only). */
  type?: string;
  /** Graph-declared shape; numbers are static dims, strings are symbolic. */
  shape?: ReadonlyArray<number | string>;
}

function checkTensorIO(
  role: 'input' | 'output',
  dimLabel: 'obs_dim' | 'act_dim',
  dimKey: string,
  expectedName: string,
  expectedDim: number,
  meta: GraphTensorMeta,
): void {
  if (!meta.isTensor) {
    throw new GraphContractError(
      `Contract ${CONTRACT_VERSION} requires ${role} "${expectedName}" to be a ` +
      `float32 tensor, but the ONNX graph declares it as a non-tensor value. ` +
      `Re-export with curriculum/common/export_onnx.py.`,
    );
  }
  if (meta.type !== 'float32') {
    throw new GraphContractError(
      `Contract ${CONTRACT_VERSION} requires ${role} "${expectedName}" to be ` +
      `float32, but the ONNX graph declares it as ${meta.type ?? 'an unknown type'}. ` +
      `Re-export with curriculum/common/export_onnx.py.`,
    );
  }
  const shape = meta.shape ?? [];
  const shapeStr = `[${shape.join(', ')}]`;
  // Contract v1 is static [1, dim]. The feature dimension must match the
  // model's stamped metadata dim exactly; a symbolic/dynamic batch dim is
  // tolerated but a static batch dim other than 1 is not.
  const featureDim = shape.length === 2 ? shape[1] : undefined;
  if (shape.length !== 2 || typeof featureDim !== 'number' || featureDim !== expectedDim) {
    throw new GraphContractError(
      `policy declares ${dimLabel}=${expectedDim} (${dimKey} metadata) but its ` +
      `ONNX graph ${role} "${expectedName}" is ${shapeStr}, not [1, ${expectedDim}]. ` +
      `The stamped metadata must match the graph shapes — re-export with ` +
      `curriculum/common/export_onnx.py.`,
    );
  }
  const batchDim = shape[0];
  if (typeof batchDim === 'number' && batchDim !== 1) {
    throw new GraphContractError(
      `Contract ${CONTRACT_VERSION} requires ${role} "${expectedName}" batch dim ` +
      `to be 1, but the ONNX graph ${role} is ${shapeStr}.`,
    );
  }
}

/** Validate the ONNX graph's declared input/output dtype and dims against BOTH
 *  contract v1 and the model's stamped z2r_obs_dim/z2r_act_dim (carried in
 *  `contract`). Throws GraphContractError on any mismatch, naming the stamped
 *  dims vs the graph dims. Pure — takes plain metadata, no ORT dependency. */
export function validateGraphContract(
  contract: PolicyContractV1,
  input: GraphTensorMeta,
  output: GraphTensorMeta,
): void {
  checkTensorIO('input', 'obs_dim', METADATA_KEYS.obsDim, INPUT_NAME, contract.obsDim, input);
  checkTensorIO('output', 'act_dim', METADATA_KEYS.actDim, OUTPUT_NAME, contract.actDim, output);
}

/** Validate metadata read from a model file against contract v1. */
export function validateContract(meta: Map<string, string>): PolicyContractV1 {
  const version = meta.get(METADATA_KEYS.contractVersion);
  if (version !== CONTRACT_VERSION) {
    throw new ContractMismatchError(version, CONTRACT_VERSION);
  }
  const obsDim = Number(meta.get(METADATA_KEYS.obsDim));
  const actDim = Number(meta.get(METADATA_KEYS.actDim));
  if (!Number.isInteger(obsDim) || obsDim <= 0 || !Number.isInteger(actDim) || actDim <= 0) {
    throw new Error(
      `Model metadata is missing valid ${METADATA_KEYS.obsDim}/${METADATA_KEYS.actDim} ` +
      `entries (got obs_dim=${meta.get(METADATA_KEYS.obsDim)}, ` +
      `act_dim=${meta.get(METADATA_KEYS.actDim)}).`,
    );
  }
  return { contractVersion: CONTRACT_VERSION, obsDim, actDim };
}

// ===========================================================================
// Contract v2 — the SAMPLER-AWARE runtime (generative policies: ch1.4 diffusion,
// ch1.5 flow). SINGLE SOURCE OF TRUTH shared with curriculum/common/export_onnx.py
// (export_sampler_policy) — change only in a coordinated cross-package PR.
//
// Contract v2:
//   inputs :
//     "point"       float32 [1, act_dim]   the current point x_t (standardized action space)
//     "flow_time"   float32 [1]            the scalar sampler time — for flow the ODE
//                                          time t in [0, 1]; for ddpm the (integer-valued)
//                                          reverse-diffusion step index, carried as a float
//     "observation" float32 [1, obs_dim]   the conditioning obs (obs normalization is
//                                          baked INSIDE the net, as in flow.py/diffusion.py)
//   output:
//     "velocity"    float32 [1, act_dim]   the net output at (point, t | obs): the predicted
//                                          VELOCITY (flow) or NOISE eps (ddpm) — same 3-in/1-out
//                                          core, the sampler that consumes it is what differs
//
//   ONNX ModelProto metadata_props must carry:
//     z2r_contract_version = "v2"
//     z2r_obs_dim, z2r_act_dim   (decimal strings, must match the graph shapes)
//     z2r_sampler   = "flow-euler" | "ddpm" (which sampler the runtime must run)
//     z2r_num_steps = "<int>"               (default net evals per action)
//     z2r_act_mean  = "m0,m1,..."           (per-dim mean to un-standardize the sample)
//     z2r_act_std   = "s0,s1,..."           (per-dim std  to un-standardize the sample)
//   ddpm ONLY additionally carries the reverse-process schedule:
//     z2r_betas     = "b0,b1,..."           (num_steps betas — the forward-noise schedule)
//     z2r_x0_clip   = "<float>"             (the manifold clamp on the predicted clean sample)
//
// The runtime (infer.ts + sampler.ts) draws SEEDED noise, runs num_steps net
// evals through the sampler named in z2r_sampler, then un-standardizes with
// act_mean/act_std — matching flow.py's ode_sample_loop (flow-euler) or
// diffusion.py's p_sample_loop (ddpm) + rollout denorm exactly (proven by the
// JS-vs-Python parity checks). v1 is untouched and keeps working alongside v2.
// ===========================================================================

export const SAMPLER_CONTRACT_VERSION = 'v2';

/** The forward-Euler ODE sampler space (ch1.5 flow) — the integration space is
 *  standardized and schedule-free (x <- x + dt*v). */
export const SAMPLER_SPACE = 'flow-euler';
/** The ancestral DDPM reverse sampler (ch1.4 diffusion) — needs the noise
 *  schedule (betas) and injects fresh noise per reverse step. */
export const SAMPLER_DDPM = 'ddpm';
/** Every sampler this runtime knows how to drive. Any other value fails closed. */
export const SAMPLER_SPACES = [SAMPLER_SPACE, SAMPLER_DDPM] as const;
export type SamplerKind = (typeof SAMPLER_SPACES)[number];

export const POINT_NAME = 'point';
export const TIME_NAME = 'flow_time';
export const SAMPLER_OBS_NAME = 'observation';
export const VELOCITY_NAME = 'velocity';

/** Metadata keys required in a contract-v2 ONNX (mirror export_onnx.py v2).
 *  `betas`/`x0Clip` are present only for the ddpm sampler. */
export const SAMPLER_METADATA_KEYS = {
  contractVersion: 'z2r_contract_version',
  obsDim: 'z2r_obs_dim',
  actDim: 'z2r_act_dim',
  sampler: 'z2r_sampler',
  numSteps: 'z2r_num_steps',
  actMean: 'z2r_act_mean',
  actStd: 'z2r_act_std',
  betas: 'z2r_betas',
  x0Clip: 'z2r_x0_clip',
} as const;

export interface PolicyContractV2 {
  contractVersion: typeof SAMPLER_CONTRACT_VERSION;
  obsDim: number;
  actDim: number;
  sampler: SamplerKind;
  /** Default net evals per sampled action (the runtime may override). */
  numSteps: number;
  /** Per-dim mean/std to un-standardize the integrated sample back to actions. */
  actMean: number[];
  actStd: number[];
  /** ddpm ONLY: the forward-noise schedule (length num_steps) the reverse loop
   *  consumes, and the clamp on the predicted clean sample x0. Absent for flow. */
  betas?: number[];
  x0Clip?: number;
}

/** Parse a comma-separated float list of exactly `dim` finite entries. */
function parseVec(raw: string | undefined, dim: number, key: string): number[] {
  if (raw === undefined) {
    throw new Error(`contract v2 model metadata is missing "${key}".`);
  }
  const parts = raw.split(',').map((s) => Number(s.trim()));
  if (parts.length !== dim || parts.some((v) => !Number.isFinite(v))) {
    throw new Error(
      `contract v2 metadata "${key}"="${raw}" is not ${dim} finite comma-separated ` +
      `floats (the action dimension). Re-export with export_onnx.export_sampler_policy.`,
    );
  }
  return parts;
}

/** Validate metadata read from a model file against contract v2. Throws
 *  ContractMismatchError if the version is not v2 (so a v1 model routed here — or
 *  vice versa — fails closed with both versions named). */
export function validateSamplerContract(meta: Map<string, string>): PolicyContractV2 {
  const version = meta.get(SAMPLER_METADATA_KEYS.contractVersion);
  if (version !== SAMPLER_CONTRACT_VERSION) {
    throw new ContractMismatchError(version, SAMPLER_CONTRACT_VERSION);
  }
  const obsDim = Number(meta.get(SAMPLER_METADATA_KEYS.obsDim));
  const actDim = Number(meta.get(SAMPLER_METADATA_KEYS.actDim));
  if (!Number.isInteger(obsDim) || obsDim <= 0 || !Number.isInteger(actDim) || actDim <= 0) {
    throw new Error(
      `contract v2 model metadata is missing valid ${SAMPLER_METADATA_KEYS.obsDim}/` +
      `${SAMPLER_METADATA_KEYS.actDim} (got obs_dim=${meta.get(SAMPLER_METADATA_KEYS.obsDim)}, ` +
      `act_dim=${meta.get(SAMPLER_METADATA_KEYS.actDim)}).`,
    );
  }
  const sampler = meta.get(SAMPLER_METADATA_KEYS.sampler);
  if (sampler !== SAMPLER_SPACE && sampler !== SAMPLER_DDPM) {
    throw new Error(
      `contract v2 requires ${SAMPLER_METADATA_KEYS.sampler} in ` +
      `[${SAMPLER_SPACES.join(', ')}], found "${sampler}". This runtime knows the ` +
      `forward-Euler ODE (flow) and ancestral DDPM (diffusion) samplers.`,
    );
  }
  const numSteps = Number(meta.get(SAMPLER_METADATA_KEYS.numSteps));
  if (!Number.isInteger(numSteps) || numSteps < 1) {
    throw new Error(
      `contract v2 metadata ${SAMPLER_METADATA_KEYS.numSteps}=` +
      `${meta.get(SAMPLER_METADATA_KEYS.numSteps)} must be a positive integer.`,
    );
  }
  const actMean = parseVec(meta.get(SAMPLER_METADATA_KEYS.actMean), actDim, SAMPLER_METADATA_KEYS.actMean);
  const actStd = parseVec(meta.get(SAMPLER_METADATA_KEYS.actStd), actDim, SAMPLER_METADATA_KEYS.actStd);
  if (actStd.some((v) => !(v > 0))) {
    throw new Error(
      `contract v2 metadata ${SAMPLER_METADATA_KEYS.actStd}="${meta.get(SAMPLER_METADATA_KEYS.actStd)}" ` +
      `has a non-positive std; un-standardization would divide the field by ~0.`,
    );
  }
  const base = {
    contractVersion: SAMPLER_CONTRACT_VERSION as typeof SAMPLER_CONTRACT_VERSION,
    obsDim, actDim, numSteps, actMean, actStd,
  };
  if (sampler === SAMPLER_DDPM) {
    // The ddpm reverse loop needs the forward-noise schedule (betas, one per
    // step) and the x0 clamp; the flow ODE needs neither.
    const betas = parseVec(meta.get(SAMPLER_METADATA_KEYS.betas), numSteps, SAMPLER_METADATA_KEYS.betas);
    if (betas.some((b) => !(b > 0 && b < 1))) {
      throw new Error(
        `contract v2 metadata ${SAMPLER_METADATA_KEYS.betas} must be per-step betas in ` +
        `(0, 1); got out-of-range entries. Re-export with export_onnx.export_sampler_policy.`,
      );
    }
    const x0Clip = Number(meta.get(SAMPLER_METADATA_KEYS.x0Clip));
    if (!(x0Clip > 0)) {
      throw new Error(
        `contract v2 metadata ${SAMPLER_METADATA_KEYS.x0Clip}=` +
        `${meta.get(SAMPLER_METADATA_KEYS.x0Clip)} must be a positive float (the ddpm ` +
        `manifold clamp on the predicted clean sample).`,
      );
    }
    return { ...base, sampler: SAMPLER_DDPM, betas, x0Clip };
  }
  return { ...base, sampler: SAMPLER_SPACE };
}

/** Check one v2 graph tensor against an expected float32 shape. `expected` is a
 *  list where numbers are exact dims and 'B' marks a batch dim (static 1 or a
 *  symbolic/dynamic dim — never a static value other than 1). */
function checkSamplerTensor(
  role: 'input' | 'output',
  name: string,
  expected: ReadonlyArray<number | 'B'>,
  meta: GraphTensorMeta,
): void {
  if (!meta.isTensor || meta.type !== 'float32') {
    throw new GraphContractError(
      `Contract ${SAMPLER_CONTRACT_VERSION} requires ${role} "${name}" to be a float32 ` +
      `tensor, but the ONNX graph declares ${meta.isTensor ? (meta.type ?? 'an unknown type') : 'a non-tensor value'}. ` +
      `Re-export with export_onnx.export_sampler_policy.`,
    );
  }
  const shape = meta.shape ?? [];
  const shapeStr = `[${shape.join(', ')}]`;
  const rankOk = shape.length === expected.length;
  const dimsOk = rankOk && expected.every((e, i) => {
    const got = shape[i];
    if (e === 'B') return typeof got !== 'number' || got === 1; // symbolic batch, or static 1
    return got === e;
  });
  if (!dimsOk) {
    throw new GraphContractError(
      `Contract ${SAMPLER_CONTRACT_VERSION} ${role} "${name}" is ${shapeStr}, ` +
      `not the expected [${expected.map((e) => (e === 'B' ? '1' : e)).join(', ')}]. ` +
      `Re-export with export_onnx.export_sampler_policy.`,
    );
  }
}

/** Validate the v2 ONNX graph's declared I/O against the contract AND the model's
 *  stamped dims. Pure — plain metadata, no ORT dependency (unit-testable). */
export function validateSamplerGraphContract(
  contract: PolicyContractV2,
  point: GraphTensorMeta,
  flowTime: GraphTensorMeta,
  observation: GraphTensorMeta,
  velocity: GraphTensorMeta,
): void {
  checkSamplerTensor('input', POINT_NAME, ['B', contract.actDim], point);
  checkSamplerTensor('input', TIME_NAME, ['B'], flowTime);
  checkSamplerTensor('input', SAMPLER_OBS_NAME, ['B', contract.obsDim], observation);
  checkSamplerTensor('output', VELOCITY_NAME, ['B', contract.actDim], velocity);
}

/** Gate a loaded SAMPLER policy against the PushT drive contract (obs[10]/act[2]).
 *  The v2 mirror of assertDrivesPushT: a contract-v2 flow/diffusion policy with
 *  the wrong dims loads and contract-checks fine but is refused at drive time. */
export function assertSamplerDrivesPushT(contract: PolicyContractV2): void {
  if (contract.obsDim !== PUSHT_POLICY.obsDim || contract.actDim !== PUSHT_POLICY.actDim) {
    throw new PolicyShapeMismatchError({
      contractVersion: contract.contractVersion,
      obsDim: contract.obsDim,
      actDim: contract.actDim,
    });
  }
}
