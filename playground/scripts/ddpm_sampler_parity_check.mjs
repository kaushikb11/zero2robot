// JS-vs-Python DDPM SAMPLER parity — the load-bearing check for the ch1.4
// diffusion contract-v2 extension.
//
// A live diffusion policy that samples DIFFERENTLY from the chapter is a defect.
// This proves the browser's ancestral DDPM reverse sampler (src/policy/sampler.ts's
// ddpmSample + makeDdpmSchedule + cosineBetas, the REAL production code, transpiled)
// reproduces diffusion.py's `make_schedule` + `p_sample_loop` + rollout
// un-standardization bit-for-bit within f32 tol — given the SAME obs, SAME noise
// SEQUENCE, SAME num_steps, evaluating the SAME shipped contract-v2 ONNX denoiser
// on both sides.
//
//   JS  : ddpmSample(noiseSeq, schedule, denoiser=onnxruntime-web, x0_clip, mean, std)
//   ref : the identical reverse loop in .venv python (torch f32 arithmetic +
//         onnxruntime), i.e. diffusion.py's p_sample_loop over the same ONNX.
//
// Transitivity to the chapter: (a) export_diffusion_onnx proves onnx == torch eps
// (~2e-06), (b) this proves JS-ddpm(onnx) == python-ddpm(onnx), (c) the python loop
// here is diffusion.py's make_schedule + p_sample_loop line-for-line => JS == chapter.
//
// The step sweep mirrors the live toy control AND diffusion.py: 100 is the default
// (uses the shipped z2r_betas exactly, as the runtime does); 2 is `--break
// few_steps`; 5 is in between (schedule REBUILT via cosineBetas, as the runtime
// does for non-default counts). Determinism: GaussianRng draws the sequence and it
// is fed to BOTH sides, so per-seed draws are removed from the comparison.
//
// Runs the REAL onnxruntime-web under Node and shells the repo venv for the
// reference (like flow_sampler_parity_check.mjs). Skips with a warning if the v2
// ONNX or the venv is absent (artifacts are provisioned, not committed). Run:
//   node playground/scripts/ddpm_sampler_parity_check.mjs

import { readFileSync, writeFileSync, mkdtempSync, existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve, join } from 'node:path';
import { tmpdir } from 'node:os';
import { execFileSync } from 'node:child_process';
import ts from 'typescript';
import * as ort from 'onnxruntime-web';

const here = dirname(fileURLToPath(import.meta.url));
const srcDir = resolve(here, '..', 'src', 'policy');
const repoRoot = resolve(here, '..', '..');
const outDir = mkdtempSync(join(tmpdir(), 'z2r-ddpm-parity-'));

// The provisioned contract-v2 ddpm denoiser (git-ignored; export_diffusion_onnx.py).
const ONNX = resolve(repoRoot, 'site', 'public', 'models', 'diffusion_denoiser.onnx');
const TOL = 1e-4; // same bar as assert_parity.py / the flow check

async function loadTs(name) {
  const src = readFileSync(join(srcDir, `${name}.ts`), 'utf8');
  const js = ts.transpileModule(src, {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const outPath = join(outDir, `${name}.mjs`);
  writeFileSync(outPath, js);
  return import(pathToFileURL(outPath).href);
}

let passed = 0, failed = 0;
const ok = (n) => { passed += 1; console.log(`  ok   ${n}`); };
const bad = (n, d) => { failed += 1; console.error(`  FAIL ${n}: ${d}`); };

// Z2R_REQUIRE_PARITY=1 (set by the CI lane that provisions the v2 ONNX) turns a
// missing artifact into a hard failure. Locally, absent the git-ignored ONNX, the
// check skips with a warning so `npm test` still runs — but that skip proves
// nothing, so CI must require it. contracts.ts/sampler.ts are hand-mirrors of the
// Python; this cross-check is the only guard against Python<->JS sampler drift.
const REQUIRE_PARITY = process.env.Z2R_REQUIRE_PARITY === '1';
if (!existsSync(ONNX)) {
  const how = `provision it: HF_HUB_OFFLINE=1 HF_TOKEN= .venv/bin/python ` +
    `curriculum/phase1_imitation/ch1.4_diffusion/export_diffusion_onnx.py`;
  if (REQUIRE_PARITY) {
    console.error(`  FAIL (Z2R_REQUIRE_PARITY=1): no contract-v2 ddpm ONNX at ${ONNX} — ${how}`);
    process.exit(1);
  }
  console.warn(`  (skipped: no contract-v2 ddpm ONNX at ${ONNX}\n   ${how})`);
  process.exit(0);
}

const { ddpmSample, makeDdpmSchedule, cosineBetas, GaussianRng } = await loadTs('sampler');
const { readOnnxMetadata } = await loadTs('onnx_metadata');
const { validateSamplerContract } = await loadTs('contracts');

// --- read the shipped contract (sampler/betas/x0_clip/mean/std) from the ONNX ---
const bytes = new Uint8Array(readFileSync(ONNX));
const contract = validateSamplerContract(readOnnxMetadata(bytes));
if (contract.sampler === 'ddpm') {
  ok(`contract-v2 ddpm metadata parses (obs ${contract.obsDim}, act ${contract.actDim}, ` +
     `num_steps ${contract.numSteps}, x0_clip ${contract.x0Clip}, ${contract.betas.length} betas)`);
} else {
  bad('contract sampler', `expected "ddpm", got "${contract.sampler}"`);
}

// The runtime rebuilds cosineBetas(num_steps) for non-default counts; prove it
// reproduces the SHIPPED z2r_betas (so the few-step control is honest).
{
  const rebuilt = cosineBetas(contract.numSteps);
  let m = 0;
  for (let k = 0; k < rebuilt.length; k++) m = Math.max(m, Math.abs(rebuilt[k] - contract.betas[k]));
  // Tol is the f32-cos (torch) vs f64-cos (JS Math) gap on the per-step betas —
  // ~1e-6, and it washes out of the sampled action (proven by the cases below,
  // which pass at 100 steps using the SHIPPED betas and at 2/5 using cosineBetas).
  m < TOL ? ok(`cosineBetas(${contract.numSteps}) ~= shipped z2r_betas (max Δ ${m.toExponential(2)})`)
          : bad('cosineBetas vs shipped betas', `max Δ ${m.toExponential(3)} > ${TOL}`);
}

// --- boot onnxruntime-web under Node (wasm EP; wasmPaths at the package dir) ----
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = resolve(repoRoot, 'playground', 'node_modules', 'onnxruntime-web', 'dist') + '/';
const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
let obsTensorData; // captured by the denoiser closure per case
const denoiser = async (point, step) => {
  const feeds = {
    point: new ort.Tensor('float32', point, [1, contract.actDim]),
    flow_time: new ort.Tensor('float32', new Float32Array([step]), [1]),
    observation: new ort.Tensor('float32', obsTensorData, [1, contract.obsDim]),
  };
  const res = await session.run(feeds);
  return res.velocity.data;
};

// --- fixtures: a few realistic PushT obs, seeded noise sequences, a step sweep --
// obs = [pusher_x, pusher_y, tee_x, tee_y, sin_yaw, cos_yaw, 0, 0, 0, 1]
const OBS = [
  [-0.10, 0.05, 0.12, -0.03, Math.sin(0.4), Math.cos(0.4), 0, 0, 0, 1],
  [0.20, -0.18, -0.09, 0.14, Math.sin(-1.1), Math.cos(-1.1), 0, 0, 0, 1],
  [0.31, -0.28, 0.00, 0.00, Math.sin(3.1), Math.cos(3.1), 0, 0, 0, 1],
].map((o) => Float32Array.from(o));
const STEP_SWEEP = [2, 5, 100]; // --break few_steps (2), an intermediate, the default (100)

// The runtime uses the SHIPPED betas at the default step count and rebuilds the
// cosine schedule otherwise — exercise BOTH paths, exactly like infer.ts.
function scheduleFor(steps) {
  const betas = steps === contract.numSteps
    ? Float32Array.from(contract.betas)
    : cosineBetas(steps);
  return makeDdpmSchedule(betas);
}

// --- JS side: run the real ddpmSample for every (obs, seeded-noise, steps) -------
const cases = [];
for (let oi = 0; oi < OBS.length; oi++) {
  for (const steps of STEP_SWEEP) {
    obsTensorData = OBS[oi];
    // Deterministic noise SEQUENCE: x_T + one z per reverse step (steps draws).
    const rng = new GaussianRng(1000 * oi + steps);
    const noiseSeq = Array.from({ length: steps }, () => rng.fill(new Float32Array(contract.actDim)));
    const action = await ddpmSample(
      noiseSeq, scheduleFor(steps), denoiser, contract.x0Clip, contract.actMean, contract.actStd);
    cases.push({
      obs: [...OBS[oi]], steps,
      noiseSeq: noiseSeq.map((n) => [...n]), js: [...action],
    });
  }
}

// --- Python reference: diffusion.py's make_schedule + p_sample_loop over the ONNX
const py = resolve(repoRoot, '.venv', 'bin', 'python');
if (!existsSync(py)) {
  if (REQUIRE_PARITY) {
    console.error('  FAIL (Z2R_REQUIRE_PARITY=1): .venv python not found — the ' +
      'Python reference cross-check could not run, so parity is unproven.');
    process.exit(1);
  }
  console.warn('  (.venv python not found — JS sampler ran, but the Python reference ' +
    'cross-check was skipped; run in an env with the repo venv to prove parity.)');
  console.log(`\n${failed === 0 ? 'PASS' : 'FAIL'} — ${failed} failure(s).`);
  process.exit(failed === 0 ? 0 : 1);
}

const fixFile = join(outDir, 'cases.json');
writeFileSync(fixFile, JSON.stringify({
  onnx: ONNX, act_mean: contract.actMean, act_std: contract.actStd, x0_clip: contract.x0Clip, cases,
}));
const pyScript = `
import json, sys, math
import numpy as np, torch, onnxruntime as ort
cfg = json.load(open(sys.argv[1]))
sess = ort.InferenceSession(cfg["onnx"], providers=["CPUExecutionProvider"])
act_mean = torch.tensor(cfg["act_mean"], dtype=torch.float32)
act_std = torch.tensor(cfg["act_std"], dtype=torch.float32)
x0_clip = float(cfg["x0_clip"])

def make_schedule(steps):  # diffusion.py.make_schedule cosine (non-broken) branch
    u = torch.linspace(0, steps, steps + 1) / steps
    acp_full = torch.cos((u + 0.008) / 1.008 * math.pi / 2) ** 2
    acp_full = acp_full / acp_full[0].clone()
    betas = (1 - acp_full[1:] / acp_full[:-1]).clamp(1e-8, 0.999)
    alphas = 1.0 - betas
    acp = torch.cumprod(alphas, dim=0)
    acp_prev = torch.cat([torch.ones(1), acp[:-1]])
    post_var = betas * (1.0 - acp_prev) / (1.0 - acp)
    return {"steps": steps, "betas": betas, "alphas": alphas, "acp": acp, "acp_prev": acp_prev,
            "sqrt_acp": acp.sqrt(), "sqrt_one_minus_acp": (1.0 - acp).sqrt(),
            "post_sigma": post_var.clamp_min(1e-20).sqrt()}

out = []
for c in cfg["cases"]:
    obs = np.asarray(c["obs"], dtype=np.float32)[None]
    seq = [torch.tensor(z, dtype=torch.float32) for z in c["noiseSeq"]]
    sch = make_schedule(int(c["steps"]))
    x = seq[0]  # x_T, standardized action space
    zi = 1
    for step in reversed(range(sch["steps"])):        # diffusion.py p_sample_loop
        eps = sess.run(None, {"point": x.numpy()[None], "flow_time": np.array([float(step)], np.float32),
                              "observation": obs})[0][0]
        eps = torch.from_numpy(eps)
        acp, acp_prev = sch["acp"][step], sch["acp_prev"][step]
        x0 = ((x - sch["sqrt_one_minus_acp"][step] * eps) / sch["sqrt_acp"][step]).clamp(-x0_clip, x0_clip)
        mean = (sch["betas"][step] * acp_prev.sqrt() / (1.0 - acp) * x0
                + (1.0 - acp_prev) * sch["alphas"][step].sqrt() / (1.0 - acp) * x)
        if step > 0:
            x = mean + sch["post_sigma"][step] * seq[zi]; zi += 1
        else:
            x = mean
    action = (x * act_std + act_mean).clamp(-1.0, 1.0)  # diffusion.py rollout denorm
    out.append([float(a) for a in action])
print(json.dumps(out))
`;
let refActions;
try {
  const stdout = execFileSync(py, ['-c', pyScript, fixFile], { encoding: 'utf8', env: { ...process.env, HF_HUB_OFFLINE: '1' } });
  refActions = JSON.parse(stdout.trim().split('\n').pop());
} catch (err) {
  bad('python reference', err.message.split('\n').slice(-3).join(' | '));
  console.log(`\n${failed === 0 ? 'PASS' : 'FAIL'} — ${failed} failure(s).`);
  process.exit(1);
}

// --- compare JS vs Python for every case ---------------------------------------
let worst = 0;
cases.forEach((c, i) => {
  const ref = refActions[i];
  let maxErr = 0;
  for (let k = 0; k < c.js.length; k++) maxErr = Math.max(maxErr, Math.abs(c.js[k] - ref[k]));
  worst = Math.max(worst, maxErr);
  const label = `obs#${Math.floor(i / STEP_SWEEP.length)} steps=${c.steps}`;
  if (maxErr <= TOL) ok(`${label}: JS action == diffusion.py sampler (max Δ ${maxErr.toExponential(2)})`);
  else bad(label, `max Δ ${maxErr.toExponential(3)} > ${TOL}\n    js =${c.js}\n    py =${ref}`);
});

// --- determinism: seeded browser noise replays; a full sample is reproducible ---
{
  const a = new GaussianRng(7).fill(new Float32Array(2));
  const b = new GaussianRng(7).fill(new Float32Array(2));
  const same = a.every((v, i) => v === b[i]);
  same ? ok('GaussianRng(seed) is reproducible (same seed -> identical noise)')
       : bad('GaussianRng determinism', `${[...a]} != ${[...b]}`);

  obsTensorData = OBS[0];
  const draw = (seed, steps) => {
    const r = new GaussianRng(seed);
    return Array.from({ length: steps }, () => r.fill(new Float32Array(2)));
  };
  const s1 = await ddpmSample(draw(42, 20), scheduleFor(20), denoiser, contract.x0Clip, contract.actMean, contract.actStd);
  const s2 = await ddpmSample(draw(42, 20), scheduleFor(20), denoiser, contract.x0Clip, contract.actMean, contract.actStd);
  const same2 = [...s1].every((v, i) => v === s2[i]);
  same2 ? ok('sampleAction is reproducible for a fixed seed (browser rollout replays)')
        : bad('sampleAction determinism', `${[...s1]} != ${[...s2]}`);
}

console.log(`\nworst JS-vs-Python action Δ = ${worst.toExponential(2)} (tol ${TOL})`);
console.log(`${failed === 0 ? 'PASS' : 'FAIL'} — ${passed} ok, ${failed} failure(s).`);
process.exit(failed === 0 ? 0 : 1);
