// JS-vs-Python SAMPLER parity — the load-bearing check for contract v2.
//
// A live generative policy that samples DIFFERENTLY from the chapter is a defect.
// This proves the browser's forward-Euler ODE sampler (src/policy/sampler.ts's
// eulerSample, the REAL production code, transpiled) reproduces flow.py's
// `ode_sample_loop` + rollout un-standardization bit-for-bit within f32 tol —
// given the SAME obs, SAME noise, SAME num_steps, evaluating the SAME shipped
// contract-v2 ONNX velocity net on both sides.
//
//   JS  : eulerSample(noise, steps, velocity=onnxruntime-web, act_mean, act_std)
//   ref : the identical Euler loop in .venv python (torch f32 arithmetic +
//         onnxruntime), i.e. flow.py's sampler math over the same ONNX.
//
// Transitivity to the chapter: (a) export_flow_onnx proves onnx == torch velocity
// (7.6e-06), (b) this proves JS-euler(onnx) == python-euler(onnx), (c) the python
// euler here is flow.py's ode_sample_loop line-for-line => JS == flow.py.
//
// Determinism: GaussianRng (the seeded browser noise) is checked reproducible
// (same seed -> identical draws). It is NOT torch's RNG (see sampler.ts) — so
// this check feeds BOTH sides explicit noise; per-seed draws are a separate axis.
//
// Runs the REAL onnxruntime-web (already a dep) under Node, and shells the repo
// venv for the reference (like obs_parity_check.mjs). Skips with a warning if the
// v2 ONNX or the venv is absent (artifacts are provisioned, not committed). Run:
//   node playground/scripts/flow_sampler_parity_check.mjs

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
const outDir = mkdtempSync(join(tmpdir(), 'z2r-flow-parity-'));

// The provisioned contract-v2 velocity net (git-ignored; export_flow_onnx.py).
const ONNX = resolve(repoRoot, 'site', 'public', 'models', 'flow_velocity.onnx');
const TOL = 1e-4; // same bar as assert_parity.py

// --- transpile the PURE runtime modules (no ORT / no Vite `?url` imports) ------
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
// nothing, so CI must require it. contracts.ts is a hand-mirror of export_onnx.py;
// this cross-check is the only guard against Python<->JS sampler drift.
const REQUIRE_PARITY = process.env.Z2R_REQUIRE_PARITY === '1';
if (!existsSync(ONNX)) {
  const how = `provision it: HF_HUB_OFFLINE=1 HF_TOKEN= .venv/bin/python ` +
    `curriculum/phase1_imitation/ch1.5_flow/export_flow_onnx.py`;
  if (REQUIRE_PARITY) {
    console.error(`  FAIL (Z2R_REQUIRE_PARITY=1): no contract-v2 ONNX at ${ONNX} — ${how}`);
    process.exit(1);
  }
  console.warn(`  (skipped: no contract-v2 ONNX at ${ONNX}\n   ${how})`);
  process.exit(0);
}

const { eulerSample, GaussianRng } = await loadTs('sampler');
const { readOnnxMetadata } = await loadTs('onnx_metadata');
const { validateSamplerContract } = await loadTs('contracts');

// --- read the shipped contract (act_mean/act_std/num_steps) from the ONNX ------
const bytes = new Uint8Array(readFileSync(ONNX));
const contract = validateSamplerContract(readOnnxMetadata(bytes));
ok(`contract-v2 metadata parses (obs ${contract.obsDim}, act ${contract.actDim}, ` +
   `num_steps ${contract.numSteps})`);

// --- boot onnxruntime-web under Node (wasm EP; wasmPaths at the package dir) ----
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = resolve(repoRoot, 'playground', 'node_modules', 'onnxruntime-web', 'dist') + '/';
const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
const velocity = async (point, t) => {
  const feeds = {
    point: new ort.Tensor('float32', point, [1, contract.actDim]),
    flow_time: new ort.Tensor('float32', new Float32Array([t]), [1]),
    observation: new ort.Tensor('float32', obsTensorData, [1, contract.obsDim]),
  };
  const res = await session.run(feeds);
  return res.velocity.data;
};

// --- fixtures: a few realistic PushT obs, explicit noise, a step sweep ---------
// obs = [pusher_x, pusher_y, tee_x, tee_y, sin_yaw, cos_yaw, 0, 0, 0, 1]
const OBS = [
  [-0.10, 0.05, 0.12, -0.03, Math.sin(0.4), Math.cos(0.4), 0, 0, 0, 1],
  [0.20, -0.18, -0.09, 0.14, Math.sin(-1.1), Math.cos(-1.1), 0, 0, 0, 1],
  [0.31, -0.28, 0.00, 0.00, Math.sin(3.1), Math.cos(3.1), 0, 0, 0, 1],
].map((o) => Float32Array.from(o));
const NOISE = [
  [0.5409961, -0.2934289],
  [-2.1787894, 0.5684313],
  [0.3922968, -1.2050371],
].map((n) => Float32Array.from(n));
const STEP_SWEEP = [2, 5, 100]; // few-step (the live control) through the default

let obsTensorData; // captured by the velocity closure per case

// --- JS side: run the real eulerSample for every (obs, noise, steps) -----------
const cases = [];
for (let oi = 0; oi < OBS.length; oi++) {
  for (const steps of STEP_SWEEP) {
    obsTensorData = OBS[oi];
    const action = await eulerSample(NOISE[oi], steps, velocity, contract.actMean, contract.actStd);
    cases.push({ obs: [...OBS[oi]], noise: [...NOISE[oi]], steps, js: [...action] });
  }
}

// --- Python reference: flow.py's ode_sample_loop math over the SAME ONNX -------
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
writeFileSync(fixFile, JSON.stringify({ onnx: ONNX, act_mean: contract.actMean, act_std: contract.actStd, cases }));
const pyScript = `
import json, sys
import numpy as np, torch, onnxruntime as ort
cfg = json.load(open(sys.argv[1]))
sess = ort.InferenceSession(cfg["onnx"], providers=["CPUExecutionProvider"])
act_mean = torch.tensor(cfg["act_mean"], dtype=torch.float32)
act_std = torch.tensor(cfg["act_std"], dtype=torch.float32)
out = []
for c in cfg["cases"]:
    obs = np.asarray(c["obs"], dtype=np.float32)[None]
    x = torch.tensor(c["noise"], dtype=torch.float32)   # standardized action space
    steps = int(c["steps"]); dt = 1.0 / steps
    for i in range(steps):                              # flow.py ode_sample_loop
        t = np.float32(i * dt)
        v = sess.run(None, {"point": x.numpy()[None], "flow_time": np.array([t], np.float32),
                            "observation": obs})[0][0]
        x = x + dt * torch.from_numpy(v)                # forward Euler, torch f32
    action = (x * act_std + act_mean).clamp(-1.0, 1.0)  # flow.py rollout denorm
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
  if (maxErr <= TOL) ok(`${label}: JS action == flow.py sampler (max Δ ${maxErr.toExponential(2)})`);
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
  const seed = 42;
  const s1 = await eulerSample(new GaussianRng(seed).fill(new Float32Array(2)), 20, velocity, contract.actMean, contract.actStd);
  const s2 = await eulerSample(new GaussianRng(seed).fill(new Float32Array(2)), 20, velocity, contract.actMean, contract.actStd);
  const same2 = [...s1].every((v, i) => v === s2[i]);
  same2 ? ok('sampleAction is reproducible for a fixed seed (browser rollout replays)')
        : bad('sampleAction determinism', `${[...s1]} != ${[...s2]}`);
}

console.log(`\nworst JS-vs-Python action Δ = ${worst.toExponential(2)} (tol ${TOL})`);
console.log(`${failed === 0 ? 'PASS' : 'FAIL'} — ${passed} ok, ${failed} failure(s).`);
process.exit(failed === 0 ? 0 : 1);
