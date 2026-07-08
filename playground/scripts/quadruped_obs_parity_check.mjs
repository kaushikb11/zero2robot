// Obs-parity self-check for the QUADRUPED: the browser's obs builder produces
// byte-equal observations to curriculum/common/envs/quadruped/quadruped_env.py for
// the SAME state. Unlike the pusher/cartpole checks (pure functions), the quadruped
// obs reads the free-joint torso height/linear-velocity AND the torso UP-VECTOR
// from the body world rotation matrix (xmat) — so this check RUNS the real
// MuJoCo-WASM binding (the same @mujoco/mujoco the browser uses), builds obs through
// src/teleop/quadruped_obs.ts's buildObs(sim) over a tiny Sim shim, and compares to
// quadruped_env._obs() computed live via .venv python. Run:
//   node scripts/quadruped_obs_parity_check.mjs
//
// TWO checks, both against the SAME python quadruped_env:
//   1. STATIC states — set an explicit qpos/qvel (incl. a tilted torso) and compare
//      obs field-for-field. Verifies the obs BUILDER (xmat up-vector + free-joint
//      slots).
//   2. CONTACT ROLLOUT — from one fixed upright start, drive an IDENTICAL action
//      sequence through the residual-position action mapping the BROWSER env uses
//      (ctrl = DEFAULT_POSE + ACTION_SCALE*clip(a), held FRAME_SKIP physics steps —
//      the exact quadruped_env.ts constants, imported here) and compare obs at EVERY
//      control step against quadruped_env.step. This is the correctness gate the
//      policy demos depend on: browser obs == training obs over a real ground-contact
//      rollout, or a walk/rewards/DR policy is fed garbage.
//
// The QUADRUPED_XML the WASM loads is extracted from src/sim/scene.ts, so this ALSO
// verifies the vendored scene string still matches the obs the training env emits.

import { readFileSync, writeFileSync, mkdtempSync, existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve, join } from 'node:path';
import { tmpdir } from 'node:os';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import ts from 'typescript';

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));
const srcDir = resolve(here, '..', 'src');
const repoRoot = resolve(here, '..', '..');
const outDir = mkdtempSync(join(tmpdir(), 'z2r-quad-obs-'));

// transpile-and-import a standalone TS module (only `import type` deps).
async function importTs(relPath, exportPick) {
  const srcText = readFileSync(join(srcDir, relPath), 'utf8');
  const js = ts.transpileModule(srcText, {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const p = join(outDir, relPath.replace(/[/\\]/g, '_').replace(/\.ts$/, '.mjs'));
  writeFileSync(p, js);
  const mod = await import(pathToFileURL(p).href);
  return exportPick ? mod[exportPick] : mod;
}

const { QUADRUPED_XML } = await importTs('sim/scene.ts');
// buildObs + the action-mapping constants the BROWSER env (quadruped_env.ts) uses.
// Importing them from quadruped_obs.ts is what pins the rollout to the shipped
// contract: if DEFAULT_POSE / ACTION_SCALE / FRAME_SKIP ever drift from the python
// env, the rollout diverges and this check fails.
const { buildObs, JOINT_NAMES, DEFAULT_POSE, ACTION_SCALE, FRAME_SKIP } =
  await importTs('teleop/quadruped_obs.ts');

// --- load the SAME MuJoCo-WASM binding the browser uses --------------------------
const mjPath = require.resolve('@mujoco/mujoco');
const wasmPath = mjPath.replace(/[^/]+$/, 'mujoco.wasm');
const loadMujoco = (await import('@mujoco/mujoco')).default;
const M = await loadMujoco({ locateFile: (p) => (p.endsWith('.wasm') ? wasmPath : p) });
const model = M.MjModel.from_xml_string(QUADRUPED_XML);
const data = new M.MjData(model);

const jointId = (name) => M.mj_name2id(model, M.mjtObj.mjOBJ_JOINT.value, name);
const bodyId = (name) => M.mj_name2id(model, M.mjtObj.mjOBJ_BODY.value, name);
// Minimal Sim shim: exactly the methods quadruped_obs.buildObs touches.
const sim = {
  jointQpos: (n) => data.qpos[model.jnt_qposadr[jointId(n)]],
  jointQvel: (n) => data.qvel[model.jnt_dofadr[jointId(n)]],
  jointQposAdr: (n) => model.jnt_qposadr[jointId(n)],
  jointDofAdr: (n) => model.jnt_dofadr[jointId(n)],
  qposAt: (i) => data.qpos[i],
  qvelAt: (i) => data.qvel[i],
  bodyXmat: (n) => { const a = bodyId(n) * 9; const o = new Array(9); for (let k = 0; k < 9; k++) o[k] = data.xmat[a + k]; return o; },
};

const rootQ = model.jnt_qposadr[jointId('root')];
const rootV = model.jnt_dofadr[jointId('root')];

// Known states: free-joint pose [x,y,z, qw,qx,qy,qz], 8 leg angles, root linvel
// [vx,vy,vz], and 8 leg vels. State 2 tilts the torso about +y (a non-identity
// quaternion) to exercise the up-vector read from xmat.
function quatY(theta) { return [Math.cos(theta / 2), 0, Math.sin(theta / 2), 0]; }
const STATES = [
  { pos: [0, 0, 0.257], quat: [1, 0, 0, 0], legs: [0.6, -1.2, 0.6, -1.2, 0.6, -1.2, 0.6, -1.2],
    linvel: [0, 0, 0], legvel: [0, 0, 0, 0, 0, 0, 0, 0] },
  { pos: [0.3, -0.1, 0.28], quat: [1, 0, 0, 0], legs: [0.7, -1.0, 0.5, -1.3, 0.55, -1.25, 0.65, -1.15],
    linvel: [0.4, -0.05, 0.02], legvel: [0.2, -0.3, 0.1, 0.4, -0.2, 0.15, 0.3, -0.1] },
  { pos: [-0.2, 0.05, 0.24], quat: quatY(0.35), legs: [0.4, -1.4, 0.8, -0.9, 0.6, -1.2, 0.5, -1.3],
    linvel: [-0.2, 0.1, -0.15], legvel: [-0.4, 0.2, 0.3, -0.1, 0.05, -0.25, 0.4, 0.2] },
];

function setState(s) {
  M.mj_resetData(model, data);
  const q = data.qpos, v = data.qvel;
  q[rootQ + 0] = s.pos[0]; q[rootQ + 1] = s.pos[1]; q[rootQ + 2] = s.pos[2];
  q[rootQ + 3] = s.quat[0]; q[rootQ + 4] = s.quat[1]; q[rootQ + 5] = s.quat[2]; q[rootQ + 6] = s.quat[3];
  JOINT_NAMES.forEach((n, i) => { q[model.jnt_qposadr[jointId(n)]] = s.legs[i]; });
  v[rootV + 0] = s.linvel[0]; v[rootV + 1] = s.linvel[1]; v[rootV + 2] = s.linvel[2];
  JOINT_NAMES.forEach((n, i) => { v[model.jnt_dofadr[jointId(n)]] = s.legvel[i]; });
  M.mj_forward(model, data);
}

// --- CONTACT ROLLOUT: one fixed upright start + a deterministic action sequence.
// A mild diagonal-trot-shaped action drives the feet into the floor so the rollout
// exercises real contact dynamics (the whole reason the quadruped needs a WASM
// parity check, not a pure-function one). Both sims see the IDENTICAL start + actions.
const ROLLOUT_START = {
  pos: [0, 0, 0.257], quat: [1, 0, 0, 0],
  legs: [...DEFAULT_POSE], linvel: [0, 0, 0], legvel: [0, 0, 0, 0, 0, 0, 0, 0],
};
const ROLLOUT_STEPS = 60;
function rolloutActions() {
  // leg order FL, FR, HL, HR; each (hip, knee). Diagonal pairs {FL,HR} vs {FR,HL}.
  const legPhase = [0, Math.PI, Math.PI, 0]; // FL, FR, HL, HR
  const acts = [];
  for (let t = 0; t < ROLLOUT_STEPS; t++) {
    const phase = 2 * Math.PI * 2.5 * t * (FRAME_SKIP * 0.005); // 2.5 Hz gait
    const a = new Array(8);
    for (let leg = 0; leg < 4; leg++) {
      const ph = phase + legPhase[leg];
      a[leg * 2 + 0] = Math.max(-1, Math.min(1, -0.5 * Math.sin(ph)));       // hip sweep
      a[leg * 2 + 1] = Math.max(-1, Math.min(1, -0.9 * Math.max(0, Math.sin(ph)))); // knee flex on swing
    }
    acts.push(a);
  }
  return acts;
}
const ROLLOUT_ACTIONS = rolloutActions();

function setRolloutStart() {
  setState(ROLLOUT_START);
}
// Replicate quadruped_env.step's physics with the SHIPPED browser constants.
function stepBrowserPhysics(action) {
  for (let i = 0; i < 8; i++) {
    const a = Math.max(-1, Math.min(1, action[i]));
    data.ctrl[i] = DEFAULT_POSE[i] + ACTION_SCALE * a;
  }
  for (let s = 0; s < FRAME_SKIP; s++) M.mj_step(model, data);
}

const TOL = 1e-5;
let failed = 0;
const ok = (n) => console.log(`  ok   ${n}`);
const bad = (n, d) => { failed += 1; console.error(`  FAIL ${n}: ${d}`); };

// python reference: quadruped_env._obs() for the same qpos/qvel, plus the rollout
// obs trajectory from quadruped_env.step over the same start + actions.
const py = resolve(repoRoot, '.venv', 'bin', 'python');
let reference = null;
if (existsSync(py)) {
  const specFile = join(outDir, 'spec.json');
  writeFileSync(specFile, JSON.stringify({
    states: STATES,
    rollout_start: ROLLOUT_START,
    rollout_actions: ROLLOUT_ACTIONS,
  }));
  const pyScript = `
import sys, json
sys.path.insert(0, ${JSON.stringify(join(repoRoot, 'curriculum', 'common', 'envs', 'quadruped'))})
import numpy as np, mujoco
from quadruped_env import QuadrupedEnv, JOINT_NAMES
env = QuadrupedEnv()
rq = env._root_qadr; rv = env._root_vadr
spec = json.load(open(sys.argv[1]))

def set_state(s):
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[rq:rq+3] = s["pos"]
    env.data.qpos[rq+3:rq+7] = s["quat"]
    for i, n in enumerate(JOINT_NAMES):
        env.data.qpos[env._jadr[i]] = s["legs"][i]
        env.data.qvel[env._vadr[i]] = s["legvel"][i]
    env.data.qvel[rv:rv+3] = s["linvel"]
    mujoco.mj_forward(env.model, env.data)

static = []
for s in spec["states"]:
    set_state(s)
    static.append([float(x) for x in env._obs()])

# rollout: fix the start, then drive env.step through the identical action sequence
set_state(spec["rollout_start"])
env._step_count = 0
rollout = []
for a in spec["rollout_actions"]:
    obs, _r, _done, _info = env.step(np.asarray(a, dtype=np.float32))
    rollout.append([float(x) for x in obs])
print(json.dumps({"static": static, "rollout": rollout}))
`;
  try {
    const stdout = execFileSync(py, ['-c', pyScript, specFile], { encoding: 'utf8' });
    reference = JSON.parse(stdout.trim().split('\n').pop());
  } catch (err) {
    console.error(`  (python reference failed: ${err.message.split('\n')[0]})`);
  }
} else {
  console.warn('  (.venv python not found — cannot cross-check; run with the repo venv)');
}

if (!reference) {
  console.error('\nFAIL — no python reference to compare against.');
  process.exit(1);
}

// 1) STATIC states
STATES.forEach((s, i) => {
  setState(s);
  const got = buildObs(sim);
  const ref = reference.static[i];
  let maxErr = 0;
  for (let k = 0; k < 23; k++) maxErr = Math.max(maxErr, Math.abs(got[k] - ref[k]));
  if (maxErr <= TOL) ok(`static ${i}: browser WASM obs == quadruped_env._obs (max Δ ${maxErr.toExponential(2)})`);
  else bad(`static ${i}`, `max Δ ${maxErr.toExponential(3)} > ${TOL}\n    got=${[...got]}\n    ref=${ref}`);
});

// 2) CONTACT ROLLOUT — obs at every control step over a real ground-contact rollout
setRolloutStart();
let rollMax = 0, rollWorst = -1;
for (let t = 0; t < ROLLOUT_STEPS; t++) {
  stepBrowserPhysics(ROLLOUT_ACTIONS[t]);
  const got = buildObs(sim);
  const ref = reference.rollout[t];
  let e = 0;
  for (let k = 0; k < 23; k++) e = Math.max(e, Math.abs(got[k] - ref[k]));
  if (e > rollMax) { rollMax = e; rollWorst = t; }
}
if (rollMax <= TOL) {
  ok(`rollout: browser env obs == quadruped_env.step over ${ROLLOUT_STEPS} contact steps (max Δ ${rollMax.toExponential(2)} @ step ${rollWorst})`);
} else {
  bad('rollout', `max Δ ${rollMax.toExponential(3)} > ${TOL} @ step ${rollWorst} of ${ROLLOUT_STEPS}`);
}

console.log(`\n${failed === 0 ? 'PASS' : 'FAIL'} — ${failed} failure(s).`);
process.exit(failed === 0 ? 0 : 1);
