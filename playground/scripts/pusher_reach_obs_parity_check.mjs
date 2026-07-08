// Obs-parity self-check: the browser's pusher-reach obs builder produces byte-equal
// observations to curriculum/common/envs/pusher_reach/pusher_reach_env.py for the
// SAME state. This is the guarantee the ch2.2 SAC (and ch4 offline/serl) drive
// loops depend on — a policy trained on pusher_reach_env obs is later fed browser
// obs, so they MUST match.
//
// assembleObs() from src/teleop/pusher_reach_obs.ts (transpiled in-memory, no deps)
// is compared, for a set of known states, against:
//   (a) values computed live from PusherReachEnv via .venv/bin/python, when available;
//   (b) an embedded golden (those same python values), always.
// The nontrivial thing this pins (beyond field order + cos/sin encoding) is that
// the browser's ANALYTIC fingertip forward-kinematics equals MuJoCo's mj_forward
// site_xpos — so dx/dy match without a WASM kinematics read. Run:
//   node scripts/pusher_reach_obs_parity_check.mjs

import { readFileSync, writeFileSync, mkdtempSync, existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve, join } from 'node:path';
import { tmpdir } from 'node:os';
import { execFileSync } from 'node:child_process';
import ts from 'typescript';

const here = dirname(fileURLToPath(import.meta.url));
const srcDir = resolve(here, '..', 'src', 'teleop');
const repoRoot = resolve(here, '..', '..');
const outDir = mkdtempSync(join(tmpdir(), 'z2r-pr-obs-'));

// pusher_reach_obs.ts imports only `import type { Sim }` (elided on transpile), so
// it transpiles standalone.
const src = readFileSync(join(srcDir, 'pusher_reach_obs.ts'), 'utf8');
const js = ts.transpileModule(src, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const modPath = join(outDir, 'pusher_reach_obs.mjs');
writeFileSync(modPath, js);
const { assembleObs } = await import(pathToFileURL(modPath).href);

// Known states: [shoulder, elbow, shoulder_vel, elbow_vel, target_x, target_y].
// The joint angles are raw hinge angles (some past +-pi to exercise the wrap seam
// in the cos/sin AND the FK). Targets sit in the reachable annulus.
const STATES = [
  { shoulder: 0.0, elbow: 0.0, shoulder_vel: 0.0, elbow_vel: 0.0, target_x: 0.15, target_y: 0.0 },
  { shoulder: 0.6, elbow: -1.2, shoulder_vel: 0.4, elbow_vel: -0.7, target_x: 0.05, target_y: 0.12 },
  { shoulder: -2.4, elbow: 1.9, shoulder_vel: -1.1, elbow_vel: 0.3, target_x: -0.08, target_y: -0.1 },
  { shoulder: 3.5, elbow: -3.4, shoulder_vel: 0.2, elbow_vel: 2.0, target_x: 0.1, target_y: -0.14 }, // past +-pi -> wraps
];

// Embedded golden = PusherReachEnv._obs() (mujoco 3.10.0); regenerated live below
// when .venv python is present, and cross-checked against it (catches drift).
const GOLDEN = [
  [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.05000000074505806, 0.0],
  [0.8253356218338013, 0.5646424889564514, 0.3623577654361725, -0.9320390820503235,
    0.4000000059604645, -0.699999988079071, -0.11506712436676025, 0.11999999731779099],
  [-0.7373937368392944, -0.6754631996154785, -0.32328957319259644, 0.9463000893592834,
    -1.100000023841858, 0.30000001192092896, -0.09401888400316238, 0.015488872304558754],
  [-0.9364566802978516, -0.35078322887420654, -0.9667981863021851, 0.2555411159992218,
    0.20000000298023224, 2.0, 0.09414525330066681, -0.11490502208471298],
];

const TOL = 1e-5;
let failed = 0;
const ok = (n) => console.log(`  ok   ${n}`);
const bad = (n, d) => { failed += 1; console.error(`  FAIL ${n}: ${d}`); };

function compare(label, reference) {
  STATES.forEach((s, i) => {
    if (!reference[i]) return;
    const got = assembleObs(s.shoulder, s.elbow, s.shoulder_vel, s.elbow_vel, s.target_x, s.target_y);
    const ref = reference[i];
    let maxErr = 0;
    for (let k = 0; k < 8; k++) maxErr = Math.max(maxErr, Math.abs(got[k] - ref[k]));
    if (maxErr <= TOL) ok(`${label} state ${i}: browser obs == ref (max Δ ${maxErr.toExponential(2)})`);
    else bad(`${label} state ${i}`, `max Δ ${maxErr.toExponential(3)} > ${TOL}\n    got=${[...got]}\n    ref=${ref}`);
  });
}

// (a) live cross-check against pusher_reach_env, when .venv python is available.
const py = resolve(repoRoot, '.venv', 'bin', 'python');
if (existsSync(py)) {
  const statesFile = join(outDir, 'states.json');
  writeFileSync(statesFile, JSON.stringify(STATES));
  const pyScript = `
import sys, json
sys.path.insert(0, ${JSON.stringify(join(repoRoot, 'curriculum', 'common', 'envs', 'pusher_reach'))})
import numpy as np
import mujoco
from pusher_reach_env import PusherReachEnv
env = PusherReachEnv()
states = json.load(open(sys.argv[1]))
out = []
for s in states:
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[env._jadr["shoulder"]] = s["shoulder"]
    env.data.qpos[env._jadr["elbow"]] = s["elbow"]
    env.data.qvel[env._vadr["shoulder"]] = s["shoulder_vel"]
    env.data.qvel[env._vadr["elbow"]] = s["elbow_vel"]
    env.data.mocap_pos[0] = np.array([s["target_x"], s["target_y"], 0.0])
    mujoco.mj_forward(env.model, env.data)
    out.append([float(x) for x in env._obs()])
print(json.dumps(out))
`;
  try {
    const stdout = execFileSync(py, ['-c', pyScript, statesFile], { encoding: 'utf8' });
    const live = JSON.parse(stdout.trim().split('\n').pop());
    compare('live pusher_reach_env', live);
    // Also confirm the embedded golden still equals the live env (catches drift).
    let goldenErr = 0;
    live.forEach((row, i) => row.forEach((v, k) => (goldenErr = Math.max(goldenErr, Math.abs(v - GOLDEN[i][k])))));
    if (goldenErr <= TOL) ok(`embedded golden still matches live pusher_reach_env (max Δ ${goldenErr.toExponential(2)})`);
    else bad('embedded golden vs live', `max Δ ${goldenErr.toExponential(3)} — regenerate GOLDEN`);
  } catch (err) {
    console.warn(`  (live pusher_reach_env check skipped: ${err.message.split('\n')[0]})`);
    compare('embedded golden', GOLDEN);
  }
} else {
  console.warn('  (.venv python not found — comparing against embedded golden only)');
  compare('embedded golden', GOLDEN);
}

console.log(`\n${failed === 0 ? 'PASS' : 'FAIL'} — ${failed} failure(s).`);
process.exit(failed === 0 ? 0 : 1);
