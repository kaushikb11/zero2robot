// Obs-parity self-check: the browser's obs builder produces byte-equal
// observations to curriculum/common/envs/pusht/pusht_env.py for the SAME state.
// This is the guarantee the Phase-3 policy loop depends on — a policy trained on
// pusht_env obs is later fed browser obs, so they MUST match.
//
// assembleObs() from src/teleop/pusht_obs.ts (transpiled in-memory, no deps) is
// compared, for a set of known states, against:
//   (a) values computed live from PushTEnv via .venv/bin/python, when available;
//   (b) an embedded golden (those same python values), always.
// Field order + the sin/cos yaw encoding are what this pins. Run:
//   node scripts/obs_parity_check.mjs

import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve, join } from 'node:path';
import { tmpdir } from 'node:os';
import { existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import ts from 'typescript';

const here = dirname(fileURLToPath(import.meta.url));
const srcDir = resolve(here, '..', 'src', 'teleop');
const repoRoot = resolve(here, '..', '..');
const outDir = mkdtempSync(join(tmpdir(), 'z2r-obs-'));

// pusht_obs.ts imports only `import type { Sim }` (elided on transpile), so it
// transpiles standalone.
const src = readFileSync(join(srcDir, 'pusht_obs.ts'), 'utf8');
const js = ts.transpileModule(src, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const modPath = join(outDir, 'pusht_obs.mjs');
writeFileSync(modPath, js);
const { assembleObs } = await import(pathToFileURL(modPath).href);

// Known states: [pusher_x, pusher_y, tee_x, tee_y, tee_yaw]. Target is fixed
// (0, 0, 0) per pusht_env.TARGET_POSE.
const STATES = [
  { pusher_x: -0.2, pusher_y: 0.1, tee_x: 0.12, tee_y: -0.05, tee_yaw: 0.3 },
  { pusher_x: 0.05, pusher_y: -0.15, tee_x: -0.08, tee_y: 0.2, tee_yaw: -1.2 },
  { pusher_x: 0.31, pusher_y: -0.28, tee_x: 0.0, tee_y: 0.0, tee_yaw: 3.1 },
];

// Embedded golden = PushTEnv._obs() (mujoco 3.10.0), see the commit that added
// this file; regenerated live below when .venv python is present.
const GOLDEN = [
  [-0.20000000298023224, 0.10000000149011612, 0.11999999731779099, -0.05000000074505806,
    0.29552021622657776, 0.9553365111351013, 0.0, 0.0, 0.0, 1.0],
  [0.05000000074505806, -0.15000000596046448, -0.07999999821186066, 0.20000000298023224,
    -0.9320390820503235, 0.3623577654361725, 0.0, 0.0, 0.0, 1.0],
  [0.3100000023841858, -0.2800000011920929, 0.0, 0.0,
    0.04158066213130951, -0.9991351366043091, 0.0, 0.0, 0.0, 1.0],
];

const TOL = 1e-6;
let failed = 0;
const ok = (n) => console.log(`  ok   ${n}`);
const bad = (n, d) => { failed += 1; console.error(`  FAIL ${n}: ${d}`); };

function compare(label, reference) {
  STATES.forEach((s, i) => {
    const got = assembleObs(s.pusher_x, s.pusher_y, s.tee_x, s.tee_y, s.tee_yaw, 0, 0, 0);
    const ref = reference[i];
    let maxErr = 0;
    for (let k = 0; k < 10; k++) maxErr = Math.max(maxErr, Math.abs(got[k] - ref[k]));
    if (maxErr <= TOL) ok(`${label} state ${i}: browser obs == ref (max Δ ${maxErr.toExponential(2)})`);
    else bad(`${label} state ${i}`, `max Δ ${maxErr.toExponential(3)} > ${TOL}\n    got=${[...got]}\n    ref=${ref}`);
  });
}

// (a) live cross-check against pusht_env, when .venv python is available.
const py = resolve(repoRoot, '.venv', 'bin', 'python');
if (existsSync(py)) {
  const statesFile = join(outDir, 'states.json');
  writeFileSync(statesFile, JSON.stringify(STATES));
  const pyScript = `
import sys, json
sys.path.insert(0, ${JSON.stringify(join(repoRoot, 'curriculum', 'common', 'envs', 'pusht'))})
import mujoco
from pusht_env import PushTEnv
env = PushTEnv()
states = json.load(open(sys.argv[1]))
out = []
for s in states:
    mujoco.mj_resetData(env.model, env.data)
    q = env.data.qpos
    for k, v in s.items():
        q[env._jadr[k]] = v
    mujoco.mj_forward(env.model, env.data)
    out.append([float(x) for x in env._obs()])
print(json.dumps(out))
`;
  try {
    const stdout = execFileSync(py, ['-c', pyScript, statesFile], { encoding: 'utf8' });
    const live = JSON.parse(stdout.trim().split('\n').pop());
    compare('live pusht_env', live);
    // Also confirm the embedded golden still equals the live env (catches drift).
    let goldenErr = 0;
    live.forEach((row, i) => row.forEach((v, k) => (goldenErr = Math.max(goldenErr, Math.abs(v - GOLDEN[i][k])))));
    if (goldenErr <= TOL) ok(`embedded golden still matches live pusht_env (max Δ ${goldenErr.toExponential(2)})`);
    else bad('embedded golden vs live', `max Δ ${goldenErr.toExponential(3)} — regenerate GOLDEN`);
  } catch (err) {
    console.warn(`  (live pusht_env check skipped: ${err.message.split('\n')[0]})`);
    compare('embedded golden', GOLDEN);
  }
} else {
  console.warn('  (.venv python not found — comparing against embedded golden only)');
  compare('embedded golden', GOLDEN);
}

console.log(`\n${failed === 0 ? 'PASS' : 'FAIL'} — ${failed} failure(s).`);
process.exit(failed === 0 ? 0 : 1);
