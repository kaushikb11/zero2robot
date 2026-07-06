// Obs-parity self-check: the browser's cartpole obs builder produces byte-equal
// observations to curriculum/common/envs/cartpole/cartpole_env.py for the SAME
// state. This is the guarantee the ch2.1 PPO drive loop depends on — a policy
// trained on cartpole_env obs is later fed browser obs, so they MUST match.
//
// assembleObs() from src/teleop/cartpole_obs.ts (transpiled in-memory, no deps)
// is compared, for a set of known states, against:
//   (a) values computed live from CartpoleEnv via .venv/bin/python, when available;
//   (b) an embedded golden (those same python values), always.
// Field order + the cos/sin(angle-from-upright) encoding are what this pins. Run:
//   node scripts/cartpole_obs_parity_check.mjs

import { readFileSync, writeFileSync, mkdtempSync, existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve, join } from 'node:path';
import { tmpdir } from 'node:os';
import { execFileSync } from 'node:child_process';
import ts from 'typescript';

const here = dirname(fileURLToPath(import.meta.url));
const srcDir = resolve(here, '..', 'src', 'teleop');
const repoRoot = resolve(here, '..', '..');
const outDir = mkdtempSync(join(tmpdir(), 'z2r-cp-obs-'));

// cartpole_obs.ts imports only `import type { Sim }` (elided on transpile), so it
// transpiles standalone.
const src = readFileSync(join(srcDir, 'cartpole_obs.ts'), 'utf8');
const js = ts.transpileModule(src, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const modPath = join(outDir, 'cartpole_obs.mjs');
writeFileSync(modPath, js);
const { assembleObs } = await import(pathToFileURL(modPath).href);

// Known states: [cart_pos, cart_vel, pole_angle, pole_angvel]. pole_angle is the
// raw hinge angle (some past +-pi to exercise the wrap seam in the cos/sin).
const STATES = [
  { cart_pos: 0.0, cart_vel: 0.0, pole_angle: 0.0, pole_angvel: 0.0 },
  { cart_pos: 0.5, cart_vel: -0.3, pole_angle: 0.15, pole_angvel: 0.8 },
  { cart_pos: -1.2, cart_vel: 0.7, pole_angle: -0.19, pole_angvel: -1.5 },
  { cart_pos: 2.1, cart_vel: 0.1, pole_angle: 3.5, pole_angvel: 0.2 }, // > pi -> wraps
];

// Embedded golden = CartpoleEnv._obs() (mujoco 3.10.0); regenerated live below
// when .venv python is present.
const GOLDEN = [
  [0.0, 0.0, 1.0, 0.0, 0.0],
  [0.5, -0.30000001192092896, 0.9887710809707642, 0.14943812787532806, 0.800000011920929],
  [-1.2000000476837158, 0.699999988079071, 0.9820042252540588, -0.18885889649391174, -1.5],
  [2.0999999046325684, 0.10000000149011612, -0.9364566802978516, -0.35078322887420654, 0.20000000298023224],
];

const TOL = 1e-6;
let failed = 0;
const ok = (n) => console.log(`  ok   ${n}`);
const bad = (n, d) => { failed += 1; console.error(`  FAIL ${n}: ${d}`); };

function compare(label, reference) {
  STATES.forEach((s, i) => {
    if (!reference[i]) return; // embedded golden may cover fewer states than live
    const got = assembleObs(s.cart_pos, s.cart_vel, s.pole_angle, s.pole_angvel);
    const ref = reference[i];
    let maxErr = 0;
    for (let k = 0; k < 5; k++) maxErr = Math.max(maxErr, Math.abs(got[k] - ref[k]));
    if (maxErr <= TOL) ok(`${label} state ${i}: browser obs == ref (max Δ ${maxErr.toExponential(2)})`);
    else bad(`${label} state ${i}`, `max Δ ${maxErr.toExponential(3)} > ${TOL}\n    got=${[...got]}\n    ref=${ref}`);
  });
}

// (a) live cross-check against cartpole_env, when .venv python is available.
const py = resolve(repoRoot, '.venv', 'bin', 'python');
if (existsSync(py)) {
  const statesFile = join(outDir, 'states.json');
  writeFileSync(statesFile, JSON.stringify(STATES));
  const pyScript = `
import sys, json
sys.path.insert(0, ${JSON.stringify(join(repoRoot, 'curriculum', 'common', 'envs', 'cartpole'))})
import mujoco
from cartpole_env import CartpoleEnv
env = CartpoleEnv()
states = json.load(open(sys.argv[1]))
out = []
for s in states:
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[env._jadr["slider"]] = s["cart_pos"]
    env.data.qpos[env._jadr["hinge"]] = s["pole_angle"]
    env.data.qvel[env._vadr["slider"]] = s["cart_vel"]
    env.data.qvel[env._vadr["hinge"]] = s["pole_angvel"]
    mujoco.mj_forward(env.model, env.data)
    out.append([float(x) for x in env._obs()])
print(json.dumps(out))
`;
  try {
    const stdout = execFileSync(py, ['-c', pyScript, statesFile], { encoding: 'utf8' });
    const live = JSON.parse(stdout.trim().split('\n').pop());
    compare('live cartpole_env', live);
  } catch (err) {
    console.warn(`  (live cartpole_env check skipped: ${err.message.split('\n')[0]})`);
    compare('embedded golden', GOLDEN);
  }
} else {
  console.warn('  (.venv python not found — comparing against embedded golden only)');
  compare('embedded golden', GOLDEN);
}

console.log(`\n${failed === 0 ? 'PASS' : 'FAIL'} — ${failed} failure(s).`);
process.exit(failed === 0 ? 0 : 1);
