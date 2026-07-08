// Thin wrapper around the official @mujoco/mujoco WASM bindings
// (google-deepmind/mujoco, wasm/ directory; single-threaded build — the /mt
// build needs COOP/COEP headers and is not required for this spike).
import loadMujoco, { type MainModule, type MjModel, type MjData } from '@mujoco/mujoco';
import mujocoWasmUrl from '@mujoco/mujoco/mujoco.wasm?url';

export interface Sim {
  module: MainModule;
  model: MjModel;
  data: MjData;
  /** Physics timestep in seconds (from the MJCF <option timestep>). */
  timestep: number;
  /** Number of actuators (ctrl dimension). */
  nu: number;
  /** Step the physics n times with the given ctrl vector. */
  step(ctrl: Float64Array | number[], nSteps: number): void;
  /** World xy position of a named geom (fresh read each call). */
  geomXY(name: string): [number, number];
  /** Planar qvel of the pusher joints [vx, vy] (PushT scenes only). */
  pusherVel(): [number, number];
  /** Read the scalar qpos of a named (1-dof) joint. */
  jointQpos(name: string): number;
  /** Write the scalar qpos of a named (1-dof) joint (call forward() after). */
  setJointQpos(name: string, value: number): void;
  /** Read the scalar qvel of a named (1-dof) joint. */
  jointQvel(name: string): number;
  /** Write the scalar qvel of a named (1-dof) joint (call forward() after). */
  setJointQvel(name: string, value: number): void;
  /** Read qpos at a raw index (for MULTI-dof joints — e.g. a free joint's 7-slot
   *  qpos block: [x,y,z, qw,qx,qy,qz]). Used by the quadruped obs builder. */
  qposAt(i: number): number;
  /** Write qpos at a raw index (call forward() after). */
  setQposAt(i: number, value: number): void;
  /** Read qvel at a raw index (for MULTI-dof joints — e.g. a free joint's 6-dof
   *  block: [vx,vy,vz, wx,wy,wz], world frame). */
  qvelAt(i: number): number;
  /** Write qvel at a raw index (call forward() after). */
  setQvelAt(i: number, value: number): void;
  /** Starting qpos address of a named joint (a free joint's 7-slot block begins
   *  here; the torso height is qpos[jointQposAdr('root') + 2]). */
  jointQposAdr(name: string): number;
  /** Starting qvel/dof address of a named joint (a free joint's 6-dof block; the
   *  torso linear velocity is qvel[jointDofAdr('root') .. +3], world frame). */
  jointDofAdr(name: string): number;
  /** World-frame rotation matrix (row-major, 9 floats) of a named body. The
   *  torso "up" vector is its 3rd column: [xmat[2], xmat[5], xmat[8]]. */
  bodyXmat(name: string): number[];
  /** World-frame position [x, y, z] of a named body's frame origin (data.xpos).
   *  Used by the quadruped side-view render to place hips/knees/feet from the
   *  free-floating pose — display only, never an obs input. */
  bodyXpos(name: string): [number, number, number];
  /** Set the ctrl vector (does not step). */
  setCtrl(ctrl: Float64Array | number[]): void;
  /** mj_resetData: zero qpos/qvel/ctrl back to the model defaults. */
  resetData(): void;
  /** mj_forward: recompute derived quantities from the current qpos/qvel. */
  forward(): void;
  /** Free the WASM-side objects. */
  dispose(): void;
}

export async function createSim(sceneXml: string): Promise<Sim> {
  // The Emscripten loader resolves mujoco.wasm relative to import.meta.url,
  // which breaks under bundling; hand it the Vite-resolved URL instead.
  const module = (await loadMujoco({
    locateFile: (path: string) => (path.endsWith('.wasm') ? mujocoWasmUrl : path),
  } as unknown)) as MainModule;

  // MjModel/MjData are Embind handles with no GC — every allocation below must
  // be freed on any failure or it leaks WASM heap. from_xml_string can throw on
  // a bad scene; guard everything after the first allocation with try/catch and
  // free in reverse order before rethrowing.
  const model = module.MjModel.from_xml_string(sceneXml);
  let data: MjData | null = null;
  let timestep: number;
  try {
    data = new module.MjData(model);
    module.mj_forward(model, data);

    // model.opt is a fresh Embind handle on each access — read the one scalar
    // we need and free the temporary immediately so it does not leak.
    const opt = model.opt;
    timestep = opt.timestep;
    opt.delete();
  } catch (err) {
    if (data) data.delete();
    model.delete();
    throw err;
  }

  // Past the try/catch, data is guaranteed allocated (the catch rethrows).
  const activeData = data!;

  const geomId = (name: string): number => {
    const id = module.mj_name2id(model, module.mjtObj.mjOBJ_GEOM.value, name);
    if (id < 0) throw new Error(`geom not found in scene: ${name}`);
    return id;
  };

  // Cache each joint's qpos address (the scenes here use only 1-dof slide/hinge
  // joints, so a single qpos slot per joint) to avoid repeated name lookups per
  // step.
  const qposAdrCache = new Map<string, number>();
  const jointQposAdr = (name: string): number => {
    const cached = qposAdrCache.get(name);
    if (cached !== undefined) return cached;
    const id = module.mj_name2id(model, module.mjtObj.mjOBJ_JOINT.value, name);
    if (id < 0) throw new Error(`joint not found in scene: ${name}`);
    const adr = model.jnt_qposadr[id] as number;
    qposAdrCache.set(name, adr);
    return adr;
  };

  // Same lazy cache for each joint's qvel (dof) address. Resolved on first use
  // — NOT eagerly at construction — so a scene without the pusher joints (e.g.
  // cartpole's slider/hinge) boots fine; only pusherVel() touches pusher_x/_y,
  // and it does so lazily here. Keeps PushT working (pusherVel unchanged) while
  // making createSim scene-agnostic.
  const qvelAdrCache = new Map<string, number>();
  const jointQvelAdr = (name: string): number => {
    const cached = qvelAdrCache.get(name);
    if (cached !== undefined) return cached;
    const id = module.mj_name2id(model, module.mjtObj.mjOBJ_JOINT.value, name);
    if (id < 0) throw new Error(`joint not found in scene: ${name}`);
    const adr = model.jnt_dofadr[id] as number;
    qvelAdrCache.set(name, adr);
    return adr;
  };

  return {
    module,
    model,
    data: activeData,
    timestep,
    nu: model.nu,
    step(ctrl, nSteps) {
      const c = activeData.ctrl;
      for (let i = 0; i < c.length && i < ctrl.length; i++) c[i] = ctrl[i];
      for (let i = 0; i < nSteps; i++) module.mj_step(model, activeData);
    },
    geomXY(name) {
      const adr = geomId(name) * 3;
      const xpos = activeData.geom_xpos;
      return [xpos[adr], xpos[adr + 1]];
    },
    pusherVel() {
      const qvel = activeData.qvel;
      return [qvel[jointQvelAdr('pusher_x')], qvel[jointQvelAdr('pusher_y')]];
    },
    jointQpos(name) {
      return activeData.qpos[jointQposAdr(name)];
    },
    setJointQpos(name, value) {
      activeData.qpos[jointQposAdr(name)] = value;
    },
    jointQvel(name) {
      return activeData.qvel[jointQvelAdr(name)];
    },
    setJointQvel(name, value) {
      activeData.qvel[jointQvelAdr(name)] = value;
    },
    qposAt(i) {
      return activeData.qpos[i];
    },
    setQposAt(i, value) {
      activeData.qpos[i] = value;
    },
    qvelAt(i) {
      return activeData.qvel[i];
    },
    setQvelAt(i, value) {
      activeData.qvel[i] = value;
    },
    jointQposAdr(name) {
      return jointQposAdr(name);
    },
    jointDofAdr(name) {
      return jointQvelAdr(name);
    },
    bodyXmat(name) {
      const id = module.mj_name2id(model, module.mjtObj.mjOBJ_BODY.value, name);
      if (id < 0) throw new Error(`body not found in scene: ${name}`);
      const xmat = activeData.xmat;
      const adr = id * 9;
      const out = new Array<number>(9);
      for (let k = 0; k < 9; k++) out[k] = xmat[adr + k];
      return out;
    },
    bodyXpos(name) {
      const id = module.mj_name2id(model, module.mjtObj.mjOBJ_BODY.value, name);
      if (id < 0) throw new Error(`body not found in scene: ${name}`);
      const xpos = activeData.xpos;
      const adr = id * 3;
      return [xpos[adr], xpos[adr + 1], xpos[adr + 2]];
    },
    setCtrl(ctrl) {
      const c = activeData.ctrl;
      for (let i = 0; i < c.length && i < ctrl.length; i++) c[i] = ctrl[i];
    },
    resetData() {
      module.mj_resetData(model, activeData);
    },
    forward() {
      module.mj_forward(model, activeData);
    },
    dispose() {
      activeData.delete();
      model.delete();
    },
  };
}
