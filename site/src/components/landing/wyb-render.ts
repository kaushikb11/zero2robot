/* ============================================================================
   wyb-render.ts — the ONE geometry source for the "What you'll build" strip.

   A card's static SSR poster and its hover canvas replay MUST agree pixel-for-
   pixel, so both go through this module:

     · normalize(kind, raw)  — build-time: raw curriculum/.../demo/vizdata.json
       → a compact, JSON-serialisable payload whose frames are already in a
       resolution-free [0,1] × [0,1] space (y-down). Runs in the Astro
       frontmatter (Node) at SSG time; the payload is embedded in the page.
     · shapesAt(payload, t)  — pure: a normalised time t∈[0,1] → a flat list of
       drawing primitives. Called by BOTH the SSR poster (t=1, the honest
       end-state) and the client rAF loop (t sweeping 0→1).
     · shapesToSvg(shapes)   — build-time: primitives → an inline <svg> string
       (the JS-off / reduced-motion poster).
     · drawShapes(ctx, ...)  — client: primitives → a <canvas> frame.

   Colours are TOKENS (never literals) so light/dark theming is inherited from
   the site design system. SVG uses var(--x) directly; canvas resolves the vars
   off the live element via getComputedStyle.
   ========================================================================== */

export type ColorTok =
  | "ink" | "inkmute" | "book" | "booksoft" | "signal"
  | "target" | "pusher" | "block" | "rule" | "alert" | "paper";

export const COLOR_VARS: Record<ColorTok, string> = {
  ink: "--ink", inkmute: "--ink-mute", book: "--book", booksoft: "--book-soft",
  signal: "--signal", target: "--entity-target", pusher: "--entity-pusher",
  block: "--entity-block", rule: "--rule-strong", alert: "--alert", paper: "--paper",
};

export type Shape =
  | { k: "line"; x1: number; y1: number; x2: number; y2: number; c: ColorTok; w?: number; dash?: boolean; o?: number }
  | { k: "poly"; pts: [number, number][]; c: ColorTok; w?: number; fill?: boolean; close?: boolean; o?: number }
  | { k: "circle"; x: number; y: number; r: number; c: ColorTok; fill?: boolean; w?: number; o?: number };

export type Payload = { kind: string; [k: string]: any };

// --- tiny math helpers ------------------------------------------------------
const r3 = (n: number) => Math.round(n * 1000) / 1000;
function subsample<T>(a: T[], keep: number): T[] {
  if (a.length <= keep) return a;
  const out: T[] = [];
  for (let i = 0; i < keep; i++) out.push(a[Math.round((i * (a.length - 1)) / (keep - 1))]);
  return out;
}
const frameIdx = (n: number, t: number) => Math.min(n - 1, Math.max(0, Math.floor(t * (n - 1) + 1e-6)));

// Isotropic fit: centre a set of world points in the [pad, 1-pad] box so a
// small trajectory fills the tile instead of hiding in one corner.
function fitBox(pts: [number, number][], pad: number) {
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
  const span = Math.max(xmax - xmin, ymax - ymin) || 1;
  const S = (1 - 2 * pad) / span;
  return { S, nx: (x: number) => r3(0.5 + (x - cx) * S), ny: (y: number) => r3(0.5 - (y - cy) * S) };
}

/* ========================================================================== *
 *  NORMALISE — one branch per card kind. World units → [0,1] (y-down).
 * ========================================================================== */
export function normalize(kind: string, raw: any): Payload {
  switch (kind) {
    case "quad": return normQuad(raw);
    case "arm": return normArm(raw);
    case "cartpole": return normCartpole(raw);
    case "reach": return normReach(raw);
    case "pusht": return normPusht(raw);
    case "contact": return normContact(raw);
    case "swarm": return normSwarm(raw);
    default: throw new Error(`wyb: unknown kind ${kind}`);
  }
}

// ---- quadruped: camera-follows the torso; the ground scrolls under it -------
function normQuad(raw: any): Payload {
  const g = raw.geometry;
  const all: any[] = raw.gait.frames;
  // Trim the honest stumble at the tail (the emergent gait falls before the
  // horizon); keep the clean stride cycle for a looping showcase.
  let end = all.length;
  for (let i = 4; i < all.length; i++) {
    if (all[i].torso[1] < 0.205) { end = i; break; }
  }
  const kept = subsample(all.slice(0, Math.max(40, end)), 46);
  const SX = 1.4, GY = 0.88, SY = 1.9; // world→norm
  const x0 = kept[0].torso[0];
  const order = ["FL", "FR", "HL", "HR"];
  const frames = kept.map((f: any) => {
    const tx = f.torso[0];
    const legs: number[] = [];
    for (const leg of order) {
      for (const p of f.legs[leg]) {
        legs.push(r3(0.5 + (p[0] - tx) * SX), r3(GY - p[1] * SY));
      }
    }
    return { ty: r3(GY - f.torso[1] * SY), gs: r3((tx - x0) * SX), legs };
  });
  return {
    kind: "quad",
    halfW: r3(g.torso_half[0] * SX),
    halfH: r3(g.torso_half[1] * SY),
    footR: 0.03,
    near: [1, 3], // FR, HR — the near (front) legs
    frames,
  };
}

// ---- 2-link arm reaching a target (top-down) -------------------------------
function normArm(raw: any): Payload {
  const fr = subsample(raw.rollout.frames as number[][], 42);
  const t = raw.rollout.target;
  const pts: [number, number][] = [[0, 0], [t[0], t[1]]];
  for (const f of fr) { pts.push([f[0], f[1]], [f[2], f[3]]); }
  const { S, nx, ny } = fitBox(pts, 0.16);
  const frames = fr.map((f) => [nx(f[0]), ny(f[1]), nx(f[2]), ny(f[3])]);
  const ring = Math.min(0.085, Math.max(0.05, raw.success_tol * S));
  return { kind: "arm", base: [nx(0), ny(0)], target: [nx(t[0]), ny(t[1])], ring, frames };
}

// ---- cartpole swing-up: camera follows the cart, rail ticks scroll ----------
function normCartpole(raw: any): Payload {
  const ep: any[] = subsample(raw.episode, 60);
  const x0 = raw.episode[0].cart_x;
  const frames = ep.map((f) => [r3(Math.sin(f.pole_angle)), r3(Math.cos(f.pole_angle)), r3((f.cart_x - x0) * 0.16)]);
  return { kind: "cartpole", pivotY: 0.56, r: 0.4, cartW: 0.13, cartH: 0.06, railY: 0.62, frames };
}

// ---- SO-101 gripper-tip path curling onto the box (top-down) ---------------
function normReach(raw: any): Payload {
  const path: number[][] = raw.clone_tip_path;
  const box = raw.box;
  const xs = path.map((p) => p[0]).concat(box[0]);
  const ys = path.map((p) => p[1]).concat(box[1]);
  const pad = 0.06;
  const xmin = Math.min(...xs) - pad, xmax = Math.max(...xs) + pad;
  const ymin = Math.min(...ys) - pad, ymax = Math.max(...ys) + pad;
  const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
  const S = 0.74 / Math.max(xmax - xmin, ymax - ymin);
  const nx = (x: number) => r3(0.5 + (x - cx) * S);
  const ny = (y: number) => r3(0.5 - (y - cy) * S);
  return {
    kind: "reach",
    box: [nx(box[0]), ny(box[1])],
    boxHalf: r3(box[2] * S),
    ring: r3(raw.success_tol * S),
    frames: subsample(path, 56).map((p) => [nx(p[0]), ny(p[1])]),
  };
}

// ---- language-conditioned VLA pushing the T-block onto its target ----------
function normPusht(raw: any): Payload {
  const tee = raw.tee, tgt = raw.target;
  const fr = raw.rollout.frames as number[][];
  // The T = a horizontal bar + a vertical stem, each a rect. Build the two
  // rects in WORLD, rotated by yaw about the tee origin.
  function rectsWorld(px: number, py: number, yaw: number) {
    const c = Math.cos(yaw), s = Math.sin(yaw);
    const rect = (hx: number, hy: number, oy: number): [number, number][] =>
      ([[-hx, oy - hy], [hx, oy - hy], [hx, oy + hy], [-hx, oy + hy]] as [number, number][])
        .map(([x, y]) => [px + (x * c - y * s), py + (x * s + y * c)] as [number, number]);
    return { bar: rect(tee.bar_half[0], tee.bar_half[1], 0), stem: rect(tee.stem_half[0], tee.stem_half[1], tee.stem_offset_y) };
  }
  const tgtW = rectsWorld(tgt.x, tgt.y, tgt.yaw);
  const framesW = fr.map((f) => ({ ...rectsWorld(f[2], f[3], f[4]), p: [f[0], f[1]] as [number, number] }));
  const pts: [number, number][] = [...tgtW.bar, ...tgtW.stem];
  for (const fw of framesW) pts.push(...fw.bar, ...fw.stem, fw.p);
  const { nx, ny } = fitBox(pts, 0.14);
  const proj = (a: [number, number][]) => a.map(([x, y]) => [nx(x), ny(y)] as [number, number]);
  return {
    kind: "pusht",
    target: { bar: proj(tgtW.bar), stem: proj(tgtW.stem) },
    frames: framesW.map((fw) => ({ bar: proj(fw.bar), stem: proj(fw.stem), p: [nx(fw.p[0]), ny(fw.p[1])] })),
  };
}

// ---- from-scratch contact: two solvers, two dropped balls ------------------
function normContact(raw: any): Payload {
  const GY = 0.9, SY = 0.72, R = 0.072;
  const map = (h: number[]) => subsample(h, 68).map((v) => r3(GY - v * SY));
  return {
    kind: "contact",
    floorY: GY, r: R, laneP: 0.35, laneL: 0.65,
    penalty: map(raw.drop.penalty.height),
    lcp: map(raw.drop.lcp.height),
  };
}

// ---- ch2.3 swarm: a field of many parallel MJX cartpoles, at two checkpoints -
// The generator (site/scripts/vizdata/ch2.3_swarm.py) records N real MJX envs at
// two policy snapshots — EARLY (flailing) and LATE (solved) — each frame a flat
// [dx, angle]*N over the envs (dx = cart pos in [-1,1] cell-local; angle in rad,
// 0 = upright). Both windows start from the SAME reset, so replaying them in
// sequence is the honest "same start, before vs after training" story. Pure
// pass-through: the numbers are already resolution-free.
function normSwarm(raw: any): Payload {
  return {
    kind: "swarm",
    rows: raw.rows | 0,
    cols: raw.cols | 0,
    n: raw.n | 0,
    early: raw.early as number[][],
    late: raw.late as number[][],
  };
}

/* ========================================================================== *
 *  SHAPES — pure. payload + t → primitives. Used by SSR poster AND canvas.
 * ========================================================================== */
export function shapesAt(p: Payload, t: number): Shape[] {
  switch (p.kind) {
    case "quad": return quadShapes(p, t);
    case "arm": return armShapes(p, t);
    case "cartpole": return cartpoleShapes(p, t);
    case "reach": return reachShapes(p, t);
    case "pusht": return pushtShapes(p, t);
    case "contact": return contactShapes(p, t);
    case "swarm": return swarmShapes(p, t);
    default: return [];
  }
}

function quadShapes(p: Payload, t: number): Shape[] {
  const f = p.frames[frameIdx(p.frames.length, t)];
  const s: Shape[] = [];
  // scrolling ground: ticks slide left as the torso advances
  const step = 0.14, gs = f.gs % step;
  s.push({ k: "line", x1: 0.04, y1: 0.9, x2: 0.96, y2: 0.9, c: "rule", w: 1.4 });
  for (let x = 0.04 - gs; x < 0.98; x += step) {
    s.push({ k: "line", x1: r3(x), y1: 0.9, x2: r3(x - 0.05), y2: 0.955, c: "rule", w: 1, o: 0.5 });
  }
  const leg = (i: number) => [
    [f.legs[i * 6], f.legs[i * 6 + 1]], [f.legs[i * 6 + 2], f.legs[i * 6 + 3]], [f.legs[i * 6 + 4], f.legs[i * 6 + 5]],
  ] as [number, number][];
  // far legs first (behind the body), then body, then near legs (in front)
  for (const i of [0, 2]) s.push({ k: "poly", pts: leg(i), c: "rule", w: 3.4 });
  // torso body
  const bx = p.halfW, by = p.halfH, cy = f.ty;
  s.push({ k: "poly", pts: [[0.5 - bx, cy - by], [0.5 + bx, cy - by], [0.5 + bx, cy + by], [0.5 - bx, cy + by]], c: "book", fill: true, close: true });
  for (const i of [1, 3]) {
    s.push({ k: "poly", pts: leg(i), c: "ink", w: 3.6 });
    const foot = leg(i)[2];
    s.push({ k: "circle", x: foot[0], y: foot[1], r: p.footR, c: "book", fill: true });
  }
  return s;
}

function armShapes(p: Payload, t: number): Shape[] {
  const n = p.frames.length, idx = frameIdx(n, t);
  const f = p.frames[idx];
  const s: Shape[] = [];
  // target ring + centre
  s.push({ k: "circle", x: p.target[0], y: p.target[1], r: p.ring, c: "target", w: 1.6, dash: true } as any);
  s.push({ k: "circle", x: p.target[0], y: p.target[1], r: 0.014, c: "target", fill: true });
  // fingertip trail
  const trail: [number, number][] = [];
  for (let i = 0; i <= idx; i++) trail.push([p.frames[i][2], p.frames[i][3]]);
  if (trail.length > 1) s.push({ k: "poly", pts: trail, c: "signal", w: 1.6, o: 0.4 });
  // arm links
  s.push({ k: "poly", pts: [p.base, [f[0], f[1]], [f[2], f[3]]], c: "ink", w: 4 });
  s.push({ k: "circle", x: p.base[0], y: p.base[1], r: 0.03, c: "book", fill: true });
  s.push({ k: "circle", x: f[0], y: f[1], r: 0.022, c: "booksoft", fill: true });
  s.push({ k: "circle", x: f[2], y: f[3], r: 0.026, c: "signal", fill: true });
  return s;
}

function cartpoleShapes(p: Payload, t: number): Shape[] {
  const f = p.frames[frameIdx(p.frames.length, t)];
  const [sinT, cosT, gs] = f;
  const s: Shape[] = [];
  const step = 0.16, sh = gs % step;
  s.push({ k: "line", x1: 0.04, y1: p.railY, x2: 0.96, y2: p.railY, c: "rule", w: 1.4 });
  for (let x = 0.04 - sh; x < 0.98; x += step) {
    s.push({ k: "line", x1: r3(x), y1: p.railY, x2: r3(x - 0.045), y2: p.railY + 0.05, c: "rule", w: 1, o: 0.5 });
  }
  const tipX = 0.5 + sinT * p.r, tipY = p.pivotY - cosT * p.r;
  const upright = cosT > 0.9;
  // cart
  s.push({ k: "poly", pts: [[0.5 - p.cartW, p.pivotY - p.cartH], [0.5 + p.cartW, p.pivotY - p.cartH], [0.5 + p.cartW, p.pivotY + p.cartH], [0.5 - p.cartW, p.pivotY + p.cartH]], c: "book", fill: true, close: true });
  // pole + tip
  s.push({ k: "line", x1: 0.5, y1: p.pivotY, x2: r3(tipX), y2: r3(tipY), c: "ink", w: 4 });
  s.push({ k: "circle", x: r3(tipX), y: r3(tipY), r: 0.03, c: upright ? "signal" : "booksoft", fill: true });
  s.push({ k: "circle", x: 0.5, y: p.pivotY, r: 0.016, c: "ink", fill: true });
  return s;
}

function reachShapes(p: Payload, t: number): Shape[] {
  const n = p.frames.length, idx = frameIdx(n, t);
  const s: Shape[] = [];
  // success region + box
  s.push({ k: "circle", x: p.box[0], y: p.box[1], r: p.ring, c: "target", w: 1.4, dash: true } as any);
  const bh = p.boxHalf;
  s.push({ k: "poly", pts: [[p.box[0] - bh, p.box[1] - bh], [p.box[0] + bh, p.box[1] - bh], [p.box[0] + bh, p.box[1] + bh], [p.box[0] - bh, p.box[1] + bh]], c: "target", fill: true, close: true, o: 0.85 });
  // gripper-tip trail
  const trail: [number, number][] = [];
  for (let i = 0; i <= idx; i++) trail.push([p.frames[i][0], p.frames[i][1]]);
  if (trail.length > 1) s.push({ k: "poly", pts: trail, c: "pusher", w: 2.6 });
  const head = p.frames[idx];
  s.push({ k: "circle", x: head[0], y: head[1], r: 0.03, c: "ink", fill: true });
  s.push({ k: "circle", x: head[0], y: head[1], r: 0.05, c: "pusher", w: 1.6 });
  return s;
}

function pushtShapes(p: Payload, t: number): Shape[] {
  const f = p.frames[frameIdx(p.frames.length, t)];
  const s: Shape[] = [];
  // target T (outline)
  s.push({ k: "poly", pts: p.target.bar, c: "target", w: 1.6, close: true, o: 0.9 });
  s.push({ k: "poly", pts: p.target.stem, c: "target", w: 1.6, close: true, o: 0.9 });
  // moving T (filled)
  s.push({ k: "poly", pts: f.bar, c: "block", fill: true, close: true });
  s.push({ k: "poly", pts: f.stem, c: "block", fill: true, close: true });
  // pusher
  s.push({ k: "circle", x: f.p[0], y: f.p[1], r: 0.028, c: "pusher", fill: true });
  return s;
}

function contactShapes(p: Payload, t: number): Shape[] {
  const idx = frameIdx(p.penalty.length, t);
  const s: Shape[] = [];
  s.push({ k: "line", x1: 0.06, y1: p.floorY, x2: 0.94, y2: p.floorY, c: "ink", w: 1.6 });
  // faint floor hatch
  for (let x = 0.08; x < 0.93; x += 0.09) s.push({ k: "line", x1: r3(x), y1: p.floorY, x2: r3(x - 0.05), y2: p.floorY + 0.05, c: "rule", w: 1, o: 0.5 });
  s.push({ k: "circle", x: p.laneP, y: p.penalty[idx], r: p.r, c: "alert", fill: true });
  s.push({ k: "circle", x: p.laneL, y: p.lcp[idx], r: p.r, c: "target", fill: true });
  return s;
}

// A grid of cartpoles. The first half of t replays the EARLY (flailing) field,
// the second half the LATE (solved) field — same reset, so it reads as "before
// vs after training". Each pole is tinted by how upright it is, so "learning"
// is legible at a glance: a field that topples, then a field that holds.
function swarmShapes(p: Payload, t: number): Shape[] {
  const late = t >= 0.5;
  const frames: number[][] = late ? p.late : p.early;
  const F = frames.length;
  const lt = Math.min(1, late ? (t - 0.5) * 2 : t * 2);
  const f = frames[frameIdx(F, lt)];

  const rows = p.rows, cols = p.cols;
  const m = 0.03; // outer margin
  const cw = (1 - 2 * m) / cols;
  const ch = (1 - 2 * m) / rows;
  const railHalf = cw * 0.34;
  const cartW = cw * 0.11, cartH = ch * 0.05;
  const cartShift = cw * 0.2;   // dx∈[-1,1] → this much travel inside the cell
  const poleLen = ch * 0.56;
  const tipR = ch * 0.075;

  const s: Shape[] = [];
  for (let e = 0; e < p.n; e++) {
    const r = (e / cols) | 0, c = e % cols;
    const cx = m + (c + 0.5) * cw;
    const baseY = m + r * ch + ch * 0.8;
    const dx = f[e * 2], ang = f[e * 2 + 1];
    const upright = Math.cos(ang) > 0.9; // within ~0.45 rad of vertical
    const cartX = cx + dx * cartShift;
    const pivotY = baseY - cartH;
    const tipX = cartX + poleLen * Math.sin(ang);
    const tipY = pivotY - poleLen * Math.cos(ang);
    // short rail under each robot
    s.push({ k: "line", x1: r3(cx - railHalf), y1: r3(baseY), x2: r3(cx + railHalf), y2: r3(baseY), c: "rule", w: 1, o: 0.45 });
    // pole (tinted by uprightness) + cart body + tip
    s.push({ k: "line", x1: r3(cartX), y1: r3(pivotY), x2: r3(tipX), y2: r3(tipY), c: upright ? "ink" : "inkmute", w: 2.4 });
    s.push({ k: "poly", pts: [[r3(cartX - cartW), r3(baseY - cartH)], [r3(cartX + cartW), r3(baseY - cartH)], [r3(cartX + cartW), r3(baseY + cartH)], [r3(cartX - cartW), r3(baseY + cartH)]], c: "book", fill: true, close: true });
    s.push({ k: "circle", x: r3(tipX), y: r3(tipY), r: r3(tipR), c: upright ? "signal" : "booksoft", fill: true });
  }
  return s;
}

/* ========================================================================== *
 *  RENDERERS
 * ========================================================================== */
export function shapesToSvg(shapes: Shape[], size = 100): string {
  const S = size;
  const col = (c: ColorTok) => `var(${COLOR_VARS[c]})`;
  const parts = shapes.map((sh) => {
    const op = sh.o != null ? ` opacity="${sh.o}"` : "";
    if (sh.k === "line") {
      const dash = sh.dash ? ` stroke-dasharray="${0.03 * S} ${0.03 * S}"` : "";
      return `<line x1="${r3(sh.x1 * S)}" y1="${r3(sh.y1 * S)}" x2="${r3(sh.x2 * S)}" y2="${r3(sh.y2 * S)}" stroke="${col(sh.c)}" stroke-width="${(sh.w ?? 1) * (S / 100)}" stroke-linecap="round"${dash}${op}/>`;
    }
    if (sh.k === "circle") {
      const dash = (sh as any).dash ? ` stroke-dasharray="${0.03 * S} ${0.03 * S}"` : "";
      const fill = sh.fill ? col(sh.c) : "none";
      const stroke = sh.fill ? "none" : col(sh.c);
      return `<circle cx="${r3(sh.x * S)}" cy="${r3(sh.y * S)}" r="${r3(sh.r * S)}" fill="${fill}" stroke="${stroke}" stroke-width="${(sh.w ?? 1.4) * (S / 100)}"${dash}${op}/>`;
    }
    const pts = sh.pts.map(([x, y]) => `${r3(x * S)},${r3(y * S)}`).join(" ");
    const fill = sh.fill ? col(sh.c) : "none";
    const stroke = sh.fill ? "none" : col(sh.c);
    const tag = sh.close || sh.fill ? "polygon" : "polyline";
    return `<${tag} points="${pts}" fill="${fill}" stroke="${stroke}" stroke-width="${(sh.w ?? 1.4) * (S / 100)}" stroke-linejoin="round" stroke-linecap="round"${op}/>`;
  });
  return `<svg viewBox="0 0 ${S} ${S}" class="wyb-svg" aria-hidden="true" preserveAspectRatio="xMidYMid meet">${parts.join("")}</svg>`;
}

// client-only: draw one frame of primitives to a 2D canvas context.
export function drawShapes(
  ctx: CanvasRenderingContext2D,
  shapes: Shape[],
  W: number,
  H: number,
  resolve: (c: ColorTok) => string,
): void {
  ctx.clearRect(0, 0, W, H);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  const sw = (w: number) => (w * Math.min(W, H)) / 100;
  for (const sh of shapes) {
    ctx.globalAlpha = sh.o != null ? sh.o : 1;
    const color = resolve(sh.c);
    if (sh.k === "line") {
      ctx.beginPath();
      ctx.setLineDash(sh.dash ? [0.03 * W, 0.03 * W] : []);
      ctx.moveTo(sh.x1 * W, sh.y1 * H);
      ctx.lineTo(sh.x2 * W, sh.y2 * H);
      ctx.strokeStyle = color;
      ctx.lineWidth = sw(sh.w ?? 1);
      ctx.stroke();
    } else if (sh.k === "circle") {
      ctx.beginPath();
      ctx.setLineDash((sh as any).dash ? [0.03 * W, 0.03 * W] : []);
      ctx.arc(sh.x * W, sh.y * H, sh.r * Math.min(W, H), 0, Math.PI * 2);
      if (sh.fill) { ctx.fillStyle = color; ctx.fill(); }
      else { ctx.strokeStyle = color; ctx.lineWidth = sw(sh.w ?? 1.4); ctx.stroke(); }
    } else {
      ctx.beginPath();
      ctx.setLineDash([]);
      sh.pts.forEach(([x, y], i) => (i ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H)));
      if (sh.close || sh.fill) ctx.closePath();
      if (sh.fill) { ctx.fillStyle = color; ctx.fill(); }
      else { ctx.strokeStyle = color; ctx.lineWidth = sw(sh.w ?? 1.4); ctx.stroke(); }
    }
  }
  ctx.globalAlpha = 1;
  ctx.setLineDash([]);
}
