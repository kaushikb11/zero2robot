// Side-view (sagittal, x-z plane) 2D canvas render of the free-floating quadruped,
// shared by the three ch2.4/2.5/2.7 live toys. This is DISPLAY-LAYER code (the
// playground graceful-degradation ladder lets the render be bespoke): it projects
// the torso box + the four legs straight from the live MuJoCo-WASM pose — no full
// 3D, no obs dependence. The torso pose (position + quaternion) comes through the
// body world transform (xpos/xmat), the legs through each thigh/shin body's world
// origin, the feet from the shin frame + the fixed foot offset. Left legs (y>0) are
// drawn near/opaque, right legs (y<0) far/faded, so the side view reads with depth.
//
// The robot itself renders IDENTICALLY in every toy (same navy torso, same
// terracotta feet — "the same robot") so the ONLY thing a viewer compares between
// two panels is the BEHAVIOUR. Per-toy differences (which policy, the fall state,
// the forward-distance marker) are carried by the `accent` colour + overlays the
// caller draws, never by the robot's own colours.

// Fixed geometry from QUADRUPED_XML (src/sim/scene.ts) — the shin's foot offset and
// the torso half-extents in the sagittal slice.
const FOOT_LOCAL_Z = -0.13; // foot geom is at (0,0,-0.13) in the shin frame
const TORSO_HX = 0.18; // torso box half-length (x)
const TORSO_HZ = 0.035; // torso box half-height (z)

// leg order matches quadruped_env JOINT_NAMES: FL, FR, HL, HR
const LEGS: ReadonlyArray<{ thigh: string; shin: string; near: boolean }> = [
  { thigh: 'FL_thigh', shin: 'FL_shin', near: true },
  { thigh: 'FR_thigh', shin: 'FR_shin', near: false },
  { thigh: 'HL_thigh', shin: 'HL_shin', near: true },
  { thigh: 'HR_thigh', shin: 'HR_shin', near: false },
];

export interface QuadColors {
  torso: string;
  leg: string;
  foot: string;
  ground: string;
  grid: string;
  paper: string;
  ink: string;
  accent: string; // per-panel signal (walks / stalls / robust / brittle)
}

export const QUAD_COLORS_LIGHT: QuadColors = {
  torso: '#333e6b', // brand navy (XML rgba 0.20 0.24 0.42)
  leg: '#3a4576',
  foot: '#cc5a4c', // terracotta (XML rgba 0.80 0.35 0.30)
  ground: '#c8bc9e',
  grid: 'rgba(200,188,158,0.5)',
  paper: '#fbf9f3',
  ink: '#6d6252',
  accent: '#1f56de',
};

export interface DrawOpts {
  W: number;
  H: number;
  scale: number; // px per metre
  camX: number; // world x mapped to the camera anchor
  colors: QuadColors;
  originFrac?: number; // where camX lands horizontally (0..1); default 0.32
  groundFrac?: number; // where world z=0 lands vertically (0..1); default 0.74
  startX?: number; // world x of the reset line (draws a faint start marker)
  fallen?: boolean; // dim the robot + tint the feet when it has gone down
  label?: string; // small caption drawn top-left inside the panel
}

/** Project + paint one quadruped frame. Pure over (ctx, sim state) — the caller
 *  clears/positions the canvas and draws HUD/overlays around this. */
export function drawQuadruped(ctx: CanvasRenderingContext2D, sim: any, o: DrawOpts): void {
  const { W, H, scale, camX, colors: c } = o;
  const originFrac = o.originFrac ?? 0.32;
  const groundY = H * (o.groundFrac ?? 0.74);
  const originX = W * originFrac;
  const px = (x: number) => originX + (x - camX) * scale;
  const pz = (z: number) => groundY - z * scale; // world +z is screen-up

  // --- background: warm paper + scrolling ground grid (motion made visible) ----
  ctx.fillStyle = c.paper;
  ctx.fillRect(0, 0, W, H);

  // vertical grid lines scroll with the camera so forward travel is legible
  ctx.strokeStyle = c.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  const spacing = 0.25 * scale; // a line every 0.25 m
  const phase = ((camX * scale) % spacing + spacing) % spacing;
  for (let gx = -phase; gx <= W; gx += spacing) {
    ctx.moveTo(gx, 0);
    ctx.lineTo(gx, groundY);
  }
  ctx.stroke();

  // ground
  ctx.strokeStyle = c.ground;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, groundY);
  ctx.lineTo(W, groundY);
  ctx.stroke();
  // ground hatching below the line
  ctx.strokeStyle = c.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let hx = -phase; hx <= W; hx += spacing) {
    ctx.moveTo(hx, groundY);
    ctx.lineTo(hx - 8, groundY + 10);
  }
  ctx.stroke();

  // start line (where this episode began) — a faint accent marker to read distance
  if (o.startX !== undefined) {
    const sx = px(o.startX);
    if (sx > -20 && sx < W + 20) {
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.globalAlpha = 0.5;
      ctx.strokeStyle = c.accent;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(sx, groundY - 6);
      ctx.lineTo(sx, groundY - H * 0.34);
      ctx.stroke();
      ctx.restore();
    }
  }

  const alive = !o.fallen;
  const legOf = (thigh: string, shin: string) => {
    const hip = sim.bodyXpos(thigh) as [number, number, number];
    const knee = sim.bodyXpos(shin) as [number, number, number];
    const sm = sim.bodyXmat(shin) as number[];
    // foot = shin_xpos + shin_R · (0,0,FOOT_LOCAL_Z)
    const footX = knee[0] + sm[2] * FOOT_LOCAL_Z;
    const footZ = knee[2] + sm[8] * FOOT_LOCAL_Z;
    return {
      hip: [px(hip[0]), pz(hip[2])],
      knee: [px(knee[0]), pz(knee[2])],
      foot: [px(footX), pz(footZ)],
    };
  };

  const drawLeg = (thigh: string, shin: string, near: boolean) => {
    const { hip, knee, foot } = legOf(thigh, shin);
    ctx.save();
    ctx.globalAlpha = (near ? 1.0 : 0.42) * (alive ? 1 : 0.55);
    ctx.strokeStyle = c.leg;
    ctx.lineWidth = near ? 5 : 4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(hip[0], hip[1]);
    ctx.lineTo(knee[0], knee[1]);
    ctx.lineTo(foot[0], foot[1]);
    ctx.stroke();
    // foot
    ctx.fillStyle = o.fallen ? c.accent : c.foot;
    ctx.beginPath();
    ctx.arc(foot[0], foot[1], near ? 4.6 : 3.6, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  };

  // far legs first (behind the torso), then torso, then near legs on top
  for (const l of LEGS) if (!l.near) drawLeg(l.thigh, l.shin, false);

  // --- torso box: project its sagittal face corners through the body transform --
  const tp = sim.bodyXpos('torso') as [number, number, number];
  const tm = sim.bodyXmat('torso') as number[]; // row-major world rotation
  // world = R · local + xpos ; local corner (lx, 0, lz)
  const corner = (lx: number, lz: number): [number, number] => {
    const wx = tm[0] * lx + tm[2] * lz + tp[0];
    const wz = tm[6] * lx + tm[8] * lz + tp[2];
    return [px(wx), pz(wz)];
  };
  const cs: [number, number][] = [
    corner(-TORSO_HX, TORSO_HZ),
    corner(TORSO_HX, TORSO_HZ),
    corner(TORSO_HX, -TORSO_HZ),
    corner(-TORSO_HX, -TORSO_HZ),
  ];
  ctx.save();
  ctx.globalAlpha = alive ? 1 : 0.7;
  ctx.fillStyle = c.torso;
  ctx.beginPath();
  cs.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.fill();
  // a small "head" nub on the +x (forward) end so orientation reads
  const headMid = corner(TORSO_HX + 0.03, 0);
  ctx.fillStyle = c.accent;
  ctx.beginPath();
  ctx.arc(headMid[0], headMid[1], 3.6, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  for (const l of LEGS) if (l.near) drawLeg(l.thigh, l.shin, true);

  // panel label
  if (o.label) {
    ctx.save();
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = c.ink;
    ctx.font = '600 12px ui-monospace, monospace';
    ctx.textBaseline = 'top';
    ctx.fillText(o.label, 10, 9);
    ctx.restore();
  }
}
