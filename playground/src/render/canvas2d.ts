// Top-down orthographic canvas-2D renderer. Adequate for the planar PushT
// placeholder; the production playground will decide between this and a
// Three.js 3D view per the graceful-degradation ladder in CLAUDE.md
// (full sim -> reduced render -> video fallback).
import type { Sim } from '../sim/mujoco_sim';
import { WORLD_HALF_EXTENT } from '../teleop/viewport';

export function renderScene(ctx: CanvasRenderingContext2D, sim: Sim): void {
  const { model, data, module } = sim;
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  const scale = w / (2 * WORLD_HALF_EXTENT);

  ctx.clearRect(0, 0, w, h);

  const toPx = (x: number, y: number): [number, number] => [
    w / 2 + x * scale,
    h / 2 - y * scale, // world +y is up
  ];

  const geomXpos = data.geom_xpos;
  const geomXmat = data.geom_xmat;
  const geomType = model.geom_type;
  const geomSize = model.geom_size;
  const geomRgba = model.geom_rgba;

  const BOX = module.mjtGeom.mjGEOM_BOX.value;
  const CYLINDER = module.mjtGeom.mjGEOM_CYLINDER.value;
  const SPHERE = module.mjtGeom.mjGEOM_SPHERE.value;
  const PLANE = module.mjtGeom.mjGEOM_PLANE.value;

  for (let g = 0; g < model.ngeom; g++) {
    const type = geomType[g];
    const r = Math.round(geomRgba[g * 4] * 255);
    const gr = Math.round(geomRgba[g * 4 + 1] * 255);
    const b = Math.round(geomRgba[g * 4 + 2] * 255);
    const a = geomRgba[g * 4 + 3];
    ctx.fillStyle = `rgba(${r}, ${gr}, ${b}, ${a})`;

    const [px, py] = toPx(geomXpos[g * 3], geomXpos[g * 3 + 1]);

    if (type === PLANE) {
      const hx = geomSize[g * 3] * scale;
      const hy = geomSize[g * 3 + 1] * scale;
      ctx.fillRect(px - hx, py - hy, 2 * hx, 2 * hy);
    } else if (type === BOX) {
      // Heading from the world-frame rotation matrix (row-major 3x3):
      // first column is the body x-axis projected into world.
      const heading = Math.atan2(geomXmat[g * 9 + 3], geomXmat[g * 9]);
      const hx = geomSize[g * 3] * scale;
      const hy = geomSize[g * 3 + 1] * scale;
      ctx.save();
      ctx.translate(px, py);
      ctx.rotate(-heading);
      ctx.fillRect(-hx, -hy, 2 * hx, 2 * hy);
      ctx.restore();
    } else if (type === CYLINDER || type === SPHERE) {
      ctx.beginPath();
      ctx.arc(px, py, geomSize[g * 3] * scale, 0, Math.PI * 2);
      ctx.fill();
    }
    // Other geom types are not needed by the placeholder scene.
  }

  // Goal site (visual only).
  const siteXpos = data.site_xpos;
  const siteSize = model.site_size;
  for (let s = 0; s < model.nsite; s++) {
    const [px, py] = toPx(siteXpos[s * 3], siteXpos[s * 3 + 1]);
    ctx.beginPath();
    ctx.arc(px, py, siteSize[s * 3] * scale, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(102, 204, 102, 0.25)';
    ctx.fill();
  }
}
