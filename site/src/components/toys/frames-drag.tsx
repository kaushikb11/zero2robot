/**
 * frames-drag — ch0.3 "Spatial Reasoning Without Tears" concept-toy.
 *
 * The LINKED-REPRESENTATION showcase: drag the tee coordinate frame and watch
 * three views of the SAME rigid transform move together — the frame's RGB basis
 * arrows, its quaternion [w, x, y, z], and the pusher expressed in the frame
 * (world→tee inverse ∘ pusher). Then flip the quaternion convention wxyz → xyzw
 * and watch the axes and the number swing WRONG with no error message — the
 * chapter's Break It #1, felt. A second toggle reverses the composition order so
 * the translation diverges while the rotation looks unchanged (Break It #2).
 *
 * This toy follows the FROZEN CONCEPT-TOY CONTRACT documented in PlateIsland.tsx
 * (SSR poster = JS-off fallback in one square figure so booting causes no reflow;
 * ONE control + live readouts + default-interesting; colour discipline; keyboard
 * + reduced-motion accessibility). Unlike ch1.1 it needs NO MuJoCo-WASM — it is
 * pure quaternion/frame math + inline SVG, so it hydrates an interactive SVG with
 * no dynamic WASM import. The math below is the chapter's transforms.py toolkit
 * re-implemented from scratch in TypeScript, SAME wxyz convention. Validated
 * against transforms.py (seed-0 chapter case): pusher_in_tee = [0.167228,
 * 0.030575, 0]; the xyzw break leaves 0.0973 m of silent error (meta: 0.097) and
 * the reversed compose order leaves 0.0912 m (meta ex1: 0.091).
 */
import { useEffect, useRef, useState } from "preact/hooks";
import "./frames-drag.css";

// ---------------------------------------------------------------- from-scratch math
// Mirror of curriculum/phase0_foundations/ch0.3_transforms/transforms.py, wxyz.
type Quat = [number, number, number, number]; // [w, x, y, z]
type Vec3 = [number, number, number];

/** Hamilton product — composes rotations, NOT componentwise, NOT commutative. */
function quatMultiply(a: Quat, b: Quat): Quat {
  const [w1, x1, y1, z1] = a;
  const [w2, x2, y2, z2] = b;
  return [
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, // w
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, // x
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, // y
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, // z
  ];
}
/** For a unit quaternion the conjugate is the inverse rotation. */
const quatConjugate = ([w, x, y, z]: Quat): Quat => [w, -x, -y, -z];

/** The quaternion as a 3x3 rotation matrix (rows). Columns are the images of the
 *  basis vectors e_x, e_y, e_z — i.e. where the frame's axes point in world. */
function quatToMatrix([w, x, y, z]: Quat): number[][] {
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}
/** Rotate a vector: sandwich q * (0,v) * conj(q), read off the vector part.
 *  Feed this a wrong-ORDER quaternion and you get Break It #1. */
function rotateVector(q: Quat, v: Vec3): Vec3 {
  const pure: Quat = [0, v[0], v[1], v[2]];
  const r = quatMultiply(quatMultiply(q, pure), quatConjugate(q));
  return [r[1], r[2], r[3]];
}
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const neg = (a: Vec3): Vec3 => [-a[0], -a[1], -a[2]];

// A rigid Frame: rotation (unit quaternion) + translation, read "world_from_x".
interface Frame { rotation: Quat; translation: Vec3; }
const transformPoint = (f: Frame, p: Vec3): Vec3 => add(rotateVector(f.rotation, p), f.translation);
const compose = (a: Frame, b: Frame): Frame => ({
  rotation: quatMultiply(a.rotation, b.rotation),
  translation: transformPoint(a, b.translation),
});
const inverse = (f: Frame): Frame => {
  const ir = quatConjugate(f.rotation);
  return { rotation: ir, translation: neg(rotateVector(ir, f.translation)) };
};
/** A planar rotation about +z — the only rotation a PushT block makes. */
const yawQuaternion = (yaw: number): Quat => [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
const norm3 = (a: Vec3): number => Math.hypot(a[0], a[1], a[2]);

// ---------------------------------------------------------------- scene constants
const V = 500;                 // square SVG viewBox (px)
const WORLD_HALF = 0.4;        // ± metres mapped to the box edges
const S = V / (2 * WORLD_HALF); // metres → px
const AXIS_LEN = 0.13;         // visual length of a basis arrow (m)
const LONG_AXIS: Vec3 = [0.06, 0, 0]; // the tee's long bar, in its own frame (chapter's probe)
const GRAB_M = 0.06;           // pointer→handle grab radius (m)
const NUDGE_M = 0.02;          // arrow-key translate step (m)
const YAW_STEP = Math.PI / 24; // keyboard rotate step (7.5°)
const TRANS_BOUND = 0.3;       // keep the frame origin inside the arena

// Default-interesting start: correct convention, frame turned a little (so the
// quaternion has a real w AND z to read), pusher off to the upper-right.
const DEF_TX = 0.1, DEF_TY = -0.04, DEF_YAW = 0.45;
const PUSHER_WORLD: Vec3 = [0.22, 0.12, 0];

const w2s = (x: number, y: number): [number, number] => [V / 2 + x * S, V / 2 - y * S];
const fmt = (n: number, d = 3): string => (Object.is(n, -0) ? 0 : n).toFixed(d);

// ---------------------------------------------------------------- derived scene
interface Scene {
  q: Quat;              // the correct wxyz tee rotation
  renderQuat: Quat;     // what actually gets fed to the routines (broken under xyzw)
  worldFromTee: Frame;  // uses renderQuat, so the break flows into every readout
  pusherInTee: Vec3;    // world→tee inverse ∘ pusher — the chapter's whole point
  exTip: [number, number]; eyTip: [number, number]; ezTip: [number, number]; // axis tips (px)
  correctExTip: [number, number]; // where x SHOULD point (px) — for the error vector
  origin: [number, number];       // tee origin (px)
  silentErr: number;    // ‖correct − shown‖ of the long axis (m) — Break It #1, measured
  composeGap: number;   // ‖A∘B − B∘A‖ translation (m) — Break It #2, measured
  swappedPx: [number, number]; // the reversed-order pusher reconstruction (px)
}

function buildScene(tx: number, ty: number, yaw: number, xyzw: boolean): Scene {
  const q = yawQuaternion(yaw);
  // wxyz → xyzw is the classic silent swap: q[[1,2,3,0]] fed as if it were wxyz.
  const renderQuat: Quat = xyzw ? [q[1], q[2], q[3], q[0]] : q;
  const worldFromTee: Frame = { rotation: renderQuat, translation: [tx, ty, 0] };
  const pusherInTee = transformPoint(inverse(worldFromTee), PUSHER_WORLD);

  const axisTip = (Q: Quat, col: number): [number, number] => {
    const R = quatToMatrix(Q);
    const basis: Vec3 = [R[0][col], R[1][col], R[2][col]];         // image of e_col
    return w2s(tx + basis[0] * AXIS_LEN, ty + basis[1] * AXIS_LEN); // top-down: drop z
  };
  // Break It #1, made physical: how far the long axis lands from the truth.
  const silentErr = norm3(add(rotateVector(q, LONG_AXIS), neg(rotateVector(renderQuat, LONG_AXIS))));

  // Break It #2 — compose order. Isolated from the convention toggle (uses the
  // correct rotation) so the two bugs read independently, exactly as the chapter
  // measures them: world_from_tee ∘ tee_from_pusher vs the reversed product.
  const teeFromPusher: Frame = { rotation: yawQuaternion(0.7), translation: pusherInTee };
  const worldFromTeeCorrect: Frame = { rotation: q, translation: [tx, ty, 0] };
  const correct = compose(worldFromTeeCorrect, teeFromPusher).translation; // == PUSHER_WORLD
  const swapped = compose(teeFromPusher, worldFromTeeCorrect).translation;  // wrong order
  const composeGap = norm3(add(correct, neg(swapped)));

  return {
    q, renderQuat, worldFromTee, pusherInTee,
    exTip: axisTip(renderQuat, 0), eyTip: axisTip(renderQuat, 1), ezTip: axisTip(renderQuat, 2),
    correctExTip: axisTip(q, 0),
    origin: w2s(tx, ty),
    silentErr, composeGap,
    swappedPx: w2s(swapped[0], swapped[1]),
  };
}

// ---------------------------------------------------------------- SVG helpers
function Arrow({ o, tip, cls, head = 9 }: { o: [number, number]; tip: [number, number]; cls: string; head?: number }) {
  const dx = tip[0] - o[0], dy = tip[1] - o[1];
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len;
  const bx = tip[0] - ux * head, by = tip[1] - uy * head; // base of the head
  const hw = head * 0.55;
  const pts = `${tip[0].toFixed(1)},${tip[1].toFixed(1)} ${(bx - uy * hw).toFixed(1)},${(by + ux * hw).toFixed(1)} ${(bx + uy * hw).toFixed(1)},${(by - ux * hw).toFixed(1)}`;
  return (
    <g class={cls}>
      <line x1={o[0].toFixed(1)} y1={o[1].toFixed(1)} x2={bx.toFixed(1)} y2={by.toFixed(1)} />
      <polygon points={pts} />
    </g>
  );
}

/** Faint instrument graph paper + arena border. */
function Grid() {
  return (
    <>
      <rect class="fd-arena" x={2} y={2} width={V - 4} height={V - 4} rx={6} />
      <g class="fd-grid">
        {Array.from({ length: 9 }, (_, i) => ((i + 1) * V) / 10).map((v) => (
          <>
            <line x1={v} y1={2} x2={v} y2={V - 2} />
            <line x1={2} y1={v} x2={V - 2} y2={v} />
          </>
        ))}
      </g>
    </>
  );
}

/** The whole scene as SVG children — shared by the SSR poster and the live island
 *  so booting causes no reflow. `s` is the derived geometry; `xyzw` toggles the
 *  wrong-convention skin; handles/labels differ only in the wrapper. */
function SceneGraphics({ s, xyzw, composeBA }: { s: Scene; xyzw: boolean; composeBA: boolean }) {
  const [ox, oy] = s.origin;
  const [wpx, wpy] = w2s(PUSHER_WORLD[0], PUSHER_WORLD[1]);
  const worldO = w2s(0, 0);
  return (
    <>
      <Grid />

      {/* world frame (identity) — the fixed reference, drawn faint */}
      <g class="fd-world">
        <Arrow o={worldO} tip={w2s(AXIS_LEN, 0)} cls="fd-ax fd-ax-x fd-ref" head={7} />
        <Arrow o={worldO} tip={w2s(0, AXIS_LEN)} cls="fd-ax fd-ax-y fd-ref" head={7} />
        <circle class="fd-origin-ref" cx={worldO[0]} cy={worldO[1]} r={3.5} />
        <text class="fd-lbl fd-lbl-ref" x={worldO[0] + 8} y={worldO[1] + 16}>world</text>
      </g>

      {/* pusher — a fixed world point (amber). Its world coords never change;
          its tee-frame coords (the readout) do, as you drag the frame. */}
      <g class="fd-pusher">
        <circle class="fd-pusher-ring" cx={wpx} cy={wpy} r={0.03 * S} />
        <circle class="fd-pusher-core" cx={wpx} cy={wpy} r={0.014 * S} />
        <text class="fd-lbl fd-lbl-pusher" x={wpx + 10} y={wpy - 8}>pusher</text>
      </g>

      {/* Break It #2 ghost: the reversed-order reconstruction of the pusher.
          Shown only when composeBA — it sits OFF the pusher by ~0.09 m while the
          rotation looks unchanged (translation diverges, the subtle lesson). */}
      {composeBA && (
        <g class="fd-ghost">
          <line class="fd-ghost-link" x1={wpx} y1={wpy} x2={s.swappedPx[0]} y2={s.swappedPx[1]} />
          <circle class="fd-ghost-dot" cx={s.swappedPx[0]} cy={s.swappedPx[1]} r={0.014 * S} />
          <text class="fd-lbl fd-lbl-ghost" x={s.swappedPx[0] + 9} y={s.swappedPx[1] + 4}>B∘A</text>
        </g>
      )}

      {/* the tee frame — the draggable rigid transform (RGB basis arrows).
          Under xyzw the arrows visibly swing wrong; the z-axis (blue) leaves the
          plane and projects INTO it, the loudest tell. */}
      <g class={`fd-tee ${xyzw ? "fd--wrong" : ""}`}>
        {/* the silent-error vector: from where x lands to where it SHOULD land */}
        {xyzw && (
          <line class="fd-errvec" x1={s.exTip[0]} y1={s.exTip[1]} x2={s.correctExTip[0]} y2={s.correctExTip[1]} />
        )}
        <Arrow o={s.origin} tip={s.ezTip} cls="fd-ax fd-ax-z" />
        <Arrow o={s.origin} tip={s.eyTip} cls="fd-ax fd-ax-y" />
        <Arrow o={s.origin} tip={s.exTip} cls="fd-ax fd-ax-x" />
        <circle class="fd-origin" cx={ox} cy={oy} r={0.02 * S} />
        <text class="fd-lbl fd-lbl-tee" x={ox - 8} y={oy + 24}>tee</text>
      </g>
    </>
  );
}

// ---------------------------------------------------------------- the toy
type DragMode = null | "translate" | "rotate";

export default function FramesDrag() {
  const svgRef = useRef<SVGSVGElement>(null);
  const dragRef = useRef<{ mode: DragMode; pointerId: number }>({ mode: null, pointerId: -1 });
  const [booted, setBooted] = useState(false);
  const [tx, setTx] = useState(DEF_TX);
  const [ty, setTy] = useState(DEF_TY);
  const [yaw, setYaw] = useState(DEF_YAW);
  const [xyzw, setXyzw] = useState(false);      // Break It #1: quaternion convention
  const [composeBA, setComposeBA] = useState(false); // Break It #2: compose order
  const [mode, setMode] = useState<DragMode>(null);

  // Client-only: reveal the interactive island (the poster is the JS-off fallback).
  useEffect(() => { setBooted(true); }, []);

  const s = buildScene(tx, ty, yaw, xyzw);
  const q = s.renderQuat; // the quaternion actually fed to the routines

  const clamp = (v: number) => Math.max(-TRANS_BOUND, Math.min(TRANS_BOUND, v));
  const reset = () => { setTx(DEF_TX); setTy(DEF_TY); setYaw(DEF_YAW); setXyzw(false); setComposeBA(false); };

  // pointer → world (m), via the live SVG bounding box (client-only)
  const toWorld = (clientX: number, clientY: number): [number, number] => {
    const el = svgRef.current!;
    const r = el.getBoundingClientRect();
    const px = ((clientX - r.left) / r.width) * V;
    const py = ((clientY - r.top) / r.height) * V;
    return [(px - V / 2) / S, (V / 2 - py) / S];
  };

  const onPointerDown = (e: PointerEvent) => {
    const [wx, wy] = toWorld(e.clientX, e.clientY);
    const dOrigin = Math.hypot(wx - tx, wy - ty);
    // the rotate knob rides at the x-axis tip
    const knobX = tx + Math.cos(yaw) * AXIS_LEN, knobY = ty + Math.sin(yaw) * AXIS_LEN;
    const dKnob = Math.hypot(wx - knobX, wy - knobY);
    const pick: DragMode = dKnob < dOrigin && dKnob < GRAB_M * 1.6 ? "rotate"
      : dOrigin < GRAB_M * 2 ? "translate" : null;
    if (!pick) return;
    dragRef.current = { mode: pick, pointerId: e.pointerId };
    setMode(pick);
    svgRef.current?.setPointerCapture(e.pointerId);
    e.preventDefault();
    applyDrag(pick, wx, wy);
  };
  const applyDrag = (m: DragMode, wx: number, wy: number) => {
    if (m === "translate") { setTx(clamp(wx)); setTy(clamp(wy)); }
    else if (m === "rotate") { setYaw(Math.atan2(wy - ty, wx - tx)); }
  };
  const onPointerMove = (e: PointerEvent) => {
    if (!dragRef.current.mode) return;
    const [wx, wy] = toWorld(e.clientX, e.clientY);
    applyDrag(dragRef.current.mode, wx, wy);
  };
  const onPointerUp = (e: PointerEvent) => {
    if (dragRef.current.pointerId === e.pointerId) {
      dragRef.current = { mode: null, pointerId: -1 };
      setMode(null);
      try { svgRef.current?.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    }
  };

  const onKeyDown = (e: KeyboardEvent) => {
    const trans: Record<string, [number, number]> = {
      ArrowUp: [0, NUDGE_M], ArrowDown: [0, -NUDGE_M], ArrowLeft: [-NUDGE_M, 0], ArrowRight: [NUDGE_M, 0],
    };
    if (e.key in trans) {
      e.preventDefault();
      const [dx, dy] = trans[e.key];
      setTx((v) => clamp(v + dx)); setTy((v) => clamp(v + dy)); setMode("translate");
    } else if (e.key === "," || e.key === "[") { e.preventDefault(); setYaw((v) => v + YAW_STEP); setMode("rotate"); }
    else if (e.key === "." || e.key === "]") { e.preventDefault(); setYaw((v) => v - YAW_STEP); setMode("rotate"); }
    else if (e.key === "r" || e.key === "R") reset();
    else if (e.key === "b" || e.key === "B") setXyzw((v) => !v);
    else if (e.key === "c" || e.key === "C") setComposeBA((v) => !v);
  };

  const yawDeg = ((yaw * 180) / Math.PI);

  return (
    <div class="fd">
      <div class="fd-stage">
        <figure
          class={`fd-figure ${xyzw ? "fd-figure--wrong" : ""}`}
          data-mode={mode ?? "idle"}
          tabIndex={booted ? 0 : -1}
          role={booted ? "application" : "img"}
          aria-label={
            booted
              ? "Interactive coordinate-frame toy. Drag the magenta tee frame's origin to move it, or drag the tip of its red x-axis to rotate it; or focus here and use arrow keys to move, comma/period to rotate. The frame's quaternion, the pusher's coordinates in the frame, and the code update together. Press B to flip the quaternion convention from wxyz to xyzw and watch the axes point wrong; press C to reverse the composition order; press R to reset."
              : "A top-down coordinate diagram: a faint world frame at the origin, a magenta tee frame turned to the upper right with red, green and blue basis arrows, and an amber pusher point. With JavaScript enabled you can drag the tee frame and watch its quaternion and the pusher's coordinates in the frame update, then flip wxyz to xyzw to see it break."
          }
          onKeyDown={booted ? onKeyDown : undefined}
        >
          <svg
            ref={svgRef}
            class="fd-svg"
            viewBox={`0 0 ${V} ${V}`}
            role="img"
            aria-hidden={booted ? "true" : undefined}
            onPointerDown={booted ? onPointerDown : undefined}
            onPointerMove={booted ? onPointerMove : undefined}
            onPointerUp={booted ? onPointerUp : undefined}
            onPointerCancel={booted ? onPointerUp : undefined}
          >
            {!booted && <title>Coordinate frames — drag the tee frame</title>}
            <SceneGraphics s={s} xyzw={xyzw} composeBA={composeBA} />

            {/* the one LIVE handle: a signal-blue grab halo on the tee origin */}
            {booted && (
              <g class="fd-handles" data-mode={mode ?? "idle"}>
                <circle class="fd-halo" cx={s.origin[0]} cy={s.origin[1]} r={GRAB_M * S} />
                <circle
                  class="fd-knob"
                  cx={w2s(tx + Math.cos(yaw) * AXIS_LEN, ty + Math.sin(yaw) * AXIS_LEN)[0]}
                  cy={w2s(tx + Math.cos(yaw) * AXIS_LEN, ty + Math.sin(yaw) * AXIS_LEN)[1]}
                  r={7}
                />
              </g>
            )}

            {/* JS-off affordance printed into the SSR poster only */}
            {!booted && (
              <text class="fd-drag-hint" x={s.origin[0] - 4} y={s.origin[1] - GRAB_M * S - 10}>
                drag the frame →
              </text>
            )}
          </svg>

          {/* live readouts — the linked numbers (poster ships the nominal state) */}
          <div class="fd-hud" aria-hidden="true">
            <div class="fd-hud-row">
              <span class="fd-k">quaternion [w x y z]{xyzw ? " · fed as xyzw" : ""}</span>
              <span class={`fd-v ${xyzw ? "fd-bad" : ""}`}>
                [{fmt(q[0], 2)} {fmt(q[1], 2)} {fmt(q[2], 2)} {fmt(q[3], 2)}]
              </span>
            </div>
            <div class="fd-hud-row">
              <span class="fd-k">pusher_in_tee (m)</span>
              <span class={`fd-v ${xyzw ? "fd-bad" : ""}`}>
                [{fmt(s.pusherInTee[0])} {fmt(s.pusherInTee[1])} {fmt(s.pusherInTee[2])}]
              </span>
            </div>
            <div class="fd-hud-row">
              <span class="fd-k">tee_yaw</span>
              <span class="fd-v">{fmt(yawDeg, 1)}°</span>
            </div>
            {xyzw && (
              <div class="fd-hud-row">
                <span class="fd-k">orientation error (silent)</span>
                <span class="fd-v fd-bad">{fmt(s.silentErr)} m ▲</span>
              </div>
            )}
            {composeBA && (
              <div class="fd-hud-row">
                <span class="fd-k">compose gap A∘B vs B∘A</span>
                <span class="fd-v fd-bad">{fmt(s.composeGap)} m ▲</span>
              </div>
            )}
          </div>

          <div class="fd-status" data-wrong={xyzw || composeBA} aria-hidden="true">
            {xyzw ? (
              <span>xyzw fed to a wxyz routine — a wrong answer, no error message</span>
            ) : composeBA ? (
              <span>B∘A — rotation looks the same, translation diverged</span>
            ) : (
              <span>wxyz ✓ · pusher fixed in world · its tee-frame coords track the drag</span>
            )}
          </div>
        </figure>

        {/* the code representation — highlights the line the drag is moving, and
            turns red on the wrong-convention feed (LINKED to the frame + numbers) */}
        <pre class="fd-code" aria-hidden="true">
          <code>
            <span class="fd-c-line" data-active={mode === "rotate" || mode === "translate"}>
              world_from_tee = Frame(
              <span class="fd-c-tok" data-active={mode === "rotate"}>yaw_quaternion(θ)</span>,{" "}
              <span class="fd-c-tok" data-active={mode === "translate"}>t</span>)
            </span>
            {"\n"}
            <span class="fd-c-line">
              pusher_in_tee = world_from_tee.<span class="fd-c-tok" data-wrong={xyzw}>inverse()</span>
            </span>
            {"\n"}
            <span class="fd-c-line">
              {"              "}.transform_point(pusher_world)
            </span>
            {xyzw && (
              <span class="fd-c-annot">{"\n"}# inverse() rotates with q[[1,2,3,0]] — xyzw fed to a wxyz routine</span>
            )}
          </code>
        </pre>
      </div>

      {/* controls — JS-only (the poster reads without them) */}
      <div class="fd-controls">
        <button
          type="button"
          class={`fd-btn ${xyzw ? "fd-btn--alert" : "fd-btn--primary"}`}
          aria-pressed={xyzw}
          onClick={() => setXyzw((v) => !v)}
          disabled={!booted}
        >
          {xyzw ? "convention: xyzw ✗ — fix it" : "flip convention: wxyz ✓ → xyzw ✗"}
        </button>
        <button
          type="button"
          class={`fd-btn ${composeBA ? "fd-btn--alert" : ""}`}
          aria-pressed={composeBA}
          onClick={() => setComposeBA((v) => !v)}
          disabled={!booted}
        >
          {composeBA ? "compose: B∘A ✗" : "compose order: A∘B → B∘A"}
        </button>
        <button type="button" class="fd-btn" onClick={reset} disabled={!booted}>
          reset
        </button>
        <span class="fd-control-note">
          drag the frame · arrow-keys move · , . rotate · poster reads with JS off
        </span>
      </div>

      {/* keyboard/SR live summary — discrete on keyboard use, no motion needed */}
      <p class="fd-sr" aria-live="polite">
        {xyzw
          ? `Wrong convention: quaternion fed as xyzw, orientation error ${fmt(s.silentErr)} metres.`
          : `pusher in tee frame: ${fmt(s.pusherInTee[0])}, ${fmt(s.pusherInTee[1])} metres; yaw ${fmt(yawDeg, 0)} degrees.`}
      </p>
    </div>
  );
}
