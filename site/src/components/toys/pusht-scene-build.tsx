/**
 * pusht-scene-build — ch0.2 "Bodies, Joints, and MJCF" concept-toy.
 *
 * THE WELD, FELT. The chapter's whole point is that the T-block is ONE body
 * carrying two geoms — that shared body IS the weld, and it gives the block
 * exactly three planar degrees of freedom. Author the same two geoms as TWO
 * bodies instead and the "T" is a phantom: six DOF, and under a push the halves
 * come apart. This toy is the chapter's `--break split-tee` Break-It made
 * interactive: one toggle (welded ⟷ two bodies) + one push, a live DOF counter
 * (3 → 6) and a live bar–stem gap that only a real weld holds at 0.06 m.
 *
 * Built to the FROZEN CONCEPT-TOY CONTRACT (see PlateIsland.tsx):
 *  1. SSR poster == JS-off fallback. <Poster/> server-renders the nominal welded
 *     scene as a captioned SVG; with JS off that poster is the whole experience
 *     and no WASM is ever fetched. No window/document is touched at module scope.
 *  2. Lazy, hydration-gated sim. createSim + PUSHT_XML are pulled by *dynamic*
 *     import() inside the post-hydration effect only (mounted client:visible, so
 *     the WASM is fetched only when scrolled into view). Canvas hidden until booted.
 *  3. Reuse the primitives verbatim. The welded scene IS playground PUSHT_XML
 *     (byte-synced to curriculum/common/envs/pusht/pusht.xml). The SPLIT scene is
 *     derived from it by swapping the tee body for the two-body variant that
 *     mirrors scene.py's TEE_SPLIT_MJCF exactly — bar body + tee_stem_body, each
 *     with its own planar joint set. Physics via createSim; only the render is
 *     bespoke so entities wear the page's --entity-* hues.
 *  4. Make the invisible visible. The DOF counter (read live off the compiled
 *     model: model.nv − 2 pusher DOF → 3 welded, 6 split) and the bar–stem gap
 *     are the structural truth a top-down picture hides. The split T looks like a
 *     T at rest; the counter says 6 and the push proves it.
 *  5. One control, immediate feedback, default-interesting. Boot WELDED and rigid.
 *     Toggle to two-bodies (counter jumps 3→6) and push to watch the T telescope.
 *  6. Colour discipline. --entity-block/-tee for the T, --entity-pusher for the
 *     pusher, --entity-target for the goal; ONE --signal blue for the live toggle;
 *     --alert red only for the "not rigid" phantom half + verdict.
 *
 * DOF / body facts are sourced, not invented: scene.py notes "the welded T has 3
 * dofs; the split T has 6" (line 186) and meta.yaml records planar vs split state
 * sizes. Here they are read straight off the compiled MjModel, so the number on
 * screen is the one MuJoCo actually built.
 */
import { useEffect, useRef, useState } from "preact/hooks";
import "./pusht-scene-build.css";

// The welded reference scene (T as ONE body, two welded geoms) is the playground
// PUSHT_XML — kept byte-for-byte in sync with curriculum/common/envs/pusht/pusht.xml.
import { PUSHT_XML } from "../../../../playground/src/sim/scene";
import { worldToPx, WORLD_HALF_EXTENT } from "../../../../playground/src/teleop/viewport";

// ---------------------------------------------------------------- scene variants
// The SPLIT scene: the SAME two geoms, but each in its own body with its own
// planar joints — mirrors scene.py's TEE_SPLIT_MJCF. Nothing welds them now, so
// the "T" carries 6 DOF and the halves can move independently. We build it by
// swapping ONLY the tee <body> out of the welded reference (regex is whitespace-
// robust); everything else — walls, target, pusher, actuators — is shared, so the
// two scenes differ by exactly the authoring mistake the chapter warns about.
const SPLIT_TEE_BODY = `<body name="tee" pos="0 0 0.0152">
      <joint name="tee_x"   type="slide" axis="1 0 0" damping="4"    frictionloss="1.2"/>
      <joint name="tee_y"   type="slide" axis="0 1 0" damping="4"    frictionloss="1.2"/>
      <joint name="tee_yaw" type="hinge" axis="0 0 1" damping="0.02" frictionloss="0.006"/>
      <geom name="tee_bar"  type="box" size="0.06 0.015 0.015" pos="0 0 0" rgba="0.45 0.5 0.95 1" mass="0.06"/>
    </body>
    <body name="tee_stem_body" pos="0 -0.06 0.0152">
      <joint name="stem_x"   type="slide" axis="1 0 0" damping="4"    frictionloss="1.2"/>
      <joint name="stem_y"   type="slide" axis="0 1 0" damping="4"    frictionloss="1.2"/>
      <joint name="stem_yaw" type="hinge" axis="0 0 1" damping="0.02" frictionloss="0.006"/>
      <geom name="tee_stem" type="box" size="0.015 0.045 0.015" pos="0 0 0" rgba="0.95 0.5 0.45 1" mass="0.045"/>
    </body>`;

/** Welded reference → two-body split scene (swap the tee body only). */
function buildSplitXml(): string {
  const split = PUSHT_XML.replace(/<body name="tee"[\s\S]*?<\/body>/, SPLIT_TEE_BODY);
  if (split === PUSHT_XML) {
    // Fail closed: if the tee body did not match we would silently ship a welded
    // scene under the "two bodies" label and the DOF counter would lie.
    throw new Error("pusht-scene-build: could not locate the tee body to split");
  }
  return split;
}

type Variant = "welded" | "split";

// ---------------------------------------------------------------------- tunables
const CANVAS_PX = 512;
const TEE_Y0 = 0.04; // rest offset north of the target, so the pusher (south) has room
const PUSHER_Y0 = -0.22; // pusher starts south of the block
const WELD_GAP = 0.06; // the constant bar↔stem centre distance a real weld holds forever
const RIGID_TOL = 0.005; // peak gap deviation under which we still call it rigid
const PUSH_STEPS = 240; // physics steps of a push burst (2.4 s sim @ dt=0.01)
const STEPS_PER_FRAME = 4; // control-step catch-up cap per animation frame

// ------------------------------------------------------------- shared SVG poster
// Square world→px frame identical to the live canvas, so booting causes no reflow.
const POSTER_V = 500;
const POSTER_S = POSTER_V / (2 * WORLD_HALF_EXTENT);
const p2s = (x: number, y: number): [number, number] => [
  POSTER_V / 2 + x * POSTER_S,
  POSTER_V / 2 - y * POSTER_S, // world +y is up
];

/** A PushT "T" in SVG px at (world) centre. Bar 0.12×0.03, stem 0.03×0.09. */
function PosterTee({ x, y, className }: { x: number; y: number; className: string }) {
  const [cx, cy] = p2s(x, y);
  const barW = 0.12 * POSTER_S, barH = 0.03 * POSTER_S;
  const stemW = 0.03 * POSTER_S, stemH = 0.09 * POSTER_S;
  return (
    <g class={className} transform={`translate(${cx.toFixed(1)} ${cy.toFixed(1)})`}>
      <rect x={-barW / 2} y={-barH / 2} width={barW} height={barH} rx={2} />
      {/* stem hangs toward the block's -y (SVG down), matching the MJCF layout */}
      <rect x={-stemW / 2} y={0.06 * POSTER_S - stemH / 2} width={stemW} height={stemH} rx={2} />
    </g>
  );
}

/** The static, captioned poster — SSR output + the JS-off experience. */
function Poster() {
  const [pcx, pcy] = p2s(0, PUSHER_Y0);
  return (
    <svg
      class="sb-poster-svg"
      viewBox={`0 0 ${POSTER_V} ${POSTER_V}`}
      role="img"
      aria-label="Top-down MJCF scene. A magenta T-block — its horizontal bar and vertical stem drawn as one welded piece — sits above a dashed green target T, with the amber cylindrical pusher below it. A badge reads: one body, three degrees of freedom, rigid. With the sim loaded you can toggle the T to two separate bodies (six degrees of freedom) and push it to watch the halves come apart."
    >
      <title>Bodies, joints & MJCF — the weld, felt</title>
      <desc>
        The T-block is ONE body carrying two welded geoms, so it has exactly three
        planar degrees of freedom and moves as a single rigid piece. Author it as
        two bodies instead and it becomes six degrees of freedom that a push pulls
        apart.
      </desc>

      {/* arena + faint graph paper */}
      <rect class="sb-arena" x={2} y={2} width={POSTER_V - 4} height={POSTER_V - 4} rx={6} />
      <g class="sb-grid">
        {Array.from({ length: 9 }, (_, i) => ((i + 1) * POSTER_V) / 10).map((v) => (
          <>
            <line x1={v} y1={2} x2={v} y2={POSTER_V - 2} />
            <line x1={2} y1={v} x2={POSTER_V - 2} y2={v} />
          </>
        ))}
      </g>

      {/* target pose (fixed at origin) — dashed emerald */}
      <PosterTee x={0} y={0} className="sb-target" />
      <text class="sb-target-label" x={p2s(0.085, -0.02)[0]} y={p2s(0.085, -0.02)[1]}>target</text>

      {/* the welded T-block (nominal rest pose) */}
      <PosterTee x={0} y={TEE_Y0} className="sb-tee" />

      {/* the pusher */}
      <g transform={`translate(${pcx.toFixed(1)} ${pcy.toFixed(1)})`}>
        <circle class="sb-pusher-ring" r={0.028 * POSTER_S} />
        <circle class="sb-pusher-core" r={0.015 * POSTER_S} />
      </g>

      {/* the structural readout the picture hides — the weld, made countable */}
      <g class="sb-poster-badge" transform={`translate(${p2s(-0.42, 0.42)[0]} ${p2s(-0.42, 0.42)[1]})`}>
        <rect x={0} y={-16} width={210} height={26} rx={5} />
        <text x={10} y={2}>1 body · 3 DOF · rigid</text>
      </g>
      <text class="sb-poster-hint" x={p2s(-0.42, -0.40)[0]} y={p2s(-0.42, -0.40)[1]}>
        ↻ toggle the weld → two bodies · then push
      </text>
    </svg>
  );
}

// ------------------------------------------------------------------- live island
interface Readout {
  ready: boolean;
  building: boolean;
  pushing: boolean;
  variant: Variant;
  tDof: number;
  bodies: number;
  gap: number;
  maxDev: number;
  rigid: boolean;
  error?: string;
}

type PaletteKey =
  | "--entity-pusher" | "--entity-block" | "--entity-target"
  | "--signal" | "--alert" | "--ink-mute" | "--rule-strong";
const PALETTE_FALLBACK: Record<PaletteKey, string> = {
  "--entity-pusher": "#b0560f",
  "--entity-block": "#a5257d",
  "--entity-target": "#0c7d5f",
  "--signal": "#1f56de",
  "--alert": "#c0362a",
  "--ink-mute": "#6d6252",
  "--rule-strong": "#c8bc9e",
};

const GEOM_ROLE: Record<string, "wall" | "target" | "bar" | "stem" | "pusher"> = {
  wall_n: "wall", wall_s: "wall", wall_e: "wall", wall_w: "wall",
  target_bar: "target", target_stem: "target",
  tee_bar: "bar", tee_stem: "stem", pusher_tip: "pusher",
};

export default function PushTSceneBuild() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const apiRef = useRef<{ toggle: () => void; push: () => void; reset: () => void } | null>(null);
  const [r, setR] = useState<Readout>({
    ready: false, building: true, pushing: false, variant: "welded",
    tDof: 3, bodies: 1, gap: WELD_GAP, maxDev: 0, rigid: true,
  });

  useEffect(() => {
    let disposed = false;
    let raf = 0;
    // hoisted so the cleanup closure can free the current WASM sim on unmount
    let current: { dispose(): void } | null = null;
    const reduceMotion =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

    (async () => {
      try {
        // --- lazy, hydration-gated: the WASM module is fetched only now --------
        const { createSim } = await import("../../../../playground/src/sim/mujoco_sim");

        const canvas = canvasRef.current!;
        canvas.width = CANVAS_PX;
        canvas.height = CANVAS_PX;
        const ctx = canvas.getContext("2d")!;

        // resolve the page design tokens once (canvas can't read CSS vars live)
        const cs = getComputedStyle(canvas);
        const col = (k: PaletteKey) => cs.getPropertyValue(k).trim() || PALETTE_FALLBACK[k];
        const C = {
          pusher: col("--entity-pusher"), block: col("--entity-block"),
          target: col("--entity-target"), signal: col("--signal"),
          alert: col("--alert"), ink: col("--ink-mute"), rule: col("--rule-strong"),
        };

        // --- mutable sim state (refs, not React state — the loop must not re-render)
        let sim: any = null;
        let variant: Variant = "welded";
        let tDof = 3, bodies = 1;
        let barGid = -1, stemGid = -1;
        let roleById = new Map<number, string>();
        let building = false;
        let pushing = false;
        let pushRemaining = 0;
        let maxDev = 0;

        const scale = CANVAS_PX / (2 * WORLD_HALF_EXTENT);

        const gapNow = (): number => {
          if (barGid < 0 || stemGid < 0) return WELD_GAP;
          const gx = sim.data.geom_xpos;
          const dx = gx[barGid * 3] - gx[stemGid * 3];
          const dy = gx[barGid * 3 + 1] - gx[stemGid * 3 + 1];
          return Math.hypot(dx, dy);
        };
        const sampleGap = () => { maxDev = Math.max(maxDev, Math.abs(gapNow() - WELD_GAP)); };

        // set the block + pusher start pose. In SPLIT we seed BOTH bodies to the
        // same y so at rest the halves still read as one T (gap = WELD_GAP): the
        // ONLY tell at rest is the DOF counter — the push is what pulls them apart.
        const seed = () => {
          sim.resetData();
          sim.setJointQpos("tee_y", TEE_Y0);
          if (variant === "split") sim.setJointQpos("stem_y", TEE_Y0);
          sim.setJointQpos("pusher_y", PUSHER_Y0);
          sim.forward();
          maxDev = Math.abs(gapNow() - WELD_GAP);
        };

        const publish = () => {
          if (disposed) return;
          const gap = gapNow();
          setR({
            ready: true, building, pushing, variant, tDof, bodies,
            gap, maxDev, rigid: maxDev < RIGID_TOL,
          });
        };

        // (re)build the sim for a variant, remap ids, seed, publish.
        const build = async (v: Variant) => {
          if (building) return;
          building = true;
          pushing = false; pushRemaining = 0;
          publish();
          const xml = v === "welded" ? PUSHT_XML : buildSplitXml();
          const next: any = await createSim(xml);
          if (disposed) { next.dispose(); return; }
          const old = sim;
          sim = next; current = next; variant = v;
          if (old) old.dispose();

          const m = sim.module, model = sim.model;
          const gid = (name: string) =>
            m.mj_name2id(model, m.mjtObj.mjOBJ_GEOM.value, name);
          roleById = new Map();
          for (const [name, role] of Object.entries(GEOM_ROLE)) {
            const id = gid(name);
            if (id >= 0) roleById.set(id, role);
          }
          barGid = gid("tee_bar"); stemGid = gid("tee_stem");
          // structural truth, read straight off the compiled model: the pusher
          // always contributes exactly 2 slide DOF, so the T-block owns the rest.
          tDof = model.nv - 2;              // welded 5−2=3 · split 8−2=6
          bodies = model.nbody - 3;         // minus world, target, pusher → 1 vs 2

          seed();
          building = false;
          publish();
        };

        // --- bespoke top-down renderer: entities in the page's --entity-* hues.
        // Reads each geom's WORLD pose (geom_xpos / geom_xmat) rather than a single
        // block pose, so a split T that comes apart is drawn coming apart.
        const render = () => {
          if (!sim) return;
          const m = sim.module, model = sim.model, data = sim.data;
          const BOX = m.mjtGeom.mjGEOM_BOX.value;
          const CYL = m.mjtGeom.mjGEOM_CYLINDER.value;
          const gpos = data.geom_xpos, gmat = data.geom_xmat;
          const gtype = model.geom_type, gsize = model.geom_size;

          // arena background (warm paper) + faint graph paper
          ctx.fillStyle = "#fbf9f3";
          ctx.fillRect(0, 0, CANVAS_PX, CANVAS_PX);
          ctx.strokeStyle = "rgba(200,188,158,0.5)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          for (let i = 1; i < 10; i++) {
            const q = (i * CANVAS_PX) / 10;
            ctx.moveTo(q, 0); ctx.lineTo(q, CANVAS_PX);
            ctx.moveTo(0, q); ctx.lineTo(CANVAS_PX, q);
          }
          ctx.stroke();

          const drawBox = (g: number, opts: { color: string; dashed?: boolean; alpha?: number }) => {
            const [px, py] = worldToPx(canvas, gpos[g * 3], gpos[g * 3 + 1]);
            const heading = Math.atan2(gmat[g * 9 + 3], gmat[g * 9]);
            const hx = gsize[g * 3] * scale, hy = gsize[g * 3 + 1] * scale;
            ctx.save();
            ctx.translate(px, py);
            ctx.rotate(-heading);
            if (opts.dashed) {
              ctx.setLineDash([5, 4]);
              ctx.strokeStyle = opts.color;
              ctx.lineWidth = 2;
              ctx.strokeRect(-hx, -hy, 2 * hx, 2 * hy);
            } else {
              ctx.globalAlpha = opts.alpha ?? 1;
              ctx.fillStyle = opts.color;
              ctx.fillRect(-hx, -hy, 2 * hx, 2 * hy);
            }
            ctx.restore();
          };

          // draw the static frame first (walls, target), then the movers
          for (let g = 0; g < model.ngeom; g++) {
            if (gtype[g] !== BOX) continue;
            const role = roleById.get(g);
            if (role === "wall") drawBox(g, { color: C.rule, alpha: 0.55 });
            else if (role === "target") drawBox(g, { color: C.target, dashed: true });
          }
          // the T — bar always magenta; the stem turns ALERT red in split mode to
          // mark the phantom second body (the thing that shouldn't be free).
          for (let g = 0; g < model.ngeom; g++) {
            if (gtype[g] !== BOX) continue;
            const role = roleById.get(g);
            if (role === "bar") drawBox(g, { color: C.block });
            else if (role === "stem")
              drawBox(g, { color: variant === "split" ? C.alert : C.block });
          }
          // pusher (cylinder) — amber, with a faint reach ring
          for (let g = 0; g < model.ngeom; g++) {
            if (gtype[g] !== CYL || roleById.get(g) !== "pusher") continue;
            const [px, py] = worldToPx(canvas, gpos[g * 3], gpos[g * 3 + 1]);
            const rPx = gsize[g * 3] * scale;
            ctx.save();
            ctx.globalAlpha = 0.5; ctx.strokeStyle = C.pusher; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.arc(px, py, rPx * 1.9, 0, Math.PI * 2); ctx.stroke();
            ctx.restore();
            ctx.fillStyle = C.pusher;
            ctx.beginPath(); ctx.arc(px, py, rPx, 0, Math.PI * 2); ctx.fill();
          }
          void C.signal; void C.ink;
        };

        // button API — the whole interaction, keyboard-native via <button>s.
        apiRef.current = {
          toggle: () => { if (!building && !pushing) build(variant === "welded" ? "split" : "welded"); },
          push: () => {
            if (!sim || building || pushing) return;
            if (reduceMotion) {
              // no animation: step to completion in chunks (still sampling the gap
              // so the peak-deviation verdict stays honest), then a single redraw.
              let remaining = PUSH_STEPS;
              while (remaining > 0) {
                const n = Math.min(remaining, 16);
                sim.step([0, 1], n); remaining -= n; sampleGap();
              }
              sim.setCtrl([0, 0]);
              render(); publish();
            } else {
              pushing = true; pushRemaining = PUSH_STEPS; publish();
            }
          },
          reset: () => { if (sim && !building) { pushing = false; pushRemaining = 0; seed(); render(); publish(); } },
        };

        await build("welded");
        render();

        // --- animation + physics loop -----------------------------------------
        let hudMark = 0;
        const nextFrame = () => new Promise<void>((res) => (raf = requestAnimationFrame(() => res())));
        while (!disposed) {
          await nextFrame();
          if (disposed) break;
          if (pushing && pushRemaining > 0 && sim) {
            const n = Math.min(pushRemaining, STEPS_PER_FRAME);
            sim.step([0, 1], n);
            pushRemaining -= n;
            sampleGap();
            if (pushRemaining <= 0) { pushing = false; sim.setCtrl([0, 0]); }
          }
          render();
          const now = performance.now();
          if (now - hudMark >= 100) { hudMark = now; publish(); } // throttle HUD ~10 Hz
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error("[ch0.2 toy] failed", err);
        setR((s) => ({ ...s, error: msg }));
      }
    })();

    return () => { disposed = true; cancelAnimationFrame(raf); if (current) current.dispose(); };
  }, []);

  const failed = !!r.error;
  const split = r.variant === "split";
  const booted = r.ready;
  const busy = !booted || failed || r.building || r.pushing;

  return (
    <div class="sb">
      <figure class="sb-figure" role="group" aria-label="Interactive MJCF weld toy: toggle the T-block between one welded body and two separate bodies, then push it to see whether it stays rigid.">
        {/* SSR poster — the JS-off experience and the pre-boot frame */}
        <div class="sb-poster" hidden={booted}><Poster /></div>

        {/* live MuJoCo-WASM canvas — shown once booted */}
        <canvas ref={canvasRef} class="sb-canvas" hidden={!booted} aria-hidden="true" />

        {/* the invisible-made-visible badge: the structural truth of the scene */}
        {booted && !failed && (
          <div class={`sb-badge ${split ? "sb-badge--split" : ""}`} aria-live="polite">
            <span class="sb-badge-line">
              <b>{r.bodies}</b> {r.bodies === 1 ? "body" : "bodies"}
            </span>
            <span class="sb-badge-sep">·</span>
            <span class="sb-badge-line">
              T-block DOF <b class={split ? "sb-dof-bad" : "sb-dof-ok"}>{r.tDof}</b>
              {split && <span class="sb-phantom"> (+3 phantom)</span>}
            </span>
          </div>
        )}

        {/* live readout — bar↔stem gap + rigidity verdict */}
        {booted && !failed && (
          <div class="sb-hud" aria-hidden="true">
            <div class="sb-hud-row">
              <span class="sb-k">bar–stem gap</span>
              <span class={`sb-v ${r.rigid ? "sb-ok" : "sb-bad"}`}>{r.gap.toFixed(3)} m</span>
            </div>
            <div class="sb-hud-row">
              <span class="sb-k">peak deviation from weld</span>
              <span class={`sb-v ${r.rigid ? "sb-ok" : "sb-bad"}`}>{r.maxDev.toFixed(3)} m</span>
            </div>
            <div class="sb-hud-row">
              <span class="sb-k">verdict</span>
              <span class={`sb-v ${r.rigid ? "sb-ok" : "sb-bad"}`}>
                {r.rigid ? "RIGID · one weld" : "NOT RIGID · came apart"}
              </span>
            </div>
          </div>
        )}

        {/* boot / status line */}
        <div class="sb-status" data-failed={failed} aria-hidden="true">
          {failed ? (
            <span>sim failed — the chapter's scene.py covers this without WASM</span>
          ) : !booted ? (
            <span>booting MuJoCo-WASM…</span>
          ) : r.building ? (
            <span>recompiling the scene…</span>
          ) : r.pushing ? (
            <span>pushing north · watch the {split ? "halves" : "T"}</span>
          ) : (
            <span>{split ? "two bodies · phantom weld" : "one welded body · rigid"}</span>
          )}
        </div>
      </figure>

      <div class="sb-controls">
        <button
          type="button"
          class={`sb-btn sb-btn--toggle ${split ? "is-split" : ""}`}
          role="switch"
          aria-checked={split}
          onClick={() => apiRef.current?.toggle()}
          disabled={!booted || failed || r.building || r.pushing}
        >
          <span class="sb-toggle-label">weld</span>
          <span class="sb-toggle-track"><span class="sb-toggle-knob" /></span>
          <span class="sb-toggle-label">two bodies</span>
        </button>
        <button
          type="button"
          class="sb-btn sb-btn--primary"
          onClick={() => apiRef.current?.push()}
          disabled={busy}
        >
          push it →
        </button>
        <button
          type="button"
          class="sb-btn"
          onClick={() => apiRef.current?.reset()}
          disabled={!booted || failed || r.building || r.pushing}
        >
          reset
        </button>
        <span class="sb-control-note">
          toggle the weld, then push · one welded body stays rigid — two bodies come apart · poster reads with JS off
        </span>
      </div>
    </div>
  );
}
