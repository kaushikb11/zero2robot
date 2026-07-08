/**
 * ch4.2 "Corrections (DAgger)" — the LIVE recovery toy (`demo: dagger_pusht_live`,
 * also stacked under `pusht_dagger_recovery` beneath the recorded recovery-curve).
 *
 * The interactive payoff the chapter's follow_up stubbed out: a REAL dagger.onnx
 * (the BEST DAgger round, obs[10]->action[2], contract v1) drives a REAL
 * MuJoCo-WASM PushT sim. The visitor drags the T-block to a FAR start — the
 * covariate-shift territory the narrow BC demos never covered (block-to-goal
 * > r_max = 0.13) — and watches the DAgger policy RECOVER: it makes progress from
 * exactly the far starts that broke the ch1.1 BC clone (which committed confident
 * wrong strokes there). That is DAgger's whole claim made live — "the policy learns
 * to recover from its own mistakes, because its on-policy corrections cover the
 * states it actually visits."
 *
 * HONEST framing, straight from ch4.2's finalized prose (do not overclaim):
 *   - DAgger recovers the COVARIATE-SHIFT LOSS. It is NOT a great PushT policy: a
 *     memoryless clone of a stateful expert tops out ~0.2 success (round 3 = 0.215
 *     vs BC 0.065 at seed 0). It makes progress from far starts; it will not land
 *     every episode. The readouts say "recovering", never "solved every time".
 *   - Select the BEST round, not the last (Ross et al.): this ships round 3, the
 *     seed-0 peak; round 4 regressed to 0.085. The control note names this.
 *   - DAgger needs an INTERACTIVE, queryable expert. The scripted labeler here is
 *     the offline stand-in for your teleop hand; this live toy is that mechanism.
 *
 * Built to the FROZEN CONCEPT-TOY CONTRACT at the top of ../PlateIsland.tsx:
 *   1. SSR poster == the JS-off fallback (<Poster/> is pure static JSX).
 *   2. Lazy, hydration-gated sim: MuJoCo-WASM + onnxruntime pulled by *dynamic*
 *      import() inside the post-hydration effect (mounted client:visible).
 *   3. Reuse the primitives verbatim: createSim + PUSHT_XML + BrowserPushTEnv +
 *      buildObs + loadPolicy through the fail-closed assertDrivesPushT gate. The
 *      driver loop mirrors dagger.py's eval:
 *        obs = env.obs(); action = await policy.act(obs); env.step(action)
 *   4. Make the invisible visible: the narrow BC-practice region (dashed circle,
 *      r_max = 0.13) is the training coverage; the start-distance meter reads how
 *      far past it the block is; pos_err -> target ticks down as DAgger recovers.
 *   5. ONE control (drag the block to a far start), immediate feedback,
 *      default-interesting (boots in-distribution, driving). Keyboard + button too.
 *   6. Colour discipline: --entity-* for entities, ONE --signal blue for the drag
 *      handle, --alert red for the far (covariate-shift) readout, neutral ink map.
 */
import "./DaggerPushtLive.css";
import { useEffect, useRef, useState } from "preact/hooks";

const MODEL_URL = "/models/dagger.onnx";
const CANVAS_PX = 512;
const WORLD_HALF = 0.45; // matches viewport.WORLD_HALF_EXTENT
const MAX_CONTROL_STEPS_PER_FRAME = 4; // mirror main.ts: cap control-step catch-up
const R_PRACTICE = 0.13; // dagger.py --r_max: BC demos only START within this of the goal
const FAR_START = 0.24; // env's far spawn edge (annulus 0.10..0.24) — the covariate-shift start
const DRAG_BOUND = 0.35; // keep the dragged block inside the walls (±0.41)
const GRAB_RADIUS = 0.1; // m — pointer-to-block distance that starts a drag
const NUDGE_STEP = 0.03; // m — arrow-key block nudge
const METER_FULL = 0.30; // m of start-distance that fills the meter (past FAR_START)

/** block-to-goal distance (goal is the origin) — the covariate-shift axis. */
const distFromGoal = (x: number, y: number) => Math.hypot(x, y);

// ---------------------------------------------------------- shared SSR poster
// Pure static JSX (no window/document) — the JS-off experience + pre-boot frame.
// Same ±WORLD_HALF top-down frame as the live canvas, so booting causes no reflow.
const POSTER_V = 500;
const POSTER_S = POSTER_V / (2 * WORLD_HALF);
const w2s = (x: number, y: number): [number, number] => [
  POSTER_V / 2 + x * POSTER_S,
  POSTER_V / 2 - y * POSTER_S,
];

/** A PushT "T" in SVG px at (world) center/yaw. Bar 0.12×0.03, stem 0.03×0.09. */
function PosterTee({ x, y, yawDeg, className }: { x: number; y: number; yawDeg: number; className: string }) {
  const [cx, cy] = w2s(x, y);
  const barW = 0.12 * POSTER_S, barH = 0.03 * POSTER_S;
  const stemW = 0.03 * POSTER_S, stemH = 0.09 * POSTER_S;
  return (
    <g class={className} transform={`translate(${cx.toFixed(1)} ${cy.toFixed(1)}) rotate(${-yawDeg})`}>
      <rect x={-barW / 2} y={-barH / 2} width={barW} height={barH} rx={2} />
      <rect x={-stemW / 2} y={0.06 * POSTER_S - stemH / 2} width={stemW} height={stemH} rx={2} />
    </g>
  );
}

function Poster() {
  // boot the poster at a FAR start (block past the practice region) — the aha the
  // toy is about: the DAgger policy recovers from here.
  const blockX = 0.18, blockY = 0.15;
  const pusherX = 0.25, pusherY = 0.21;
  const [pcx, pcy] = w2s(pusherX, pusherY);
  const [bcx, bcy] = w2s(blockX, blockY);
  const rPx = R_PRACTICE * POSTER_S;
  return (
    <svg
      class="dg-poster-svg"
      viewBox={`0 0 ${POSTER_V} ${POSTER_V}`}
      role="img"
      aria-label="Top-down PushT arena. A dashed circle marks the narrow region the behavior-cloning demonstrations practiced in; the magenta T-block sits outside it, at a far start, near the amber pusher, with the green dashed target pose at the center. With JavaScript on, the DAgger policy — trained on corrections collected from exactly these far starts — drives the block back toward the target, recovering from a start the behavior-cloning clone could not."
    >
      <title>DAgger corrections — recovery from covariate shift, live on PushT</title>
      <desc>
        The dashed circle is the narrow region the BC demos started in (block-to-goal
        &lt; 0.13 m). Behavior cloning never saw the far starts outside it and fails
        there. DAgger relabels the states the policy visits with the expert's action,
        so it covers those far starts and recovers — it is a modest reactive clone
        (~0.2 success), not a great PushT policy.
      </desc>

      <rect class="dg-arena" x={2} y={2} width={POSTER_V - 4} height={POSTER_V - 4} rx={6} />
      <g class="dg-grid">
        {Array.from({ length: 9 }, (_, i) => ((i + 1) * POSTER_V) / 10).map((v) => (
          <>
            <line x1={v} y1={2} x2={v} y2={POSTER_V - 2} />
            <line x1={2} y1={v} x2={POSTER_V - 2} y2={v} />
          </>
        ))}
      </g>

      {/* the narrow BC-practice region — the training coverage, made visible */}
      <circle class="dg-practice" cx={POSTER_V / 2} cy={POSTER_V / 2} r={rPx} />
      <circle class="dg-practice-ring" cx={POSTER_V / 2} cy={POSTER_V / 2} r={rPx} />
      <text class="dg-practice-label" x={w2s(-0.12, -0.13)[0]} y={w2s(-0.12, -0.13)[1]}>where BC practiced</text>

      <PosterTee x={0} y={0} yawDeg={0} className="dg-target" />
      <text class="dg-target-label" x={w2s(0.03, 0.02)[0]} y={w2s(0.03, 0.02)[1]}>target</text>
      <PosterTee x={blockX} y={blockY} yawDeg={-24} className="dg-tee" />
      <g transform={`translate(${pcx.toFixed(1)} ${pcy.toFixed(1)})`}>
        <circle class="dg-pusher-ring" r={0.028 * POSTER_S} />
        <circle class="dg-pusher-core" r={0.015 * POSTER_S} />
      </g>
      {/* the drag affordance — the one LIVE handle (signal blue) */}
      <g transform={`translate(${bcx.toFixed(1)} ${bcy.toFixed(1)})`}>
        <circle class="dg-halo" r={0.1 * POSTER_S} />
        <text class="dg-drag-label" x={0.11 * POSTER_S} y={-0.09 * POSTER_S}>drag to a far start →</text>
      </g>
      <text class="dg-poster-hint" x={POSTER_V / 2} y={POSTER_V - 22}>DAgger recovers from where BC failed →</text>
    </svg>
  );
}

// ------------------------------------------------------------------- live island
interface Hud {
  startDist: number; // block-to-goal distance (the covariate-shift axis)
  posErr: number;
  far: boolean; // block started/sits past the BC-practice region
  reached: boolean;
  fps: number;
  latMs: number;
  error?: string;
}

type Tok =
  | "--entity-pusher" | "--entity-block" | "--entity-target"
  | "--signal" | "--alert" | "--ink-mute" | "--rule-strong";
const TOK_FALLBACK: Record<Tok, string> = {
  "--entity-pusher": "#b0560f",
  "--entity-block": "#a5257d",
  "--entity-target": "#0c7d5f",
  "--signal": "#1f56de",
  "--alert": "#c0362a",
  "--ink-mute": "#6d6252",
  "--rule-strong": "#c8bc9e",
};

function DaggerToy() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const figureRef = useRef<HTMLElement>(null);
  const apiRef = useRef<{
    reset: () => void;
    sendFar: () => void;
    nudge: (dx: number, dy: number) => void;
  } | null>(null);
  const [booted, setBooted] = useState(false);
  const [hud, setHud] = useState<Hud>({
    startDist: 0, posErr: 0, far: false, reached: false, fps: 0, latMs: 0,
  });

  useEffect(() => {
    let disposed = false;
    let sim: { dispose(): void } | null = null;
    let raf = 0;
    const nextFrame = () => new Promise<number>((r) => (raf = requestAnimationFrame(r)));

    (async () => {
      try {
        // --- lazy, hydration-gated: WASM + ONNX pulled only now (scrolled in) ----
        const [simMod, sceneMod, envMod, obsMod, inferMod, contractsMod, vpMod] =
          await Promise.all([
            import("../../../../playground/src/sim/mujoco_sim"),
            import("../../../../playground/src/sim/scene"),
            import("../../../../playground/src/teleop/pusht_env"),
            import("../../../../playground/src/teleop/pusht_obs"),
            import("../../../../playground/src/policy/infer"),
            import("../../../../playground/src/policy/contracts"),
            import("../../../../playground/src/teleop/viewport"),
          ]);
        const { createSim } = simMod;
        const { PUSHT_XML } = sceneMod;
        const { BrowserPushTEnv } = envMod;
        const { buildObs, CONTROL_DT, POS_TOL } = obsMod;
        const { loadPolicy } = inferMod;
        const { assertDrivesPushT } = contractsMod;
        const { worldToPx, eventToWorld } = vpMod;

        // 1) boot the REAL PushT scene + env
        const realSim: any = await createSim(PUSHT_XML);
        sim = realSim;
        if (disposed) { realSim.dispose(); return; }
        const env: any = new BrowserPushTEnv(realSim);
        let seed = 11;
        env.reset(seed);
        // default-interesting: boot at a FAR start so the first thing seen is the
        // recovery (mirrors the poster). A gentle nudge outward past R_PRACTICE.
        env.perturbBlock(FAR_START * Math.cos(0.7), FAR_START * Math.sin(0.7), realSim.jointQpos("tee_yaw"));

        const canvas = canvasRef.current!;
        canvas.width = CANVAS_PX;
        canvas.height = CANVAS_PX;
        const ctx = canvas.getContext("2d")!;

        const cs = getComputedStyle(canvas);
        const col = (k: Tok) => (cs.getPropertyValue(k).trim() || TOK_FALLBACK[k]);
        const PUSHER = col("--entity-pusher"), BLOCK = col("--entity-block"),
          TARGET = col("--entity-target"), SIGNAL = col("--signal"), INK = col("--ink-mute");

        const teeCorners = (cx: number, cy: number, yaw: number, hx: number, hy: number, ox: number, oy: number) => {
          const c = Math.cos(yaw), s = Math.sin(yaw);
          return [[ox - hx, oy - hy], [ox + hx, oy - hy], [ox + hx, oy + hy], [ox - hx, oy + hy]]
            .map(([lx, ly]) => worldToPx(canvas, cx + lx * c - ly * s, cy + lx * s + ly * c));
        };
        const fillPoly = (pts: number[][], style: string) => {
          ctx.beginPath();
          pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
          ctx.closePath();
          ctx.fillStyle = style;
          ctx.fill();
        };
        const strokePoly = (pts: number[][], style: string) => {
          ctx.beginPath();
          pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
          ctx.closePath();
          ctx.strokeStyle = style;
          ctx.stroke();
        };
        const pxPerM = canvas.width / (2 * WORLD_HALF);
        const render = (dragging: boolean) => {
          const w = canvas.width, h = canvas.height;
          ctx.fillStyle = "#fbf9f3";
          ctx.fillRect(0, 0, w, h);
          ctx.strokeStyle = "rgba(200,188,158,0.5)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          for (let i = 1; i < 10; i++) {
            const p = (i * w) / 10;
            ctx.moveTo(p, 0); ctx.lineTo(p, h);
            ctx.moveTo(0, p); ctx.lineTo(w, p);
          }
          ctx.stroke();

          // the narrow BC-practice region — neutral ink (a map, not an entity)
          const [ox, oy] = worldToPx(canvas, 0, 0);
          ctx.save();
          ctx.globalAlpha = 0.1;
          ctx.fillStyle = INK;
          ctx.beginPath(); ctx.arc(ox, oy, R_PRACTICE * pxPerM, 0, Math.PI * 2); ctx.fill();
          ctx.restore();
          ctx.save();
          ctx.setLineDash([6, 4]);
          ctx.globalAlpha = 0.72;
          ctx.strokeStyle = INK;
          ctx.lineWidth = 1.4;
          ctx.beginPath(); ctx.arc(ox, oy, R_PRACTICE * pxPerM, 0, Math.PI * 2); ctx.stroke();
          ctx.restore();

          // target (fixed at origin) — dashed emerald outline
          ctx.save();
          ctx.setLineDash([5, 4]);
          ctx.lineWidth = 2;
          strokePoly(teeCorners(0, 0, 0, 0.06, 0.015, 0, 0), TARGET);
          strokePoly(teeCorners(0, 0, 0, 0.015, 0.045, 0, -0.06), TARGET);
          ctx.restore();

          // block (tee) — magenta
          const tx = realSim.jointQpos("tee_x"), ty = realSim.jointQpos("tee_y"), tyaw = realSim.jointQpos("tee_yaw");
          fillPoly(teeCorners(tx, ty, tyaw, 0.06, 0.015, 0, 0), BLOCK);
          fillPoly(teeCorners(tx, ty, tyaw, 0.015, 0.045, 0, -0.06), BLOCK);

          // grab halo around the block — the one LIVE handle (signal blue)
          const [bpx, bpy] = worldToPx(canvas, tx, ty);
          if (!dragging) {
            ctx.save();
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = SIGNAL;
            ctx.globalAlpha = 0.8;
            ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.arc(bpx, bpy, GRAB_RADIUS * pxPerM, 0, Math.PI * 2); ctx.stroke();
            ctx.restore();
          }

          // pusher (agent) — amber
          const [px, py] = worldToPx(canvas, realSim.jointQpos("pusher_x"), realSim.jointQpos("pusher_y"));
          const rPx = 0.015 * pxPerM;
          ctx.save();
          ctx.globalAlpha = 0.5; ctx.strokeStyle = PUSHER; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(px, py, rPx * 1.9, 0, Math.PI * 2); ctx.stroke();
          ctx.restore();
          ctx.fillStyle = PUSHER;
          ctx.beginPath(); ctx.arc(px, py, rPx, 0, Math.PI * 2); ctx.fill();
        };

        setBooted(true);
        render(false);

        // 2) load the REAL policy through the fail-closed contract gate
        const policy = await loadPolicy(MODEL_URL);
        assertDrivesPushT(policy.contract);
        if (disposed) return;

        // --- interaction state (refs, not React state — the loop must not re-render)
        let dragging = false;
        const clampB = (v: number) => Math.max(-DRAG_BOUND, Math.min(DRAG_BOUND, v));
        const setBlock = (x: number, y: number) => {
          env.perturbBlock(clampB(x), clampB(y), realSim.jointQpos("tee_yaw"));
        };
        const nearBlock = (wx: number, wy: number) =>
          Math.hypot(wx - realSim.jointQpos("tee_x"), wy - realSim.jointQpos("tee_y")) < GRAB_RADIUS;

        const onDown = (e: PointerEvent) => {
          const [wx, wy] = eventToWorld(canvas, e.clientX, e.clientY);
          if (!nearBlock(wx, wy)) return;
          dragging = true;
          canvas.dataset.dragging = "true";
          canvas.setPointerCapture(e.pointerId);
          setBlock(wx, wy);
          e.preventDefault();
        };
        const onMove = (e: PointerEvent) => {
          if (!dragging) return;
          const [wx, wy] = eventToWorld(canvas, e.clientX, e.clientY);
          setBlock(wx, wy);
        };
        const onUp = (e: PointerEvent) => {
          if (!dragging) return;
          dragging = false;
          canvas.dataset.dragging = "false";
          try { canvas.releasePointerCapture(e.pointerId); } catch { /* already released */ }
        };
        canvas.addEventListener("pointerdown", onDown);
        canvas.addEventListener("pointermove", onMove);
        canvas.addEventListener("pointerup", onUp);
        canvas.addEventListener("pointercancel", onUp);

        apiRef.current = {
          reset: () => env.reset(++seed),
          sendFar: () => setBlock(FAR_START * Math.cos(0.7), FAR_START * Math.sin(0.7)),
          nudge: (dx, dy) => setBlock(realSim.jointQpos("tee_x") + dx, realSim.jointQpos("tee_y") + dy),
        };

        // 3) headless-verification hooks (mirror playground's window.__policy) so a
        //    browser driver can PROVE the policy drives + obs matches training.
        (window as any).__toy = {
          contract: () => ({ ...policy.contract }),
          obsParity: () => {
            const a = env.obs(); const b = buildObs(realSim);
            let m = 0; for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k] - b[k]));
            return { equal: m === 0, maxErr: m };
          },
          blockXY: () => [realSim.jointQpos("tee_x"), realSim.jointQpos("tee_y")],
          startDist: () => distFromGoal(realSim.jointQpos("tee_x"), realSim.jointQpos("tee_y")),
          posErr: () => env.errors().posErr,
          isDragging: () => dragging,
          sendFar: () => apiRef.current?.sendFar(),
          reset: () => apiRef.current?.reset(),
          async drive(n: number) {
            const before = env.errors().posErr;
            for (let i = 0; i < n; i++) { const o = env.obs(); env.step(await policy.act(o)); }
            return {
              steps: n, posErrBefore: before, posErrAfter: env.errors().posErr,
              meanLatencyMs: policy.meanLatencyMs(), calls: policy.calls,
            };
          },
          fps: () => lastFps,
        };

        // 4) DRIVE — real-time paced to CONTROL_HZ (mirrors dagger.py eval:
        //    env.step(policy(obs))). While dragging the policy pauses so the block
        //    tracks the pointer; on release it resumes and drives its recovery.
        let lastFps = 0, frames = 0, fpsMark = performance.now(), last = performance.now(), acc = 0, hudMark = 0;

        while (!disposed) {
          await nextFrame();
          const now = performance.now();
          frames++;
          if (now - fpsMark >= 500) { lastFps = (frames * 1000) / (now - fpsMark); frames = 0; fpsMark = now; }

          if (dragging) {
            acc = 0; last = now;
          } else {
            acc += Math.min(now - last, 100) / 1000;
            last = now;
            let n = 0;
            while (acc >= CONTROL_DT && n < MAX_CONTROL_STEPS_PER_FRAME) {
              const obs = env.obs();                 // buildObs(sim) — training obs, verbatim
              const action = await policy.act(obs);  // raw action[2]; denorm baked in
              const res = env.step(action);          // mirrors dagger.py eval: env.step(policy(obs))
              acc -= CONTROL_DT; n += 1;
              if (res.done) { env.reset(++seed); }    // a fresh full-distribution start
              if (disposed) break;
            }
            if (n === MAX_CONTROL_STEPS_PER_FRAME) acc = 0;
          }

          render(dragging);

          if (now - hudMark >= 100) { // throttle HUD to ~10 Hz
            hudMark = now;
            const tx = realSim.jointQpos("tee_x"), ty = realSim.jointQpos("tee_y");
            const startDist = distFromGoal(tx, ty);
            const { posErr } = env.errors();
            setHud({
              startDist, posErr,
              far: startDist > R_PRACTICE,
              reached: posErr < POS_TOL,
              fps: lastFps, latMs: policy.meanLatencyMs(),
            });
          }
        }

        canvas.removeEventListener("pointerdown", onDown);
        canvas.removeEventListener("pointermove", onMove);
        canvas.removeEventListener("pointerup", onUp);
        canvas.removeEventListener("pointercancel", onUp);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error("[ch4.2 dagger toy] failed", err);
        setHud((h) => ({ ...h, error: msg }));
      }
    })();

    return () => { disposed = true; cancelAnimationFrame(raf); if (sim) sim.dispose(); };
  }, []);

  const failed = !!hud.error;
  const meterPct = Math.min(100, (hud.startDist / METER_FULL) * 100);

  const onKeyDown = (e: KeyboardEvent) => {
    const api = apiRef.current;
    if (!api) return;
    const map: Record<string, [number, number]> = {
      ArrowUp: [0, NUDGE_STEP], ArrowDown: [0, -NUDGE_STEP],
      ArrowLeft: [-NUDGE_STEP, 0], ArrowRight: [NUDGE_STEP, 0],
    };
    if (e.key in map) { e.preventDefault(); api.nudge(...map[e.key]); }
    else if (e.key === "r" || e.key === "R") api.reset();
    else if (e.key === "f" || e.key === "F") api.sendFar();
  };

  return (
    <div class="dg">
      <figure
        ref={figureRef}
        class="dg-figure"
        tabIndex={0}
        role="application"
        aria-label="Interactive DAgger PushT recovery toy. The DAgger policy drives the block toward the target. Grab the block with the pointer, or focus here and use the arrow keys, to move it to a far start outside the region the behavior-cloning demonstrations practiced in, and watch the policy recover — the far starts DAgger collected corrections on. Press R to reset, F to send it to a far start."
        onKeyDown={onKeyDown}
      >
        {/* SSR poster — the JS-off experience and the pre-boot frame */}
        <div class="dg-poster" hidden={booted}><Poster /></div>

        {/* live MuJoCo-WASM canvas — shown once booted */}
        <canvas ref={canvasRef} class="dg-canvas" hidden={!booted} aria-hidden="true" />

        {/* Non-visual path to the same aha: announce only the qualitative
            recovery transition (not the per-frame distance). Honest — never claims
            the episode is solved unless pos_err actually clears the tolerance. */}
        <div class="bk-sr" aria-live="polite">
          {booted && !failed
            ? hud.reached
              ? "The DAgger policy pushed the block onto the target from this start."
              : hud.far
                ? "Far start: the block is past the narrow region the demonstrations practiced in — the covariate-shift territory that broke behavior cloning. The DAgger policy, trained on corrections from exactly these starts, drives back toward the target. It is a modest reactive clone, so it recovers but does not land every episode."
                : "In the practiced region: the DAgger policy pushes the block toward the target."
            : ""}
        </div>

        {/* live HUD */}
        {booted && !failed && (
          <div class="dg-hud" aria-hidden="true">
            <div class="dg-hud-row">
              <span class="dg-k">start distance</span>
              <span class={`dg-v ${hud.far ? "dg-bad" : "dg-ok"}`}>
                {hud.startDist.toFixed(3)} m {hud.far ? "▲ far" : "✓"}
              </span>
            </div>
            <div class="dg-meter">
              <div class="dg-meter-fill" data-far={hud.far} style={`width:${meterPct}%`} />
            </div>
            <div class="dg-hud-row">
              <span class="dg-k">region</span>
              <span class={`dg-v ${hud.far ? "dg-bad" : "dg-ok"}`}>
                {hud.far ? "covariate shift (BC fails)" : "BC practiced here"}
              </span>
            </div>
            <div class="dg-hud-row">
              <span class="dg-k">pos_err → target</span>
              <span class={`dg-v ${hud.reached ? "dg-ok" : ""}`}>
                {hud.posErr.toFixed(3)} m {hud.reached ? "✓" : ""}
              </span>
            </div>
          </div>
        )}

        {/* boot / instrument status line */}
        <div class="dg-status" data-failed={failed} aria-hidden="true">
          {failed ? (
            <span>sim failed — the Colab path covers this without WASM</span>
          ) : booted ? (
            <>
              <span>real dagger.onnx · best round (DAgger 3)</span>
              <span>{hud.fps.toFixed(0)} fps · {hud.latMs.toFixed(2)} ms/call</span>
            </>
          ) : (
            <span>booting MuJoCo-WASM + policy…</span>
          )}
        </div>
      </figure>

      <div class="dg-controls">
        <button type="button" class="dg-btn dg-btn--primary" onClick={() => apiRef.current?.sendFar()} disabled={!booted || failed}>
          send it to a far start →
        </button>
        <button type="button" class="dg-btn" onClick={() => apiRef.current?.reset()} disabled={!booted || failed}>
          reset · new start
        </button>
        <span class="dg-control-note">
          DAgger recovers the covariate-shift loss (~0.2 success, not a great policy) · best round, not the last · poster reads with JS off
        </span>
      </div>
    </div>
  );
}

export default function DaggerPushtLive() {
  return <DaggerToy />;
}
