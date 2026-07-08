/**
 * ch0.4 "pusht-teleop" concept-toy — "an episode is (obs, action) over time; you
 * are the expert." The learner DRAGS the pusher to walk the T toward the target
 * while every 10 Hz control step's (observation.state[10], action[2]) is captured
 * into a frame buffer. Then a timeline SCRUBBER replays that episode: as you drag
 * the scrubber the obs[10]/action[2] arrays fill in frame-by-frame — the abstract
 * "dataset" made concrete as the exact numbers you generated with your hand. The
 * action row is highlighted because THAT is the label: your motion is what ch1.1's
 * behavior cloning learns to imitate.
 *
 * Built to the FROZEN CONCEPT-TOY CONTRACT (see PlateIsland.tsx):
 *   1. SSR poster == JS-off fallback. `<Poster/>` + a static obs/action schema
 *      render server-side (no window at module scope, no WASM). With JS disabled
 *      that captioned arena + the labelled 10-slot layout ARE the lesson.
 *   2. Lazy, hydration-gated sim. Every heavy module (createSim, the env, the
 *      DragController, the viewport transform) is pulled by dynamic import()
 *      inside the post-hydration effect — WASM is fetched only once mounted.
 *   3. Reuse the playground primitives verbatim: createSim + PUSHT_XML,
 *      BrowserPushTEnv (obs()/reset()/step()), DragController (pointer -> target
 *      velocity action[2]), worldToPx/eventToWorld. Only the render is bespoke so
 *      entities wear the page's --entity-* hues.
 *   4. Make the invisible visible: the recorded (obs, action) arrays tied to your
 *      motion; scrubbing replays your own episode; a live "N frames recorded".
 *   5. One control, immediate feedback, default-interesting: the toy boots with a
 *      short demo episode already scrubbable — honestly generated on boot by
 *      running the SAME scripted stand-in controller as record.py's local teleop
 *      (scriptedDrive) through the REAL MuJoCo env. Never faked arrays.
 *   6. Colour discipline: --entity-* for entities + obs slot labels, neutral ink
 *      for the array values, ONE --signal blue for the record/scrub control and
 *      the action you generate (the label). --alert red only for the REC toggle.
 *
 * Orchestrator wiring: mounted for demo id `pusht-teleop` (meta.yaml `demo:`).
 */
import { useEffect, useRef, useState } from "preact/hooks";
import "./pusht-teleop.css";
// Light, WASM-free constants (pure module — safe to import at the top level and
// on the server): the obs/action feature names + control cadence, the single
// source of truth shared with pusht_env.py / gen_demos.py.
import {
  ACTION_NAMES,
  CONTROL_DT,
  OBS_DIM,
  STATE_NAMES,
} from "../../../../playground/src/teleop/pusht_obs";

// ------------------------------------------------------------------- constants
const CANVAS_PX = 512;
const DEMO_SEED = 3; // fixed so the boot-time demo episode is reproducible
const DEMO_LEN = 60; // control steps to script for the demo (~6 s at 10 Hz)
const MAX_EP_FRAMES = 220; // cap a learner recording (~22 s) so buffers stay bounded
const MAX_CONTROL_STEPS_PER_FRAME = 4; // mirror main.ts: cap control-step catch-up

/** A recorded control step: the (observation-before, action-applied) pair —
 *  the same convention as record.py / gen_demos.add_frame. */
interface Frame {
  obs: number[]; // length OBS_DIM (10)
  action: number[]; // length 2
}

// Which entity each obs slot describes, for the colour-coded array readout.
type Entity = "pusher" | "block" | "target";
const OBS_SLOTS: Array<{ label: string; entity: Entity; idx: number[] }> = [
  { label: "pusher xy", entity: "pusher", idx: [0, 1] },
  { label: "T-block xy", entity: "block", idx: [2, 3] },
  { label: "T yaw (sin, cos)", entity: "block", idx: [4, 5] },
  { label: "target pose", entity: "target", idx: [6, 7, 8, 9] },
];

type PaletteKey =
  | "--entity-pusher" | "--entity-block" | "--entity-target"
  | "--signal" | "--ink-mute" | "--rule-strong";
const PALETTE_FALLBACK: Record<PaletteKey, string> = {
  "--entity-pusher": "#b0560f",
  "--entity-block": "#a5257d",
  "--entity-target": "#0c7d5f",
  "--signal": "#1f56de",
  "--ink-mute": "#6d6252",
  "--rule-strong": "#c8bc9e",
};

/** The scripted stand-in for your hand — ported verbatim from record.py's
 *  `scripted_drive`: get behind the block, then shove it toward the target. Used
 *  ONLY to honestly generate the boot-time demo episode (drives the REAL env, so
 *  the arrays are real). The learner's own recording uses the DragController. */
function scriptedDrive(obs: ArrayLike<number>): [number, number] {
  const px = obs[0], py = obs[1], tx = obs[2], ty = obs[3];
  const gx = obs[6], gy = obs[7];
  const dx = gx - tx, dy = gy - ty;
  const reach = Math.hypot(dx, dy);
  const gux = reach > 1e-3 ? dx / (reach + 1e-9) : 0;
  const guy = reach > 1e-3 ? dy / (reach + 1e-9) : 0;
  const cx = tx - gux * 0.05, cy = ty - guy * 0.05; // the spot just behind the block
  const tox = cx - px, toy = cy - py;
  const gap = Math.hypot(tox, toy);
  if (gap > 0.02) return [(tox / (gap + 1e-9)) * 0.6, (toy / (gap + 1e-9)) * 0.6];
  return [gux * 0.5, guy * 0.5];
}

// ------------------------------------------------------------- shared SVG poster
// Square viewBox; world (x,y) -> svg px, world +y up. Same ±0.45 m frame as the
// live canvas so booting causes no reflow.
const POSTER_V = 500;
const WORLD_HALF = 0.45;
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

/** The static, captioned poster — SSR output + the JS-off arena. */
function Poster() {
  const blockX = 0.14, blockY = 0.1, blockYaw = -20;
  const pusherX = 0.22, pusherY = 0.16; // just behind the block, mid-push
  const [pcx, pcy] = w2s(pusherX, pusherY);
  // a dashed "you drag the pusher" cue pointing from the pusher toward the block
  const [bx, by] = w2s(blockX + 0.03, blockY + 0.02);
  return (
    <svg
      class="pt-poster-svg"
      viewBox={`0 0 ${POSTER_V} ${POSTER_V}`}
      role="img"
      aria-label="Top-down PushT arena. The amber pusher sits behind the magenta T-block, with the green dashed target pose at the center. A blue dashed arrow marks the pusher as draggable: drag it to walk the T onto the target, and each control step's observation and action are recorded as one frame of your episode. When the sim loads you can record your own episode and scrub through the (observation, action) arrays it produced."
    >
      <title>PushT teleoperation — record an episode of (observation, action) pairs</title>
      <desc>
        An episode is a sequence of (observation.state[10], action[2]) recorded while
        you drive the robot. Drag the pusher to push the T toward the target; every
        10 Hz control step captures one frame. Then scrub the timeline to watch the
        arrays you generated fill in, frame by frame — your motion is the label.
      </desc>

      <rect class="pt-arena" x={2} y={2} width={POSTER_V - 4} height={POSTER_V - 4} rx={6} />
      <g class="pt-grid">
        {Array.from({ length: 9 }, (_, i) => ((i + 1) * POSTER_V) / 10).map((v) => (
          <>
            <line x1={v} y1={2} x2={v} y2={POSTER_V - 2} />
            <line x1={2} y1={v} x2={POSTER_V - 2} y2={v} />
          </>
        ))}
      </g>

      {/* target pose (fixed at origin) */}
      <PosterTee x={0} y={0} yawDeg={0} className="pt-target" />
      <text class="pt-target-label" x={w2s(0.02, 0.02)[0]} y={w2s(0.02, 0.02)[1]}>target</text>

      {/* the block, mid-push */}
      <PosterTee x={blockX} y={blockY} yawDeg={blockYaw} className="pt-tee" />

      {/* the pusher — the agent you steer */}
      <g transform={`translate(${pcx.toFixed(1)} ${pcy.toFixed(1)})`}>
        <circle class="pt-pusher-ring" r={0.03 * POSTER_S} />
        <circle class="pt-pusher-core" r={0.015 * POSTER_S} />
      </g>

      {/* the drag cue — the one LIVE handle (signal blue) */}
      <line class="pt-drag" x1={pcx} y1={pcy} x2={bx} y2={by} />
      <circle class="pt-drag-head" cx={bx} cy={by} r={4} />
      <text class="pt-drag-label" x={pcx + 8} y={pcy - 10}>drag to push &amp; record →</text>
    </svg>
  );
}

// --------------------------------------------------------- the array readout row
/** Format a single obs/action scalar for the readout. */
function fmt(v: number | undefined): string {
  if (v === undefined || Number.isNaN(v)) return "—";
  return (v >= 0 ? " " : "") + v.toFixed(3);
}

// --------------------------------------------------------------- the live island
type Mode = "review" | "recording";

/** ch0.4 — the teleop episode-recording concept-toy (SSR poster + live island). */
export default function PushTTeleopToy() {
  const figureRef = useRef<HTMLElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // sim/env/render handles, set once booted (kept in refs so the record loop and
  // scrub effect never trigger React re-renders).
  const simRef = useRef<{ jointQpos(n: string): number; dispose(): void } | null>(null);
  const envRef = useRef<{
    reset(seed: number): Float32Array;
    obs(): Float32Array;
    step(a: ArrayLike<number>): { done: boolean };
  } | null>(null);
  const controllerRef = useRef<{ isDragging: boolean; action(px: number, py: number): [number, number]; dispose(): void } | null>(null);
  const paintFrameRef = useRef<((f: Frame | null) => void) | null>(null);
  const paintLiveRef = useRef<(() => void) | null>(null);

  const demoRef = useRef<Frame[]>([]); // the boot-time generated episode (restorable)
  const bufRef = useRef<Frame[]>([]); // the in-progress recording
  const recordingRef = useRef(false);
  const disposedRef = useRef(false);
  const rafRef = useRef(0);
  const seedRef = useRef(DEMO_SEED);
  const keysRef = useRef<Set<string>>(new Set());
  const lastActionRef = useRef<[number, number]>([0, 0]);

  const [booted, setBooted] = useState(false);
  const [mode, setMode] = useState<Mode>("review");
  const [episode, setEpisode] = useState<Frame[]>([]);
  const [scrubIndex, setScrubIndex] = useState(0);
  const [liveCount, setLiveCount] = useState(0);
  const [liveFrame, setLiveFrame] = useState<Frame | null>(null); // newest (obs,action) captured while recording
  const [isDemo, setIsDemo] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // -------------------- boot: lazy WASM + honest demo episode -------------------
  useEffect(() => {
    disposedRef.current = false;

    (async () => {
      try {
        const [simMod, sceneMod, envMod, ctrlMod, vpMod] = await Promise.all([
          import("../../../../playground/src/sim/mujoco_sim"),
          import("../../../../playground/src/sim/scene"),
          import("../../../../playground/src/teleop/pusht_env"),
          import("../../../../playground/src/teleop/controller"),
          import("../../../../playground/src/teleop/viewport"),
        ]);
        const { createSim } = simMod;
        const { PUSHT_XML } = sceneMod;
        const { BrowserPushTEnv } = envMod;
        const { DragController } = ctrlMod;
        const { worldToPx } = vpMod;

        const realSim: any = await createSim(PUSHT_XML);
        if (disposedRef.current) { realSim.dispose(); return; }
        simRef.current = realSim;
        const env: any = new BrowserPushTEnv(realSim);
        envRef.current = env;

        const canvas = canvasRef.current!;
        canvas.width = CANVAS_PX;
        canvas.height = CANVAS_PX;
        const ctx = canvas.getContext("2d")!;
        controllerRef.current = new DragController(canvas);

        const pxPerM = worldToPx(canvas, 1, 0)[0] - worldToPx(canvas, 0, 0)[0];

        // --- bespoke top-down renderer (entities in the page's --entity-* hues) --
        const teeCorners = (cx: number, cy: number, yaw: number, hx: number, hy: number, ox: number, oy: number) => {
          const c = Math.cos(yaw), s = Math.sin(yaw);
          return [[ox - hx, oy - hy], [ox + hx, oy - hy], [ox + hx, oy + hy], [ox - hx, oy + hy]]
            .map(([lx, ly]) => worldToPx(canvas, cx + lx * c - ly * s, cy + lx * s + ly * c));
        };
        const strokeOrFillTee = (cx: number, cy: number, yaw: number, style: string, dashed: boolean) => {
          const parts = [teeCorners(cx, cy, yaw, 0.06, 0.015, 0, 0), teeCorners(cx, cy, yaw, 0.015, 0.045, 0, -0.06)];
          ctx.save();
          if (dashed) { ctx.setLineDash([5, 4]); ctx.strokeStyle = style; ctx.lineWidth = 2; }
          else ctx.fillStyle = style;
          parts.forEach((pts) => {
            ctx.beginPath();
            pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
            ctx.closePath();
            dashed ? ctx.stroke() : ctx.fill();
          });
          ctx.restore();
        };
        // draw the whole scene from explicit poses + the action you applied.
        const drawScene = (
          px: number, py: number, tx: number, ty: number, tyaw: number,
          action: [number, number] | null, showGrab: boolean,
        ) => {
          const w = canvas.width, h = canvas.height;
          // Resolve the page's design tokens LIVE each paint (a canvas can't read CSS
          // vars natively) so a post-boot theme toggle recolours the entities instead
          // of freezing the boot-time snapshot. The arena stays a warm-light lab
          // surface in both themes.
          const cs = getComputedStyle(canvas);
          const col = (k: PaletteKey) => cs.getPropertyValue(k).trim() || PALETTE_FALLBACK[k];
          const PUSHER = col("--entity-pusher"), BLOCK = col("--entity-block"),
            TARGET = col("--entity-target"), SIGNAL = col("--signal");
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

          strokeOrFillTee(0, 0, 0, TARGET, true); // target at origin
          strokeOrFillTee(tx, ty, tyaw, BLOCK, false); // the block

          const [ppx, ppy] = worldToPx(canvas, px, py);
          const rPx = 0.015 * pxPerM;
          if (showGrab) {
            ctx.save();
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = SIGNAL; ctx.globalAlpha = 0.8; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.arc(ppx, ppy, rPx * 2.6, 0, Math.PI * 2); ctx.stroke();
            ctx.restore();
          }
          ctx.save();
          ctx.globalAlpha = 0.5; ctx.strokeStyle = PUSHER; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(ppx, ppy, rPx * 1.9, 0, Math.PI * 2); ctx.stroke();
          ctx.restore();
          ctx.fillStyle = PUSHER;
          ctx.beginPath(); ctx.arc(ppx, ppy, rPx, 0, Math.PI * 2); ctx.fill();

          // the action arrow — the label you generated (signal blue). action is a
          // target velocity in [-1,1]; ~0.12 m of lead saturates it, so scale by 0.12.
          if (action && (Math.abs(action[0]) > 1e-3 || Math.abs(action[1]) > 1e-3)) {
            const [ax, ay] = worldToPx(canvas, px + action[0] * 0.12, py + action[1] * 0.12);
            ctx.strokeStyle = SIGNAL; ctx.fillStyle = SIGNAL; ctx.lineWidth = 2.5;
            ctx.beginPath(); ctx.moveTo(ppx, ppy); ctx.lineTo(ax, ay); ctx.stroke();
            const ang = Math.atan2(ay - ppy, ax - ppx), ah = 8;
            ctx.beginPath();
            ctx.moveTo(ax, ay);
            ctx.lineTo(ax - ah * Math.cos(ang - 0.4), ay - ah * Math.sin(ang - 0.4));
            ctx.lineTo(ax - ah * Math.cos(ang + 0.4), ay - ah * Math.sin(ang + 0.4));
            ctx.closePath(); ctx.fill();
          }
        };

        paintFrameRef.current = (f: Frame | null) => {
          if (!f) { drawScene(0, 0, 0.14, 0.1, -0.35, null, false); return; }
          const o = f.obs;
          drawScene(o[0], o[1], o[2], o[3], Math.atan2(o[4], o[5]), [f.action[0], f.action[1]], false);
        };
        paintLiveRef.current = () => {
          const s = realSim, c = controllerRef.current!;
          drawScene(
            s.jointQpos("pusher_x"), s.jointQpos("pusher_y"),
            s.jointQpos("tee_x"), s.jointQpos("tee_y"), s.jointQpos("tee_yaw"),
            lastActionRef.current, !c.isDragging,
          );
        };

        // --- honestly generate the boot-time demo: the SAME scripted stand-in as
        //     record.py's local teleop, driving the REAL MuJoCo env. Every array
        //     below is what the env actually produced — nothing is invented. ------
        env.reset(DEMO_SEED);
        const demo: Frame[] = [];
        for (let i = 0; i < DEMO_LEN; i++) {
          const obs = env.obs();
          const a = scriptedDrive(obs);
          demo.push({ obs: Array.from(obs) as number[], action: [a[0], a[1]] });
          if (env.step(a).done) break;
        }
        demoRef.current = demo;

        setEpisode(demo);
        setScrubIndex(0);
        setIsDemo(true);
        setMode("review");
        setBooted(true);
        // Reduced motion: the toy boots to a single still demo frame and never starts
        // an autonomous rAF loop — the only loop (startRecording) is gated behind the
        // explicit Record control, so there is no auto-motion to suppress here.
        paintFrameRef.current(demo[0] ?? null);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error("[ch0.4 toy] failed", err);
        setError(msg);
      }
    })();

    return () => {
      disposedRef.current = true;
      recordingRef.current = false;
      cancelAnimationFrame(rafRef.current);
      controllerRef.current?.dispose();
      simRef.current?.dispose();
    };
  }, []);

  // ------------- redraw the scrubbed frame whenever review state changes --------
  useEffect(() => {
    if (!booted || mode !== "review") return;
    paintFrameRef.current?.(episode[scrubIndex] ?? null);
  }, [booted, mode, episode, scrubIndex]);

  // -------------------------- record / scrub imperatives ------------------------
  const keyboardAction = (): [number, number] | null => {
    const k = keysRef.current;
    const vx = (k.has("ArrowRight") ? 1 : 0) - (k.has("ArrowLeft") ? 1 : 0);
    const vy = (k.has("ArrowUp") ? 1 : 0) - (k.has("ArrowDown") ? 1 : 0);
    if (vx === 0 && vy === 0) return null;
    return [vx, vy];
  };

  const stopRecording = () => {
    if (!recordingRef.current) return;
    recordingRef.current = false;
    cancelAnimationFrame(rafRef.current);
    const rec = bufRef.current.slice();
    // keep at least one frame so the scrubber/readout always has data
    const finalEp = rec.length > 0 ? rec : demoRef.current;
    setEpisode(finalEp);
    setScrubIndex(Math.max(0, finalEp.length - 1));
    setIsDemo(rec.length === 0);
    setMode("review");
  };

  const startRecording = () => {
    const env = envRef.current;
    if (!env || recordingRef.current) return;
    seedRef.current += 1;
    env.reset(seedRef.current);
    bufRef.current = [];
    lastActionRef.current = [0, 0];
    keysRef.current.clear();
    recordingRef.current = true;
    setLiveCount(0);
    setLiveFrame(null);
    setMode("recording");
    figureRef.current?.focus();

    let last = performance.now(), acc = 0, hudMark = 0;
    const loop = () => {
      if (!recordingRef.current || disposedRef.current) return;
      rafRef.current = requestAnimationFrame(loop);
      const now = performance.now();
      acc += Math.min(now - last, 100) / 1000;
      last = now;
      let n = 0, ended = false;
      while (acc >= CONTROL_DT && n < MAX_CONTROL_STEPS_PER_FRAME) {
        const obs = env.obs();
        // keyboard (arrow keys) is the accessible fallback to the pointer drag;
        // otherwise the DragController maps pointer lead -> target velocity.
        const a = keyboardAction() ?? controllerRef.current!.action(obs[0], obs[1]);
        lastActionRef.current = a;
        // record the (obs-before, action) pair — exactly record.py's convention
        bufRef.current.push({ obs: Array.from(obs) as number[], action: [a[0], a[1]] });
        const done = env.step(a).done;
        acc -= CONTROL_DT; n += 1;
        if (done || bufRef.current.length >= MAX_EP_FRAMES) { ended = true; break; }
      }
      if (n === MAX_CONTROL_STEPS_PER_FRAME) acc = 0;
      paintLiveRef.current?.();
      if (now - hudMark >= 100) {
        hudMark = now;
        setLiveCount(bufRef.current.length);
        // surface the newest captured (obs, action) so the arrays panel updates live
        // while recording — bufRef alone is a ref, so the readout would look frozen.
        setLiveFrame(bufRef.current[bufRef.current.length - 1] ?? null);
      }
      if (ended) stopRecording();
    };
    rafRef.current = requestAnimationFrame(loop);
  };

  const showDemo = () => {
    if (recordingRef.current) stopRecording();
    setEpisode(demoRef.current);
    setScrubIndex(0);
    setIsDemo(true);
    setMode("review");
  };

  // figure keyboard: arrows drive the pusher while recording, else scrub frames.
  const onFigureKeyDown = (e: KeyboardEvent) => {
    if (mode === "recording") {
      if (e.key.startsWith("Arrow")) { keysRef.current.add(e.key); e.preventDefault(); }
      else if (e.key === "r" || e.key === "R" || e.key === " ") { e.preventDefault(); stopRecording(); }
      return;
    }
    if (e.key === "ArrowLeft") { e.preventDefault(); setScrubIndex((i) => Math.max(0, i - 1)); }
    else if (e.key === "ArrowRight") { e.preventDefault(); setScrubIndex((i) => Math.min(episode.length - 1, i + 1)); }
    else if (e.key === "r" || e.key === "R") { e.preventDefault(); startRecording(); }
  };
  const onFigureKeyUp = (e: KeyboardEvent) => {
    if (e.key.startsWith("Arrow")) keysRef.current.delete(e.key);
  };

  const recording = mode === "recording";
  // While recording, show the live-captured frame (updated ~10 Hz from the record
  // loop) so the (obs, action) arrays fill in as you drive — not a stale review frame.
  const frame = booted ? (recording ? liveFrame : episode[scrubIndex] ?? null) : null;
  const nFrames = episode.length;

  return (
    <div class="pt">
      <figure
        ref={figureRef}
        class="pt-figure"
        data-recording={recording}
        tabIndex={0}
        role="application"
        aria-label="PushT teleoperation recorder. Press R to start recording, then drag the pusher (or use the arrow keys) to push the T-block toward the target; each control step is captured as one (observation, action) frame. Press R or space to stop. When reviewing, the left and right arrow keys scrub through the frames you recorded."
        onKeyDown={onFigureKeyDown}
        onKeyUp={onFigureKeyUp}
      >
        {/* SSR poster — the JS-off experience and the pre-boot frame */}
        <div class="pt-poster" hidden={booted}><Poster /></div>

        {/* live MuJoCo-WASM canvas — shown once booted */}
        <canvas ref={canvasRef} class="pt-canvas" hidden={!booted} aria-hidden="true" />

        {booted && !error && (
          <div class="pt-badge" aria-hidden="true">
            <span>
              {recording
                ? <span class="pt-rec">REC</span>
                : isDemo ? "demo episode" : "your episode"}
            </span>
            <span>
              {recording
                ? <>recording · <b>{liveCount}</b> frames</>
                : <>frame <b>{nFrames ? scrubIndex + 1 : 0}</b> / {nFrames}</>}
            </span>
          </div>
        )}

        <div class="pt-status" data-failed={!!error} aria-hidden="true">
          {error
            ? "sim failed to load — the Colab path covers this without WASM"
            : booted
              ? recording ? "drag the pusher to push the T · press R or space to stop"
                          : "scrub the timeline below to replay your episode"
              : "booting MuJoCo-WASM…"}
        </div>
      </figure>

      {/* Non-visual path to the same aha (this toy previously had no aria-live region):
          announce mode transitions + the episode under review. Per-frame numbers stay
          in the visual arrays panel + the scrubber's native value announcement, so this
          stays qualitative and does not spam the screen reader as frames stream in. */}
      <div class="bk-sr" aria-live="polite">
        {!booted || error
          ? ""
          : recording
            ? "Recording your episode — drive the pusher with a drag or the arrow keys; each control step is captured as one observation-and-action frame. Press R or space to stop."
            : nFrames
              ? `Reviewing the ${isDemo ? "demo" : "recorded"} episode — ${nFrames} frames of observation.state[10] and action[2]. Scrub to replay; the action row is your motion, the label behaviour cloning imitates.`
              : ""}
      </div>

      {/* the timeline scrubber — the one live control (signal blue). Native range
          input = keyboard-accessible drag + arrow-step for free. */}
      <div class="pt-scrub" hidden={!booted || recording || nFrames === 0}>
        <span class="pt-scrub-count">frame {nFrames ? scrubIndex + 1 : 0}/{nFrames}</span>
        <input
          class="pt-range"
          type="range"
          min={0}
          max={Math.max(0, nFrames - 1)}
          value={scrubIndex}
          onInput={(e) => setScrubIndex(Number((e.currentTarget as HTMLInputElement).value))}
          aria-label={`Scrub the recorded episode. Frame ${scrubIndex + 1} of ${nFrames}.`}
        />
        <span class="pt-step">
          <button type="button" class="pt-btn" onClick={() => setScrubIndex((i) => Math.max(0, i - 1))} aria-label="previous frame">◄</button>
          <button type="button" class="pt-btn" onClick={() => setScrubIndex((i) => Math.min(nFrames - 1, i + 1))} aria-label="next frame">►</button>
        </span>
      </div>

      {/* the arrays — obs[10] + action[2] for the scrubbed frame. This is the
          dataset made concrete: the numbers you generated with your hand. Renders
          server-side too (with em-dash placeholders) so the 10-slot layout reads
          with JS off. */}
      <div class="pt-arrays">
        <div class="pt-arrays-head">
          <span>this frame is one row of your dataset: <b>observation.state[{OBS_DIM}]</b> + <b>action[2]</b></span>
        </div>

        <div class="pt-obs">
          {OBS_SLOTS.map((slot) => (
            <div class="pt-slot" data-entity={slot.entity}>
              <span class="pt-slot-label">{slot.label}</span>
              <span class="pt-cells">
                {slot.idx.map((i) => (
                  <span class="pt-cell">
                    <span class="pt-name"><span class="pt-idx">[{i}]</span> {STATE_NAMES[i]}</span>
                    <span class="pt-val">{fmt(frame?.obs[i])}</span>
                  </span>
                ))}
              </span>
            </div>
          ))}
        </div>

        <div class="pt-action">
          <span class="pt-slot-label">action[2] — your motion, the label</span>
          <span class="pt-cells">
            {ACTION_NAMES.map((name, i) => (
              <span class="pt-cell">
                <span class="pt-name"><span class="pt-idx">[{i}]</span> {name}</span>
                <span class="pt-val">{fmt(frame?.action[i])}</span>
              </span>
            ))}
          </span>
          <span class="pt-action-note">
            ch1.1's behavior cloning learns to predict this action from the observation above.
          </span>
        </div>
      </div>

      <div class="pt-controls">
        {recording ? (
          <button type="button" class="pt-btn pt-btn--rec" onClick={stopRecording} disabled={!booted || !!error}>
            ■ stop recording
          </button>
        ) : (
          <button type="button" class="pt-btn pt-btn--primary" onClick={startRecording} disabled={!booted || !!error}>
            ● record your own episode
          </button>
        )}
        <button type="button" class="pt-btn" onClick={showDemo} disabled={!booted || !!error || recording}>
          watch the demo
        </button>
        <span class="pt-note">
          drag the pusher (or arrow-keys) to record · scrub to replay · poster reads with JS off
        </span>
      </div>
    </div>
  );
}
