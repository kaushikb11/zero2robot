/**
 * ch0.1 "The Simulation Loop" — the TIMESTEP concept-toy (`demo: sim-loop-perturb`).
 *
 * The chapter's Break-It, felt: expose exactly ONE control — the simulation
 * timestep `dt` (mjModel.opt.timestep) — and let the learner drag it up until the
 * integrator can no longer keep up and the sim explodes. A free box orbits a
 * spring anchor; at a small dt the orbit is smooth and its energy holds steady,
 * and as dt climbs past the integrator's stability limit the same loop injects
 * energy every step, the orbit spirals out, and the box flies off the table.
 *
 * This mirrors sim_loop.py's scene spirit (a floor plane + a free box you
 * perturb) and its Break-It (`--timestep too large -> energy explosion`), reduced
 * to the one variable that is the whole lesson here.
 *
 * Built to the FROZEN CONCEPT-TOY CONTRACT documented at the top of
 * ../PlateIsland.tsx (read it before touching this):
 *   1. SSR poster == the JS-off fallback. <Poster/> is pure static JSX (no
 *      window/document at module scope or first render); with JS off it is the
 *      whole experience and no WASM is ever fetched.
 *   2. Lazy, hydration-gated sim. MuJoCo-WASM is pulled by a *dynamic* import()
 *      INSIDE the post-hydration effect (mounted client:visible, so it loads only
 *      when scrolled into view). The canvas is hidden until booted; the poster is
 *      hidden after.
 *   3. Reuse the sim primitive verbatim (createSim); only the render is bespoke,
 *      so the box/table wear the page's ink and the ONE live control is signal blue.
 *   4. Make the invisible visible: a live energy meter (kinetic + spring PE,
 *      relative to the reset baseline) that stays flat when stable and climbs when
 *      the timestep goes unstable.
 *   5. ONE control (the dt slider), immediate feedback, default-interesting (boots
 *      at a stable dt with a gentle orbit), plus a reset.
 *   6. Colour discipline: neutral ink for the box + table, ONE --signal blue for
 *      the dt slider + its readout, --alert red only for the unstable/exploding state.
 */
import "./sim-loop-perturb.css";
import { useEffect, useRef, useState } from "preact/hooks";

// ------------------------------------------------------------------- constants
// Physics is a 2-D spring-mass integrated by MuJoCo's Euler integrator. The
// stability limit of that integrator for this system is dt_crit ~= 2/omega with
// omega = sqrt(K/m); the numbers below put dt_crit ~= 0.028 s, squarely inside
// the slider's 0.001..0.05 range so the learner can drive the sim across it.
const WORLD_HALF = 0.45; // metres; the ±extent the top-down view maps to the canvas
const BOX_MASS = 0.2; // kg — matches the MJCF geom mass
const K_SPRING = 1000; // N/m — horizontal restoring force toward the anchor at origin
const C_DAMP = 0.3; // N·s/m — light drag; the stable orbit only decays slowly
const OMEGA = Math.sqrt(K_SPRING / BOX_MASS); // ~70.7 rad/s
const INIT_R = 0.11; // m — orbit radius (visible, gentle)
const INIT_VY = OMEGA * INIT_R; // gives a near-circular orbit at radius INIT_R
const INIT_Z = 0.045; // m — resting height (gravity is off; z never changes)

const DT_MIN = 0.001;
const DT_MAX = 0.05;
const DT_STEP = 0.001;
const DT_DEFAULT = 0.002; // stable + smooth: the default-interesting start

const SIM_PER_FRAME = 0.003; // target sim-seconds advanced per animation frame, so
                             // apparent orbit speed stays ~constant for small dt and
                             // the ONLY thing the slider changes is stability
const MAX_SUBSTEPS = 12; // cap mj_step calls per frame (guards tiny dt)

const UNSTABLE_RATIO = 1.4; // energy/baseline above which we call it unstable
const METER_MAX_RATIO = 6; // energy/baseline that fills the meter
const EXPLODE_R = 1.4; // m from origin past which we declare the box exploded

const CANVAS_PX = 512;
const TRAIL_MAX = 90; // orbit trail ring-buffer length

// Minimal self-contained MJCF: a (visual-only) table plane + one free box.
// Gravity is off and both geoms are non-colliding (contype/conaffinity 0), so the
// dynamics are a clean, analysable 2-D spring-mass — the box's whole motion comes
// from the restoring force this component writes into mjData.xfrc_applied each
// step. That keeps the instability purely about the integrator + timestep, which
// is the lesson (no contact-solver variability muddying it).
const SCENE_XML = `
<mujoco model="sim_loop_timestep">
  <option timestep="${DT_DEFAULT}" integrator="Euler" gravity="0 0 0"/>
  <worldbody>
    <light pos="0 0 1" dir="0 0 -1"/>
    <geom name="table" type="plane" size="1 1 0.1" contype="0" conaffinity="0"
          rgba="0.85 0.85 0.85 1"/>
    <body name="box" pos="${INIT_R} 0 ${INIT_Z}">
      <freejoint name="box_free"/>
      <geom name="box_geom" type="box" size="0.045 0.045 0.045" mass="${BOX_MASS}"
            contype="0" conaffinity="0" rgba="0.3 0.28 0.24 1"/>
    </body>
    <!-- Parked pusher: the chapter's scene is "empty-scene-pusher", and the shared
         createSim resolves pusher_x/pusher_y at boot. Undriven + non-colliding, so
         it never touches the box/spring dynamics; the box stays body index 1. -->
    <body name="pusher" pos="0 -0.34 0.02">
      <joint name="pusher_x" type="slide" axis="1 0 0"/>
      <joint name="pusher_y" type="slide" axis="0 1 0"/>
      <geom name="pusher_geom" type="cylinder" size="0.02 0.02"
            contype="0" conaffinity="0" rgba="0.55 0.55 0.58 1"/>
    </body>
  </worldbody>
</mujoco>
`;

// ------------------------------------------------------------- SSR poster (JS-off)
// Pure static JSX — the complete experience with JavaScript disabled and the
// pre-boot frame. NOTHING here reads window/document.
const POSTER_V = 500;
const POSTER_S = POSTER_V / (2 * WORLD_HALF);
const w2s = (x: number, y: number): [number, number] => [
  POSTER_V / 2 + x * POSTER_S,
  POSTER_V / 2 - y * POSTER_S, // world +y is up
];

function Poster() {
  const [bx, by] = w2s(INIT_R, 0);
  const [ox, oy] = w2s(0, 0);
  const boxSide = 0.09 * POSTER_S;
  const orbitR = INIT_R * POSTER_S;
  return (
    <svg
      class="sl-poster-svg"
      viewBox={`0 0 ${POSTER_V} ${POSTER_V}`}
      role="img"
      aria-label="Top-down simulation arena. A dark box sits on a light table, orbiting a small blue spring anchor at the centre along a dashed circular path. With JavaScript on, one control — the simulation timestep — is exposed: at a small timestep the orbit stays smooth and its energy holds steady; drag the timestep higher and the integrator goes unstable, the orbit spirals outward, and the box explodes off the table."
    >
      <title>The simulation loop — timestep-instability toy</title>
      <desc>
        A free box is held by a spring toward the centre. The simulation advances in
        fixed timesteps; when the timestep is small the motion is faithful, and when
        it is too large for the integrator the energy grows without bound and the box
        flies away. Increase the timestep with the slider to feel it.
      </desc>

      {/* table + graph paper */}
      <rect class="sl-arena" x={2} y={2} width={POSTER_V - 4} height={POSTER_V - 4} rx={6} />
      <g class="sl-grid">
        {Array.from({ length: 9 }, (_, i) => ((i + 1) * POSTER_V) / 10).map((v) => (
          <>
            <line x1={v} y1={2} x2={v} y2={POSTER_V - 2} />
            <line x1={2} y1={v} x2={POSTER_V - 2} y2={v} />
          </>
        ))}
      </g>

      {/* the orbit the box traces at a stable timestep */}
      <circle class="sl-orbit" cx={ox} cy={oy} r={orbitR} />

      {/* the spring anchor at the origin */}
      <circle class="sl-anchor-ring" cx={ox} cy={oy} r={0.03 * POSTER_S} />
      <circle class="sl-anchor-dot" cx={ox} cy={oy} r={3.2} />

      {/* the restoring force, drawn from the box toward the anchor (invisible-made-visible) */}
      <line class="sl-force" x1={bx} y1={by} x2={ox + (bx - ox) * 0.28} y2={oy + (by - oy) * 0.28} />

      {/* the box */}
      <rect
        class="sl-box"
        x={bx - boxSide / 2}
        y={by - boxSide / 2}
        width={boxSide}
        height={boxSide}
        rx={3}
      />

      {/* the one live control, as an affordance */}
      <text class="sl-hint" x={w2s(-0.4, -0.34)[0]} y={w2s(-0.4, -0.34)[1]}>
        drag the timestep up →
      </text>
    </svg>
  );
}

// ------------------------------------------------------------------- live island
interface Hud {
  speed: number; // m/s
  ratio: number; // energy / baseline energy at reset
  unstable: boolean;
  exploded: boolean;
}

type Tok = "--ink" | "--ink-mute" | "--signal" | "--alert" | "--rule-strong" | "--paper";
const TOK_FALLBACK: Record<Tok, string> = {
  "--ink": "#23201b",
  "--ink-mute": "#6d6252",
  "--signal": "#1f56de",
  "--alert": "#c0362a",
  "--rule-strong": "#c8bc9e",
  "--paper": "#f7f3ea",
};

function SimLoopToy() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dtRef = useRef(DT_DEFAULT);
  const runningRef = useRef(true);
  const apiRef = useRef<{
    setDt: (dt: number) => void;
    reset: () => void;
    setRunning: (r: boolean) => void;
  } | null>(null);

  const [booted, setBooted] = useState(false);
  const [dt, setDt] = useState(DT_DEFAULT);
  const [running, setRunning] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hud, setHud] = useState<Hud>({ speed: 0, ratio: 1, unstable: false, exploded: false });

  useEffect(() => {
    // Respect reduced-motion: don't autoplay the orbit; the learner opts in with Play.
    const reduce =
      typeof window !== "undefined" &&
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      runningRef.current = false;
      setRunning(false);
    }

    let disposed = false;
    let sim: { dispose(): void } | null = null;
    let raf = 0;

    (async () => {
      try {
        // --- lazy, hydration-gated: the MuJoCo-WASM module is fetched only now ---
        const { createSim } = await import("../../../../playground/src/sim/mujoco_sim");

        const realSim: any = await createSim(SCENE_XML);
        sim = realSim;
        if (disposed) { realSim.dispose(); return; }
        const { module, model, data } = realSim;

        // resolve joint/body addresses once (free joint => 7 qpos, 6 qvel slots)
        const BODY = module.mjtObj.mjOBJ_BODY.value;
        const JOINT = module.mjtObj.mjOBJ_JOINT.value;
        const boxBody = module.mj_name2id(model, BODY, "box");
        const boxJoint = module.mj_name2id(model, JOINT, "box_free");
        const qadr = model.jnt_qposadr[boxJoint] as number; // qpos[qadr..+2]=xyz, +3..+6=quat wxyz
        const dadr = model.jnt_dofadr[boxJoint] as number; // qvel[dadr..+2]=linear, +3..+5=angular
        const fbase = boxBody * 6; // xfrc_applied row for the box (fx,fy,fz,tx,ty,tz)

        const applyDt = (v: number) => {
          // mjModel is "fixed while it runs", but the timestep is ours to set between
          // steps (exactly what sim_loop.py does: model.opt.timestep = args.timestep).
          const opt = model.opt;
          opt.timestep = v;
          opt.delete();
        };
        applyDt(dtRef.current);

        let energy0 = 1;
        const energy = () => {
          const x = data.qpos[qadr], y = data.qpos[qadr + 1];
          const vx = data.qvel[dadr], vy = data.qvel[dadr + 1];
          return 0.5 * BOX_MASS * (vx * vx + vy * vy) + 0.5 * K_SPRING * (x * x + y * y);
        };

        const trail: Array<[number, number]> = [];
        let exploded = false;
        let last: [number, number] = [INIT_R, 0];

        const reset = () => {
          module.mj_resetData(model, data);
          data.qpos[qadr] = INIT_R; data.qpos[qadr + 1] = 0; data.qpos[qadr + 2] = INIT_Z;
          data.qpos[qadr + 3] = 1; data.qpos[qadr + 4] = 0; data.qpos[qadr + 5] = 0; data.qpos[qadr + 6] = 0;
          data.qvel[dadr] = 0; data.qvel[dadr + 1] = INIT_VY; data.qvel[dadr + 2] = 0;
          data.qvel[dadr + 3] = 0; data.qvel[dadr + 4] = 0; data.qvel[dadr + 5] = 0;
          data.xfrc_applied[fbase] = 0; data.xfrc_applied[fbase + 1] = 0; data.xfrc_applied[fbase + 2] = 0;
          module.mj_forward(model, data);
          energy0 = Math.max(energy(), 1e-6);
          trail.length = 0;
          exploded = false;
          last = [INIT_R, 0];
        };
        reset();

        // one physics frame: n Euler substeps, re-deriving the spring force each step
        const stepN = (n: number) => {
          for (let i = 0; i < n && !exploded; i++) {
            const x = data.qpos[qadr], y = data.qpos[qadr + 1];
            const vx = data.qvel[dadr], vy = data.qvel[dadr + 1];
            // F = -K x - C v : this is the "conversation" — every step reads mjData's
            // state and writes the next force, then mj_step integrates one dt forward.
            data.xfrc_applied[fbase] = -K_SPRING * x - C_DAMP * vx;
            data.xfrc_applied[fbase + 1] = -K_SPRING * y - C_DAMP * vy;
            module.mj_step(model, data);
            const nx = data.qpos[qadr], ny = data.qpos[qadr + 1];
            if (!Number.isFinite(nx) || !Number.isFinite(ny) || Math.hypot(nx, ny) > EXPLODE_R) {
              exploded = true;
              break;
            }
            last = [nx, ny];
          }
        };

        // --- bespoke top-down renderer (box + table in ink; anchor in signal) ---
        const canvas = canvasRef.current!;
        canvas.width = CANVAS_PX;
        canvas.height = CANVAS_PX;
        const ctx = canvas.getContext("2d")!;
        const cs = getComputedStyle(canvas);
        const col = (k: Tok) => (cs.getPropertyValue(k).trim() || TOK_FALLBACK[k]);
        const INK = col("--ink"), INK_MUTE = col("--ink-mute"), SIGNAL = col("--signal"),
          ALERT = col("--alert"), RULE = col("--rule-strong");

        const scale = CANVAS_PX / (2 * WORLD_HALF);
        const toPx = (x: number, y: number): [number, number] => [
          CANVAS_PX / 2 + x * scale,
          CANVAS_PX / 2 - y * scale,
        ];

        const render = () => {
          ctx.fillStyle = "#fbf9f3";
          ctx.fillRect(0, 0, CANVAS_PX, CANVAS_PX);
          // graph paper
          ctx.strokeStyle = RULE;
          ctx.globalAlpha = 0.4;
          ctx.lineWidth = 1;
          ctx.beginPath();
          for (let i = 1; i < 10; i++) {
            const p = (i * CANVAS_PX) / 10;
            ctx.moveTo(p, 0); ctx.lineTo(p, CANVAS_PX);
            ctx.moveTo(0, p); ctx.lineTo(CANVAS_PX, p);
          }
          ctx.stroke();
          ctx.globalAlpha = 1;

          const unstableNow = exploded || energy() / energy0 > UNSTABLE_RATIO;

          // orbit trail — smooth ring when stable, a divergent spiral when not
          if (trail.length > 1) {
            ctx.save();
            ctx.lineWidth = 2;
            ctx.strokeStyle = unstableNow ? ALERT : INK_MUTE;
            ctx.beginPath();
            trail.forEach(([x, y], i) => {
              const [px, py] = toPx(x, y);
              i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
            });
            ctx.globalAlpha = 0.45;
            ctx.stroke();
            ctx.restore();
          }

          // the spring anchor at the origin — the one fixed reference (signal blue)
          const [ax, ay] = toPx(0, 0);
          ctx.save();
          ctx.setLineDash([3, 3]);
          ctx.strokeStyle = SIGNAL;
          ctx.globalAlpha = 0.85;
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.arc(ax, ay, 0.03 * scale, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
          ctx.fillStyle = SIGNAL;
          ctx.beginPath();
          ctx.arc(ax, ay, 3.4, 0, Math.PI * 2);
          ctx.fill();

          // box position (clamp to the frame edge if it has already blown up)
          let bx = data.qpos[qadr], by = data.qpos[qadr + 1];
          if (!Number.isFinite(bx) || !Number.isFinite(by)) { bx = last[0]; by = last[1]; }
          const r = Math.hypot(bx, by);
          const drawR = Math.min(r, WORLD_HALF * 0.98);
          const dbx = r > 1e-6 ? (bx / r) * drawR : bx;
          const dby = r > 1e-6 ? (by / r) * drawR : by;

          // restoring-force ray from box toward the anchor (invisible-made-visible)
          const [pbx, pby] = toPx(dbx, dby);
          ctx.save();
          ctx.strokeStyle = SIGNAL;
          ctx.globalAlpha = 0.6;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(pbx, pby);
          ctx.lineTo(pbx + (ax - pbx) * 0.32, pby + (ay - pby) * 0.32);
          ctx.stroke();
          ctx.restore();

          // the box itself — neutral ink, red once unstable
          const side = 0.09 * scale;
          ctx.save();
          ctx.translate(pbx, pby);
          ctx.fillStyle = unstableNow ? ALERT : INK;
          ctx.strokeStyle = unstableNow ? ALERT : INK;
          ctx.beginPath();
          ctx.rect(-side / 2, -side / 2, side, side);
          ctx.fill();
          ctx.restore();
        };

        // publish the interaction API for the React controls
        apiRef.current = {
          setDt: (v) => { dtRef.current = v; applyDt(v); },
          reset,
          setRunning: (r) => { runningRef.current = r; },
        };

        setBooted(true);
        render();

        // --- the animation loop: advance the sim (when running) + render + HUD ----
        let hudMark = 0;
        const frame = () => {
          if (disposed) return;
          if (runningRef.current && !exploded) {
            const d = dtRef.current;
            const n = Math.max(1, Math.min(MAX_SUBSTEPS, Math.round(SIM_PER_FRAME / d)));
            stepN(n);
            const bx = data.qpos[qadr], by = data.qpos[qadr + 1];
            if (Number.isFinite(bx) && Number.isFinite(by)) {
              trail.push([bx, by]);
              if (trail.length > TRAIL_MAX) trail.shift();
            }
          }
          render();

          const now = performance.now();
          if (now - hudMark >= 100) { // ~10 Hz HUD
            hudMark = now;
            const vx = data.qvel[dadr], vy = data.qvel[dadr + 1];
            const speed = Math.hypot(vx, vy);
            const ratio = energy() / energy0;
            setHud({
              speed: Number.isFinite(speed) ? speed : Infinity,
              ratio: Number.isFinite(ratio) ? ratio : Infinity,
              unstable: exploded || ratio > UNSTABLE_RATIO,
              exploded,
            });
          }
          raf = requestAnimationFrame(frame);
        };
        raf = requestAnimationFrame(frame);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error("[ch0.1 toy] failed", err);
        setError(msg);
      }
    })();

    return () => { disposed = true; cancelAnimationFrame(raf); if (sim) sim.dispose(); };
  }, []);

  const failed = !!error;
  const meterPct = Math.max(0, Math.min(100, ((hud.ratio - 1) / (METER_MAX_RATIO - 1)) * 100));
  const ratioText = !Number.isFinite(hud.ratio) ? "∞" : `${hud.ratio.toFixed(2)}×`;
  const speedText = !Number.isFinite(hud.speed) ? "∞" : hud.speed.toFixed(2);

  const onDt = (e: Event) => {
    const v = parseFloat((e.currentTarget as HTMLInputElement).value);
    setDt(v);
    apiRef.current?.setDt(v);
  };
  const onReset = () => {
    apiRef.current?.reset();
    setHud({ speed: 0, ratio: 1, unstable: false, exploded: false });
  };
  const onToggleRun = () => {
    const next = !running;
    setRunning(next);
    apiRef.current?.setRunning(next);
  };

  return (
    <div class="sl">
      <figure
        class="sl-figure"
        aria-label="Top-down simulation view: a box orbiting a spring anchor. Its stability is controlled by the timestep slider below."
      >
        {/* SSR poster — the JS-off experience and the pre-boot frame */}
        <div class="sl-poster" hidden={booted}><Poster /></div>

        {/* live MuJoCo-WASM canvas — shown once booted */}
        <canvas ref={canvasRef} class="sl-canvas" hidden={!booted} aria-hidden="true" />

        {/* Non-visual path to the outcome: announce only the qualitative stability
            transition (not the per-frame energy ratio). The visual HUD is aria-hidden. */}
        <div class="bk-sr" aria-live="polite">
          {booted && !failed
            ? hud.exploded
              ? "The simulation blew up: this timestep is too large for the integrator and the energy is diverging."
              : hud.unstable
                ? "Going unstable: the timestep is getting too large and the energy is climbing."
                : "Stable: the motion is faithful at this timestep."
            : ""}
        </div>

        {/* live HUD — energy relative to the reset baseline (invisible made visible) */}
        {booted && !failed && (
          <div class="sl-hud" aria-hidden="true">
            <div class="sl-hud-row">
              <span class="sl-k">energy (× baseline)</span>
              <span class={`sl-v ${hud.unstable ? "sl-bad" : "sl-ok"}`}>
                {ratioText} {hud.unstable ? "▲" : "✓"}
              </span>
            </div>
            <div class="sl-meter">
              <div class="sl-meter-fill" data-unstable={hud.unstable} style={`width:${meterPct}%`} />
            </div>
            <div class="sl-hud-row">
              <span class="sl-k">integrator</span>
              <span class={`sl-v ${hud.unstable ? "sl-bad" : "sl-ok"}`}>
                {hud.exploded ? "exploded" : hud.unstable ? "going unstable" : "stable"}
              </span>
            </div>
            <div class="sl-hud-row">
              <span class="sl-k">box speed · mjData.qvel</span>
              <span class="sl-v">{speedText} m/s</span>
            </div>
          </div>
        )}

        {/* boot / status line */}
        <div class="sl-status" data-failed={failed} data-unstable={hud.unstable} aria-hidden="true">
          {failed ? (
            <span>sim failed — the Colab path covers this without WASM</span>
          ) : booted ? (
            <>
              <span>mjModel.opt.timestep = {dt.toFixed(3)} s</span>
              <span>{hud.exploded ? "reset to recover" : hud.unstable ? "too big a step" : "faithful"}</span>
            </>
          ) : (
            <span>booting MuJoCo-WASM…</span>
          )}
        </div>
      </figure>

      {/* THE one control — the timestep slider (keyboard-native) — plus reset/play */}
      <div class="sl-controls">
        <div class="sl-slider-row">
          <label class="sl-slider-label" for="sl-dt">
            timestep <b>dt</b>
          </label>
          <input
            id="sl-dt"
            class="sl-slider"
            type="range"
            min={DT_MIN}
            max={DT_MAX}
            step={DT_STEP}
            value={dt}
            onInput={onDt}
            disabled={!booted || failed}
            aria-label="Simulation timestep in seconds. Drag up to make the integrator unstable and watch the box explode."
            aria-valuetext={`${dt.toFixed(3)} seconds`}
          />
          <span class="sl-dt-val" data-unstable={hud.unstable}>{dt.toFixed(3)} s</span>
        </div>
        <div class="sl-btn-row">
          <button type="button" class="sl-btn" onClick={onReset} disabled={!booted || failed}>
            reset
          </button>
          <button type="button" class="sl-btn" onClick={onToggleRun} disabled={!booted || failed}>
            {running ? "pause" : "play"}
          </button>
          <span class="sl-note">
            small dt → smooth · big dt → it explodes · poster reads with JS off
          </span>
        </div>
      </div>
    </div>
  );
}

export default function SimLoopPerturb() {
  return <SimLoopToy />;
}
