/**
 * RealLoopToy — ch5.8 "The Real Loop" concept-toy (demo id `so101_reach_replay`).
 * THE graduation loop, made honest as a pure DATA-VIEWER over a precomputed SO-101
 * rollout — no MuJoCo-WASM, no ONNX, no dynamic import. This chapter runs the WHOLE
 * LeRobot loop (drive → record → train → deploy → eval) on the SO-101's real
 * morphology; the artifact emits demo/vizdata.json (two recorded gripper-tip paths +
 * the ch1.6 success-rate headline), and this is the cheapest embed tier: it reads the
 * JSON and draws two paths + three bars.
 *
 * WHAT IT SHOWS. A top-down (x, y) plot of the SO-101's reachable patch. The held-out
 * box (vizdata.box, a green square; success_tol drawn as its halo). The CLONE's
 * gripper-tip path curls in and lands ON the box — a reach, inside the tolerance. A
 * TRAINED / BROKEN toggle swaps to the `--break obs_swap` rollout: SAME box, SAME
 * trained weights — the only difference is the order the deploy script read box_x /
 * box_y, so the tip curls to the MIRRORED side and stops short. A scrubber replays the
 * rollout frame-by-frame and walks the drive→record→train→deploy→eval loop diagram in
 * turn. Below, three success-rate bars (clone ≫ no-op / random) make the ch1.6 headline
 * visible: the LOOP closes on the real body.
 *
 * All numbers + geometry are REAL: read verbatim from real_loop.py's committed seed-0
 * vizdata.json. Nothing is mocked.
 *
 * HONESTY (matches the chapter). This is the loop closing on the SO-101's real
 * kinematics + the real LeRobot dataset format — a *reach in sim*. It is NOT
 * hardware fidelity: servo backlash, unmodeled friction, read→send latency, and
 * calibration drift are the reality gap, and they stay reading — the graduation G1
 * module. The toy never claims dexterity or torque fidelity; a clean 100% here is the
 * loop working, not manipulation solved.
 *
 * Pure inline SVG + design tokens: theme-aware for free (light AND dark), and the
 * server-rendered default (the clone's landed reach + the three success bars + the
 * honest note) IS the JS-off experience. Hydration only adds the trained/broken
 * toggle and the frame scrubber.
 *
 * Follows the FROZEN CONCEPT-TOY CONTRACT documented at the top of ../PlateIsland.tsx.
 */
import "./RealLoopToy.css";
import { useMemo, useState } from "preact/hooks";
// Real recorded SO-101 rollout + ch1.6 success rates from real_loop.py's reference
// run (seed 0, default config) — committed small text (numeric paths), no binary.
// Same co-located-vizdata pattern the other data-viewer toys use.
import viz from "../../../../curriculum/phase5_practitioner/ch5.8_real_loop/demo/vizdata.json";

// ---------------------------------------------------------------- typed vizdata
const TASK: string = viz.task;                       // the reach instruction recorded into every frame
const BOX: number[] = viz.box as number[];           // [x, y, z] — held-out box placement (eval seed 10_000)
const SUCCESS_TOL: number = viz.success_tol;         // 0.08 m — gripper-to-box distance drawn as the halo
const CLONE_RATE: number = viz.clone_rate;           // ~1.0 — the headline
const NOOP_RATE: number = viz.noop_rate;             // ~0.03 — no-op baseline
const RANDOM_RATE: number = viz.random_rate;         // ~0.00 — random baseline
const CLONE: number[][] = viz.clone_tip_path as number[][]; // [x,y,z] per control step — the reach
const BREAK: number[][] = viz.break_tip_path as number[][]; // [x,y,z] per control step — obs_swap miss

type Mode = "clone" | "break";
const PATH: Record<Mode, number[][]> = { clone: CLONE, break: BREAK };

// the drive → record → train → deploy → eval loop the whole chapter is about; the
// scrubber walks these in turn as it steps through the recorded rollout.
const STAGES: { key: string; note: string }[] = [
  { key: "drive", note: "teleop the arm" },
  { key: "record", note: "LeRobot dataset" },
  { key: "train", note: "BC on joints" },
  { key: "deploy", note: "policy → arm" },
  { key: "eval", note: "did it reach?" },
];

// ------------------------------------------------------------ number formatting
const m3 = (v: number) => `${v.toFixed(3)} m`;
const pct = (v: number) => `${+(v * 100).toFixed(1)}%`;

/** full 3-D gripper-tip → box distance — exactly what success_tol (0.08 m) measures. */
function dist3(tip: number[], box: number[]): number {
  return Math.hypot(tip[0] - box[0], tip[1] - box[1], tip[2] - box[2]);
}

// ===================================================================== GEOMETRY
// A top-down (x, y) plot, EQUAL-ASPECT so the success_tol halo is a true circle and
// the mirrored obs_swap miss reads as a real distance. World +x → right, +y → up.
// Bounds are computed from the real data (both paths + the box + its halo) so nothing
// is hand-placed. SSR renders this verbatim — it is the JS-off view.
const VW = 384;
const VH = 340;
const PAD = { l: 30, r: 14, t: 14, b: 26 };
const PLOT_W = VW - PAD.l - PAD.r;
const PLOT_H = VH - PAD.t - PAD.b;

const BOUNDS = (() => {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const eat = (x: number, y: number) => {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  };
  for (const p of [...CLONE, ...BREAK]) eat(p[0], p[1]);
  // include the box and the full success halo so the tolerance ring is never clipped
  eat(BOX[0] - SUCCESS_TOL, BOX[1] - SUCCESS_TOL);
  eat(BOX[0] + SUCCESS_TOL, BOX[1] + SUCCESS_TOL);
  const padX = (maxX - minX) * 0.08, padY = (maxY - minY) * 0.08;
  return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
})();

const WORLD_W = BOUNDS.maxX - BOUNDS.minX;
const WORLD_H = BOUNDS.maxY - BOUNDS.minY;
const SCALE = Math.min(PLOT_W / WORLD_W, PLOT_H / WORLD_H); // px per metre (shared → equal aspect)
const OX = PAD.l + (PLOT_W - WORLD_W * SCALE) / 2;          // centre the content in the plot
const OY = PAD.t + (PLOT_H - WORLD_H * SCALE) / 2;
const sx = (wx: number) => OX + (wx - BOUNDS.minX) * SCALE;
const sy = (wy: number) => OY + (BOUNDS.maxY - wy) * SCALE; // flip: world +y up
const HALO_R = SUCCESS_TOL * SCALE;                          // the tolerance, drawn as a circle

const [BOX_PX, BOX_PY] = [sx(BOX[0]), sy(BOX[1])];
const BOX_HALF = 0.012 * SCALE; // a ~2.4 cm box drawn to scale (a small target square)

/** a path (up to `n` points) as an SVG polyline point string. */
function poly(path: number[][], n: number): string {
  return path.slice(0, Math.max(1, n)).map((p) => `${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(" ");
}

// gridlines every 0.05 m, snapped to the world bounds — light scale reference only.
const GRID_STEP = 0.05;
const gx: number[] = [];
for (let x = Math.ceil(BOUNDS.minX / GRID_STEP) * GRID_STEP; x < BOUNDS.maxX; x += GRID_STEP) gx.push(x);
const gy: number[] = [];
for (let y = Math.ceil(BOUNDS.minY / GRID_STEP) * GRID_STEP; y < BOUNDS.maxY; y += GRID_STEP) gy.push(y);

// ==================================================================== THE ISLAND
export default function RealLoopToy() {
  // default-interesting: the CLONE rollout at its LAST frame — the tip landed ON the
  // box, the reach done. That is the aha, and it is what SSR renders, so the JS-off
  // view already tells the whole story (landed reach + the success bars below).
  const [mode, setMode] = useState<Mode>("clone");
  const path = PATH[mode];
  const [frameIdx, setFrameIdx] = useState(CLONE.length - 1);

  // clamp the shared frame index into the active path (clone/break differ by a frame)
  const maxFrame = path.length - 1;
  const fi = Math.min(frameIdx, maxFrame);
  const tip = path[fi];
  const d = dist3(tip, BOX);
  const within = d <= SUCCESS_TOL;
  const finalReached = dist3(path[maxFrame], BOX) <= SUCCESS_TOL;

  // scrub progress walks the 5-stage loop diagram in turn (drive…eval)
  const frac = maxFrame > 0 ? fi / maxFrame : 1;
  const stageIdx = Math.max(0, Math.min(STAGES.length - 1, Math.floor(frac * STAGES.length)));

  const clone = mode === "clone";

  const svgLabel =
    `Top-down x-y plot of the SO-101 reachable patch. The held-out box sits at x ${BOX[0].toFixed(2)}, ` +
    `y ${BOX[1].toFixed(2)} metres, ringed by the ${SUCCESS_TOL} metre reach tolerance. ` +
    (clone
      ? `The clone's recorded gripper-tip path curls in and lands on the box — a reach, ` +
        `ending ${m3(dist3(CLONE[CLONE.length - 1], BOX))} from it, inside the tolerance.`
      : `The obs_swap rollout uses the same box and the same trained weights, but the deploy script ` +
        `read box_x and box_y in the wrong order, so the tip curls to the mirrored side and stops ` +
        `${m3(dist3(BREAK[BREAK.length - 1], BOX))} short.`) +
    ` Currently at frame ${fi + 1} of ${path.length}; the tip is ${m3(d)} from the box.`;

  const announce =
    `${clone ? "Clone" : "Broken (obs_swap)"} rollout, frame ${fi + 1} of ${path.length}. ` +
    `Loop stage: ${STAGES[stageIdx].key}. Gripper tip ${m3(d)} from the box — ` +
    `${within ? `within the ${SUCCESS_TOL} metre reach tolerance` : `beyond the ${SUCCESS_TOL} metre tolerance`}. ` +
    (clone
      ? "The recorded-then-cloned policy reproduces the reach on the arm's real body."
      : "Same weights, same box — only the deploy script's box_x / box_y order changed, and the reach misses.");

  const onKeyDown = (e: KeyboardEvent) => {
    const k = e.key;
    if (k === "t" || k === "T") { e.preventDefault(); setMode((mo) => (mo === "clone" ? "break" : "clone")); return; }
    if (k === "ArrowRight" || k === "]" || k === "." || k === "PageDown") {
      e.preventDefault(); setFrameIdx(() => Math.min(maxFrame, fi + 1)); return;
    }
    if (k === "ArrowLeft" || k === "[" || k === "," || k === "PageUp") {
      e.preventDefault(); setFrameIdx(() => Math.max(0, fi - 1)); return;
    }
    if (k === "Home") { e.preventDefault(); setFrameIdx(0); return; }
    if (k === "End") { e.preventDefault(); setFrameIdx(maxFrame); return; }
  };

  const setMode2 = (mo: Mode) => { setMode(mo); setFrameIdx(PATH[mo].length - 1); };

  // rate bars — the ch1.6 headline. clone ≫ no-op / random.
  const bars = useMemo(
    () => [
      { key: "clone", label: "cloned policy", rate: CLONE_RATE, kind: "hero" as const },
      { key: "noop", label: "no-op baseline", rate: NOOP_RATE, kind: "base" as const },
      { key: "random", label: "random baseline", rate: RANDOM_RATE, kind: "base" as const },
    ],
    [],
  );

  return (
    <div class="rl">
      <header class="rl-head">
        <h3 class="rl-title">The loop, deployed two ways</h3>
        <p class="rl-sub">
          The whole LeRobot loop — <b>drive → record → train → deploy → eval</b> — run on the{" "}
          <b>SO-101's real morphology</b>. The <b>clone</b> curls in and lands on the box: a <b>reach</b>. The{" "}
          <b>broken</b> rollout uses the <b>same box and the same trained weights</b> — the only difference is the
          order the deploy script read <b>box_x / box_y</b>, and the tip mirrors to the wrong side.
        </p>
      </header>

      {/* the drive→record→train→deploy→eval loop; the scrubber lights each stage in turn */}
      <ol class="rl-loop" aria-label={`Graduation loop, currently at the ${STAGES[stageIdx].key} stage`}>
        {STAGES.map((s, i) => (
          <li class="rl-stage" data-active={i === stageIdx} data-done={i < stageIdx}>
            <span class="rl-stage-k">{s.key}</span>
            <span class="rl-stage-n">{s.note}</span>
          </li>
        ))}
      </ol>

      <div class="rl-stage-round" aria-hidden="true">↺ the loop closes — eval feeds the next drive</div>

      {/* the top-down replay — SSR renders the clone's landed reach (the JS-off view) */}
      <figure
        class="rl-fig"
        tabIndex={0}
        role="group"
        aria-label="Interactive SO-101 reach replay. Press T to toggle the clone versus the broken obs_swap rollout, and left/right arrow keys to scrub the recorded rollout frame by frame."
        onKeyDown={onKeyDown}
      >
        <svg class="rl-svg" viewBox={`0 0 ${VW} ${VH}`} role="img" aria-label={svgLabel} data-mode={mode}>
          <title>SO-101 gripper-tip reach — top-down replay</title>

          {/* plot frame + faint scale grid (0.05 m) */}
          <rect class="rl-plot" x={PAD.l} y={PAD.t} width={PLOT_W} height={PLOT_H} rx={4} />
          <g class="rl-grid">
            {gx.map((x) => <line x1={sx(x)} y1={PAD.t} x2={sx(x)} y2={PAD.t + PLOT_H} />)}
            {gy.map((y) => <line x1={PAD.l} y1={sy(y)} x2={PAD.l + PLOT_W} y2={sy(y)} />)}
          </g>

          {/* the reach tolerance, drawn as the box's halo */}
          <circle class="rl-halo" cx={BOX_PX} cy={BOX_PY} r={HALO_R} />
          {/* the held-out box — the green target both rollouts aim for */}
          <rect
            class="rl-box"
            x={BOX_PX - BOX_HALF}
            y={BOX_PY - BOX_HALF}
            width={BOX_HALF * 2}
            height={BOX_HALF * 2}
            rx={1.5}
          >
            <title>box · x {BOX[0].toFixed(3)} m, y {BOX[1].toFixed(3)} m · reach tolerance {SUCCESS_TOL} m</title>
          </rect>

          {/* the full recorded path (faint ghost) + the traced portion up to the scrub frame */}
          <polyline class="rl-ghost" data-mode={mode} points={poly(path, path.length)} />
          <polyline class="rl-trace" data-mode={mode} points={poly(path, fi + 1)} />

          {/* start dot + the moving gripper tip */}
          <circle class="rl-start" cx={sx(path[0][0])} cy={sy(path[0][1])} r={3} />
          <circle class="rl-tip" data-mode={mode} data-within={within} cx={sx(tip[0])} cy={sy(tip[1])} r={4.5} />

          {/* axis caption */}
          <text class="rl-axcap" x={PAD.l} y={VH - 8}>top-down · SO-101 workspace (x, y) · metres</text>
        </svg>

        <figcaption class="rl-cap" aria-hidden="true">
          {clone ? "clone" : "broken · obs_swap"} · frame {fi + 1}/{path.length} · tip{" "}
          <b data-within={within}>{m3(d)}</b> from box{" "}
          {fi === maxFrame && (finalReached ? <b class="rl-ok">reached ✓</b> : <b class="rl-miss">short ✗</b>)}
        </figcaption>
      </figure>

      {/* --- controls: trained/broken toggle + frame scrubber (keyboard-accessible) --- */}
      <div class="rl-controls">
        <div class="rl-toggle" role="group" aria-label="Rollout: clone versus broken obs_swap">
          <button type="button" class="rl-tbtn" data-on={clone} aria-pressed={clone} onClick={() => setMode2("clone")}>
            clone · reaches
          </button>
          <button type="button" class="rl-tbtn" data-on={!clone} aria-pressed={!clone} onClick={() => setMode2("break")}>
            broken · obs_swap
          </button>
        </div>

        <div class="rl-scrub">
          <label class="rl-scrub-lbl" for="rl-frame">frame</label>
          <input
            id="rl-frame"
            class="rl-slider"
            type="range"
            min={0}
            max={maxFrame}
            step={1}
            value={fi}
            onInput={(e) => setFrameIdx(Number((e.target as HTMLInputElement).value))}
            aria-valuetext={
              `frame ${fi + 1} of ${path.length}, ${clone ? "clone" : "broken"} rollout — ` +
              `tip ${m3(d)} from the box, ${within ? "within" : "beyond"} the ${SUCCESS_TOL} metre tolerance`
            }
          />
          <output class="rl-scrub-out" for="rl-frame">{fi + 1}/{path.length}</output>
        </div>

        <span class="rl-control-note">toggle clone/broken · scrub the rollout · poster reads with JS off</span>
      </div>

      {/* box readout — always visible (JS-off friendly), no hover required */}
      <div class="rl-readout" aria-hidden="true">
        <span class="rl-ro"><span class="rl-ro-k">box (held-out)</span><span class="rl-ro-v">x {BOX[0].toFixed(3)} · y {BOX[1].toFixed(3)} m</span></span>
        <span class="rl-ro"><span class="rl-ro-k">reach tolerance</span><span class="rl-ro-v">{SUCCESS_TOL} m</span></span>
        <span class="rl-ro"><span class="rl-ro-k">tip → box</span><span class="rl-ro-v" data-within={within}>{m3(d)} {within ? "✓" : "✗"}</span></span>
      </div>

      {/* THE HEADLINE — three success-rate bars (ch1.6): clone ≫ no-op / random */}
      <figure class="rl-rates">
        <figcaption class="rl-rates-cap">Does the loop close? · success rate over 30 held-out eval episodes</figcaption>
        <div class="rl-bars">
          {bars.map((b) => (
            <div class="rl-bar-row" data-kind={b.kind}>
              <span class="rl-bar-name">{b.label}</span>
              <div class="rl-bar-track">
                <div class="rl-bar-fill" data-kind={b.kind} style={`width:${Math.max(b.rate * 100, b.rate > 0 ? 1.5 : 0)}%`} />
              </div>
              <span class="rl-bar-val" data-kind={b.kind}>{pct(b.rate)}</span>
            </div>
          ))}
        </div>
        <p class="rl-rates-note">
          The recorded-then-cloned policy reproduces the reach on the arm's real body, clearly above doing nothing or
          flailing — the <b>loop closes</b>. A <b>direction</b>, not a headline %: this is the loop working, not
          manipulation solved.
        </p>
      </figure>

      {/* non-visual path to the same aha — the qualitative story, not per-frame spam */}
      <div class="bk-sr" aria-live="polite">{announce}</div>

      {/* the honest framing — the load-bearing caveat, straight from the chapter */}
      <p class="rl-note">
        This is a <b>reach in sim</b> on the SO-101's real <b>kinematics</b> and the real <b>LeRobot dataset
        format</b> — the loop <b>closing</b> on the arm's body, not hardware fidelity. What the twin cannot give you
        is the <b>reality gap</b>: servo backlash, unmodeled friction, read-to-send latency, calibration drift. A
        clean {pct(CLONE_RATE)} here does not paper over it. Those gaps stay reading — the graduation <b>G1</b>
        module, where you buy the arm. The broken rollout is the same lesson as ch0.4 (<b>you can only imitate the
        obs you recorded</b>) with a physical consequence. Real recorded rollout + rates from real_loop.py
        (seed 0, cpu); poster reads with JS off. Task: “{TASK}”.
      </p>
    </div>
  );
}
