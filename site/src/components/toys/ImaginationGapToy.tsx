/**
 * ImaginationGapToy — ch3.2 "World Models II: Acting in Imagination" concept-toy
 * (demo id `imagined_vs_real`). THE IMAGINATION GAP, told HONESTLY.
 *
 * A recorded-data sibling of ch3.1's WorldModelToy and ch3.3's EngineDriftToy: no
 * WASM, pure SVG over numbers dreamer.py itself measured (seed 0, cpu), regenerated
 * by site/scripts/vizdata/ch3.2_dreamer.py into the co-located vizdata.json and gated
 * against meta.yaml's seed-sweep bands (STOP-on-drift).
 *
 * THE ONE THING THIS TOY MUST NOT LET YOU MISBELIEVE — and the misconception it
 * exists to break: "imagination is as good as your world model." A policy trained
 * ENTIRELY inside the learned world model looks like a champion IN IMAGINATION (its
 * imagined return climbs, its dreamed block parks ~0.01 m from the target) and FAILS
 * in reality (real return floors, the real block barely moves ~0.16 m, and real task
 * success is 0% on EVERY seed). Imagination is only as good as your world model — and
 * this one (from 3.1) got the block dynamics wrong. This toy NEVER shows a solved
 * PushT and NEVER implies the policy works; the whole point is the gap.
 *
 * THREE panels (the gap first, deliberately):
 *   1. THE IMAGINATION GAP (headline) — the SAME policy scored in two worlds: two
 *      return bars (imagined ≫ real) + the reality check (0% real success; dream
 *      parks the block ~0.01 m vs real ~0.16 m).
 *   2. THE POLICY DID TRAIN — imagined return climbs as the actor learns to game the
 *      dream, while the SAME policy's REAL return stays flat on the floor: the curve
 *      climbs AWAY from reality. An iteration scrubber reads out the widening delusion.
 *   3. STEP 1 (unchanged from 3.1) — the world model the actor dreams inside:
 *      reconstruction (the easy half) falls fast; the dynamics loss stays low.
 *
 * Colour by MEANING: --signal blue = IMAGINATION (the dream — rosy but DELUDED, never
 * an endorsement; the badges say so), --alert red = REALITY (the honest floor: real
 * return, 0% success, the block that never parks), --ink-mute = neutral instrument.
 * The server-rendered default (both bars + the reality check + both full curves + the
 * final-iteration readout) IS the JS-off experience; only the iteration scrubber goes
 * inert without hydration.
 *
 * Follows the FROZEN CONCEPT-TOY CONTRACT documented at the top of ../PlateIsland.tsx.
 */
import "./ImaginationGapToy.css";
import { useState } from "preact/hooks";
// Real numbers from dreamer.py, seed 0, cpu — see the file's `provenance` and the
// generator site/scripts/vizdata/ch3.2_dreamer.py. Committed small text, no binary.
import viz from "../../../../curriculum/phase3_advanced/ch3.2_dreamer/demo/vizdata.json";

// ---------------------------------------------------------------- typed vizdata
const GAP = viz.gap as {
  imagined_return: number; real_return: number; gap: number;
  imagined_final_tee_dist: number; real_final_tee_dist: number; real_success_rate: number;
};
const CURVE = viz.imagined_return_curve as {
  iters: number[]; reward_per_step: number[]; start: number; final: number; real_return: number;
};
const WM = viz.wm_losses as { steps: number[]; recon: number[]; dyn: number[] };
const CFG = viz.config as { imag_iters: number; eval_episodes: number; eval_horizon: number };
const BAND = viz.seed_band as unknown as Record<string, [number, number]>;

const fmt = (v: number) => v.toFixed(3);
const sgn = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(3)}`;
const pct = (v: number) => `${Math.round(v * 100)}%`;

// ================================================================ PANEL 1: the gap
// Two return bars on a shared axis (0 = perfect at the top; more negative hangs lower).
// The imagined bar hangs LESS (rosier); the real bar hangs to the floor. The vertical
// distance between the two bar-bottoms IS the delusion, drawn as a bracket.
const BW = 300, BH = 232;
const BP = { l: 40, r: 16, t: 26, b: 40 };
const BPW = BW - BP.l - BP.r;
const BPH = BH - BP.t - BP.b;
const RET_MIN = -0.45;                       // holds the real floor (~-0.38) with headroom
const by = (v: number) => BP.t + (v / RET_MIN) * BPH;   // v in [RET_MIN, 0] -> px (0 at top)
const BAR_W = 62;
const IMAG_X = BP.l + BPW * 0.30 - BAR_W / 2;
const REAL_X = BP.l + BPW * 0.72 - BAR_W / 2;

function GapBars() {
  const y0 = by(0);
  const yImag = by(GAP.imagined_return);
  const yReal = by(GAP.real_return);
  const yTicks = [0, -0.1, -0.2, -0.3, -0.4];
  const brX = REAL_X + BAR_W + 4;
  return (
    <svg
      class="ig-svg ig-bars"
      viewBox={`0 0 ${BW} ${BH}`}
      role="img"
      aria-label={
        `The imagination gap. The same policy scored in two worlds over ${CFG.eval_episodes} held-out ` +
        `starts. In imagination it earns ${fmt(GAP.imagined_return)} reward per step; in the real PushT sim ` +
        `it earns only ${fmt(GAP.real_return)} per step — a gap of ${sgn(GAP.gap)}. Imagination is ` +
        `systematically rosier than reality because the policy optimizes a reward read off the block pose the ` +
        `world model hallucinated, and that world model got the block dynamics wrong.`
      }
    >
      {/* y grid + ticks (reward/step) */}
      {yTicks.map((v) => (
        <g>
          <line class="ig-grid" x1={BP.l} y1={by(v)} x2={BW - BP.r} y2={by(v)} />
          <text class="ig-tick" x={BP.l - 5} y={by(v) + 3} text-anchor="end">{v.toFixed(1)}</text>
        </g>
      ))}
      {/* the perfect-return (0) line, at the top */}
      <line class="ig-zero" x1={BP.l} y1={y0} x2={BW - BP.r} y2={y0} />
      <text class="ig-axis-title" x={BP.l - 34} y={BP.t - 8}>reward/step</text>

      {/* imagined bar (the dream — rosy but DELUDED) */}
      <rect class="ig-bar ig-bar--imag" x={IMAG_X} y={y0} width={BAR_W} height={Math.max(0, yImag - y0)} rx={2} />
      <text class="ig-bar-val ig-v--imag" x={IMAG_X + BAR_W / 2} y={yImag - 6} text-anchor="middle">{fmt(GAP.imagined_return)}</text>
      <text class="ig-bar-lab ig-v--imag" x={IMAG_X + BAR_W / 2} y={BH - BP.b + 14} text-anchor="middle">imagined</text>
      <text class="ig-bar-sub" x={IMAG_X + BAR_W / 2} y={BH - BP.b + 26} text-anchor="middle">in the dream</text>

      {/* real bar (the truth — floors) */}
      <rect class="ig-bar ig-bar--real" x={REAL_X} y={y0} width={BAR_W} height={Math.max(0, yReal - y0)} rx={2} />
      <text class="ig-bar-val ig-v--real" x={REAL_X + BAR_W / 2} y={yReal - 6} text-anchor="middle">{fmt(GAP.real_return)}</text>
      <text class="ig-bar-lab ig-v--real" x={REAL_X + BAR_W / 2} y={BH - BP.b + 14} text-anchor="middle">real</text>
      <text class="ig-bar-sub" x={REAL_X + BAR_W / 2} y={BH - BP.b + 26} text-anchor="middle">true PushT</text>

      {/* THE DELUSION — the vertical gap between the two bar-bottoms */}
      <line class="ig-gap-line" x1={brX} y1={yImag} x2={brX} y2={yReal} marker-start="url(#ig-cap)" marker-end="url(#ig-cap)" />
      <text class="ig-gap-lab" x={brX + 4} y={(yImag + yReal) / 2 + 3}>gap {sgn(GAP.gap)}</text>

      <defs>
        <marker id="ig-cap" viewBox="0 0 8 4" refX="4" refY="2" markerWidth="8" markerHeight="4" orient="auto">
          <path class="ig-caphead" d="M0 2 L8 2" />
        </marker>
      </defs>
    </svg>
  );
}

// ============================================================ PANEL 2: it DID train
// Imagined reward/step vs policy-gradient iteration — it climbs as the actor learns to
// game the dream. The SAME policy's REAL return is a flat dashed floor: the curve
// climbs AWAY from reality. That widening distance is the delusion, growing with training.
const CW = 520, CH = 232;
const CP = { l: 46, r: 66, t: 22, b: 34 };
const CPW = CW - CP.l - CP.r;
const CPH = CH - CP.t - CP.b;
const IT_MAX = CURVE.iters[CURVE.iters.length - 1];
const RY_MIN = -0.42, RY_MAX = -0.10;
const ix = (it: number) => CP.l + (it / IT_MAX) * CPW;
const iy = (v: number) => CP.t + ((RY_MAX - v) / (RY_MAX - RY_MIN)) * CPH;

/** imagined reward/step at an arbitrary iteration, linearly interpolated between the
 *  recorded checkpoints (the curve is subsampled; between points we interpolate). */
function imagAt(it: number): number {
  const xs = CURVE.iters, ys = CURVE.reward_per_step;
  if (it <= xs[0]) return ys[0];
  if (it >= xs[xs.length - 1]) return ys[ys.length - 1];
  for (let i = 1; i < xs.length; i++) {
    if (it <= xs[i]) {
      const t = (it - xs[i - 1]) / (xs[i] - xs[i - 1]);
      return ys[i - 1] + t * (ys[i] - ys[i - 1]);
    }
  }
  return ys[ys.length - 1];
}

function TrainCurve({ sel }: { sel: number }) {
  const pts = CURVE.iters.map((it, i) => `${ix(it).toFixed(1)},${iy(CURVE.reward_per_step[i]).toFixed(1)}`).join(" ");
  const yTicks = [-0.10, -0.18, -0.26, -0.34, -0.42];
  const xTicks = [0, IT_MAX * 0.25, IT_MAX * 0.5, IT_MAX * 0.75, IT_MAX].map((v) => Math.round(v));
  const realY = iy(CURVE.real_return);
  const selImag = imagAt(sel);
  const xc = ix(sel);
  return (
    <svg
      class="ig-svg ig-curve"
      viewBox={`0 0 ${CW} ${CH}`}
      role="img"
      aria-label={
        `Imagined reward per step versus policy-gradient iteration. It climbs from about ` +
        `${fmt(CURVE.start)} to ${fmt(CURVE.final)} as the actor learns to drive the hallucinated block ` +
        `toward the target — proof the policy did train. The same policy's real return is a flat floor at ` +
        `${fmt(CURVE.real_return)}: the imagined curve climbs away from reality, so the delusion widens with ` +
        `training. It is learning, but inside a dream whose hard half is wrong.`
      }
    >
      {yTicks.map((v) => (
        <g>
          <line class="ig-grid" x1={CP.l} y1={iy(v)} x2={CW - CP.r} y2={iy(v)} />
          <text class="ig-tick" x={CP.l - 5} y={iy(v) + 3} text-anchor="end">{v.toFixed(2)}</text>
        </g>
      ))}
      {/* x axis */}
      <line class="ig-axis" x1={CP.l} y1={CH - CP.b} x2={CW - CP.r} y2={CH - CP.b} />
      {xTicks.map((t) => (
        <text class="ig-tick" x={ix(t)} y={CH - CP.b + 13} text-anchor="middle">{t}</text>
      ))}
      <text class="ig-axis-title" x={CW - CP.r} y={CH - 4} text-anchor="end">policy-gradient iteration →</text>

      {/* the flat REAL floor — reality does not climb */}
      <line class="ig-real-floor" x1={CP.l} y1={realY} x2={CW - CP.r} y2={realY} />
      <text class="ig-real-lab" x={CW - CP.r + 3} y={realY + 3}>real floor</text>

      {/* the climbing IMAGINED curve */}
      <polyline class="ig-line ig-line--imag" points={pts} />
      <text class="ig-series-lab ig-v--imag" x={CW - CP.r + 3} y={iy(CURVE.final) + 3}>imagined</text>

      {/* scrub cursor + readout dot at the selected iteration */}
      <line class="ig-cursor" x1={xc} y1={CP.t} x2={xc} y2={CH - CP.b} />
      <circle class="ig-dot ig-v--imag" cx={xc} cy={iy(selImag)} r={3.6} />
      <circle class="ig-dot ig-v--real" cx={xc} cy={realY} r={3.6} />
    </svg>
  );
}

// ============================================================ PANEL 3: the WM (step 1)
const LW = 300, LH = 168;
const LP = { l: 40, r: 14, t: 16, b: 28 };
const LPW = LW - LP.l - LP.r;
const LPH = LH - LP.t - LP.b;
const L_MAX = Math.ceil((Math.max(...WM.recon) * 1.05) / 0.1) * 0.1;
const lx = (i: number) => LP.l + (WM.steps[i] / WM.steps[WM.steps.length - 1]) * LPW;
const ly = (v: number) => LP.t + (1 - v / L_MAX) * LPH;

function WmLosses() {
  const reconPts = WM.recon.map((v, i) => `${lx(i).toFixed(1)},${ly(v).toFixed(1)}`).join(" ");
  const dynPts = WM.dyn.map((v, i) => `${lx(i).toFixed(1)},${ly(v).toFixed(1)}`).join(" ");
  const yTicks = [0, L_MAX / 2, L_MAX];
  return (
    <svg
      class="ig-svg ig-losses"
      viewBox={`0 0 ${LW} ${LH}`}
      role="img"
      aria-label={
        `World-model training losses over steps. Reconstruction — the easy half — falls fast from about ` +
        `${fmt(WM.recon[0])} to ${fmt(WM.recon[WM.recon.length - 1])}. The dynamics loss stays low. This is ` +
        `step one, unchanged from chapter 3.1: the world model the actor will later dream inside.`
      }
    >
      {yTicks.map((v) => (
        <g>
          <line class="ig-grid" x1={LP.l} y1={ly(v)} x2={LW - LP.r} y2={ly(v)} />
          <text class="ig-tick" x={LP.l - 5} y={ly(v) + 3} text-anchor="end">{v.toFixed(1)}</text>
        </g>
      ))}
      <line class="ig-axis" x1={LP.l} y1={LH - LP.b} x2={LW - LP.r} y2={LH - LP.b} />
      <text class="ig-axis-title" x={LW - LP.r} y={LH - 3} text-anchor="end">wm step →</text>
      <polyline class="ig-line ig-line--recon" points={reconPts} />
      <polyline class="ig-line ig-line--dyn" points={dynPts} />
      <text class="ig-series-lab ig-l--recon" x={lx(3) + 4} y={ly(WM.recon[3]) - 4}>reconstruction</text>
      <text class="ig-series-lab ig-l--dyn" x={LW - LP.r - 2} y={ly(WM.dyn[WM.dyn.length - 1]) - 5} text-anchor="end">dynamics</text>
    </svg>
  );
}

// ==================================================================== THE ISLAND
export default function ImaginationGapToy() {
  // default-interesting: the FINAL iteration — the delusion is widest (the policy has
  // trained the most, so imagination is rosiest while reality is unmoved). SSR renders
  // here, so this IS the JS-off view; only the scrubber needs hydration.
  const [sel, setSel] = useState<number>(IT_MAX);
  const selImag = imagAt(sel);
  const selDelusion = selImag - CURVE.real_return;

  const announce =
    `The policy trained entirely inside the world model believes it earns ${fmt(GAP.imagined_return)} reward per ` +
    `step in imagination, but earns only ${fmt(GAP.real_return)} in the real PushT sim — a gap of ${sgn(GAP.gap)}. ` +
    `In the dream the block parks ${fmt(GAP.imagined_final_tee_dist)} m from the target; in reality it barely moves, ` +
    `ending ${fmt(GAP.real_final_tee_dist)} m away, and real task success is ${pct(GAP.real_success_rate)}. At ` +
    `iteration ${sel} the imagined return is ${fmt(selImag)} while the real floor stays ${fmt(CURVE.real_return)}, ` +
    `a delusion of ${sgn(selDelusion)}. Imagination is only as good as your world model, and this one got the block ` +
    `dynamics wrong.`;

  return (
    <div class="ig">
      {/* ---- PANEL 1 (headline): the imagination gap + the reality check ---- */}
      <figure class="ig-headline">
        <figcaption class="ig-h">
          <span>the same policy, two worlds</span>
          <b>seed 0 · cpu · {CFG.eval_episodes} held-out starts</b>
        </figcaption>
        <div class="ig-headline-grid">
          <GapBars />
          <div class="ig-reality">
            <div class="ig-real-card ig-real-card--fail">
              <span class="ig-real-k">real task success</span>
              <span class="ig-real-v">{pct(GAP.real_success_rate)}</span>
              <span class="ig-real-note">the imagination-trained policy <b>never</b> solves the real task</span>
            </div>
            <div class="ig-teerow">
              <div class="ig-tee ig-tee--imag">
                <span class="ig-tee-k">dream tee-dist</span>
                <span class="ig-tee-v">{fmt(GAP.imagined_final_tee_dist)} m</span>
                <span class="ig-tee-note">the dreamed block <b>parks</b> at the target</span>
              </div>
              <div class="ig-tee ig-tee--real">
                <span class="ig-tee-k">real tee-dist</span>
                <span class="ig-tee-v">{fmt(GAP.real_final_tee_dist)} m</span>
                <span class="ig-tee-note">the real block <b>barely moved</b> from spawn</span>
              </div>
            </div>
          </div>
        </div>
        <p class="ig-headline-note">
          The policy looks like a champion <b class="ig-imag">in imagination</b> and <b>fails</b>{" "}
          <b class="ig-real">in reality</b>. It optimized a reward read off the block pose the world model{" "}
          <b>hallucinated</b> — and 3.1 measured that this model learned the pusher kinematics, <b>not</b> the
          block dynamics. This is <b>not</b> Dreamer solving PushT: it emphatically does not, at free-tier scale.
        </p>
      </figure>

      {/* ---- PANEL 2: proof it trained — the curve climbs AWAY from reality ---- */}
      <figure class="ig-panel">
        <figcaption class="ig-cap">
          <span>imagined return climbs · reality stays flat</span>
          <b>the policy DID train</b>
        </figcaption>
        <TrainCurve sel={sel} />

        <div class="ig-slider-row">
          <label class="ig-slider-label" for="ig-iter">iteration <b>t</b></label>
          <input
            id="ig-iter"
            class="ig-slider"
            type="range"
            min={0}
            max={IT_MAX}
            step={1}
            value={sel}
            onInput={(e) => setSel(parseInt((e.currentTarget as HTMLInputElement).value, 10))}
            aria-label={`Policy-gradient iteration, from 0 to ${IT_MAX}. Drag right and the imagined return climbs while the real floor stays flat — the delusion widens.`}
            aria-valuetext={`iteration ${sel}: imagined ${fmt(selImag)}, real floor ${fmt(CURVE.real_return)}, delusion ${sgn(selDelusion)}`}
          />
        </div>
        <div class="ig-readout" aria-hidden="true">
          <span class="ig-ro-k">iter {sel}</span>
          <span class="ig-ro-v ig-v--imag">imagined {fmt(selImag)}</span>
          <span class="ig-ro-v ig-v--real">real floor {fmt(CURVE.real_return)}</span>
          <span class="ig-ro-delusion">delusion {sgn(selDelusion)}</span>
        </div>
      </figure>

      {/* ---- PANEL 3: step 1 — the world model the actor dreams inside ---- */}
      <figure class="ig-panel ig-panel--wm">
        <figcaption class="ig-cap">
          <span>step 1 — the world model (from 3.1)</span>
          <b>reconstruction falls · dynamics low</b>
        </figcaption>
        <WmLosses />
        <p class="ig-wm-note">
          Unchanged from 3.1: reconstruction (the <b>easy half</b>) falls fast; the dynamics loss stays low. The
          actor then <b>freezes</b> this model and dreams inside it — inheriting the very block dynamics it got wrong.
        </p>
      </figure>

      {/* non-visual path to the same aha — the gap + the current-iteration delusion */}
      <div class="bk-sr" aria-live="polite">{announce}</div>

      <p class="ig-note">
        The honest lesson: <b>imagination is only as good as your world model.</b> The policy learned to move the
        pusher and to raise its <b>imagined</b> return — but the world it trained in is wrong on the block dims the
        reward depends on, so its "success" never transfers (<b>{pct(GAP.real_success_rate)}</b> real success). Across
        seeds 0–2 the gap holds: imagined {BAND.imagined_return[0]}…{BAND.imagined_return[1]}, real{" "}
        {BAND.real_return[0]}…{BAND.real_return[1]}, real success {pct(BAND.real_success_rate[0])} on{" "}
        <b>every</b> seed. Real numbers from dreamer.py (seed 0, cpu), matching meta.yaml; the whole toy reads with
        JS off, only the iteration scrubber needs it.
      </p>
    </div>
  );
}
