/**
 * LoraRankDialToy — ch5.6 "LoRA From Scratch" concept-toy (demo id `lora_rank_dial`).
 * THE RANK DIAL, made honest as a pure DATA-VIEWER over a precomputed rank sweep —
 * no MuJoCo-WASM, no ONNX, no dynamic import. This chapter trains NO env policy and
 * exports NO model; the artifact emits a 7-point sweep of scalar metrics, so this is
 * the cheapest embed tier: it just reads demo/vizdata.json and draws two synced curves.
 *
 * THE HERO — THE ELBOW. A single slider turns the LoRA rank r over the swept values
 * (0, 1, 2, 4, 8, 16). Two readouts move in sync: (1) "% of base weights trained"
 * CLIMBS left→right (0% → 5.4%), while (2) "held-out skill fit (R²)" RISES then
 * PLATEAUS — it flattens onto the dashed full-fine-tune ceiling long before the
 * params do. At r=4 you train ~1.3% of the weights and already recover ~84% of
 * full-FT's held-out gain; past that knee, more trainable params buy almost nothing.
 * r=0 (frozen zero-shot) anchors the floor, full fine-tune (100% of params) anchors
 * the ceiling. The one line to feel: "fewer params" is not "worse fit."
 *
 * THE HONEST TWIST. A muted third trace shows the in-distribution task_A fit FALLING
 * as the dial turns — the skill the frozen base already knew is being forgotten as the
 * adapter grows, even though the base W is frozen. LoRA buys parameter efficiency; it
 * does NOT prevent forgetting (the chapter measured that it doesn't).
 *
 * All numbers are REAL: read verbatim from lora.py's committed seed-0 vizdata.json
 * (per-rank trainable_pct / held_out_r2 / task_a_r2, plus the full-FT and frozen
 * anchors). Nothing is mocked or interpolated between rank stops.
 *
 * Pure inline SVG + design tokens: theme-aware for free (light AND dark), and the
 * server-rendered default (the full sweep curve + the ceiling/floor references + both
 * readouts at the highlighted rank) IS the JS-off experience. Hydration only adds the
 * rank slider, the forgetting-trace toggle, and the aria-live readout.
 *
 * Follows the FROZEN CONCEPT-TOY CONTRACT documented at the top of ../PlateIsland.tsx.
 */
import "./LoraRankDialToy.css";
import { useMemo, useState } from "preact/hooks";
// Real per-rank sweep + anchors from lora.py's reference run (seed 0, default config)
// — committed small text (a 7-point scalar sweep), no binary. Same co-located-vizdata
// pattern the other data-viewer toys use.
import viz from "../../../../curriculum/phase5_practitioner/ch5.6_lora/demo/vizdata.json";

// ---------------------------------------------------------------- typed vizdata
const RANKS: number[] = viz.ranks as number[];            // [0, 1, 2, 4, 8, 16]
const TRAIN_PCT: number[] = viz.trainable_pct as number[]; // % of base params trained
const HELD_R2: number[] = viz.held_out_r2 as number[];     // held-out skill fit (rises → plateaus)
const TASK_A_R2: number[] = viz.task_a_r2 as number[];     // in-distribution fit (falls: forgetting)
const BASE_PARAMS: number = viz.base_total_params;         // 77,958 frozen base params
const HEADLINE_RANK: number = viz.headline_rank;           // 4 — the elbow the toy highlights
const FULL = viz.full as { held_out_r2: number; task_a_r2: number; trainable_pct: number };
const FROZEN = viz.frozen as { held_out_r2: number; task_a_r2: number };

const N = RANKS.length;
const HEADLINE_IDX = Math.max(0, RANKS.indexOf(HEADLINE_RANK));

// full-FT's held-out gain over the frozen floor — the denominator for "% of full's
// gain recovered". Frozen sits below zero (a wrong extrapolation), full sits at ~1.0.
const FULL_GAIN = FULL.held_out_r2 - FROZEN.held_out_r2;

/** what fraction of full-FT's held-out gain rank i has recovered (0 at frozen, ~1 at full). */
function gainFrac(i: number): number {
  return FULL_GAIN > 0 ? (HELD_R2[i] - FROZEN.held_out_r2) / FULL_GAIN : 0;
}

// ------------------------------------------------------------ number formatting
const pct2 = (v: number) => `${v.toFixed(2)}%`;
const pct0 = (v: number) => `${Math.round(v * 100)}%`;
const r2 = (v: number) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
const paramsOf = (i: number) => Math.round((TRAIN_PCT[i] / 100) * BASE_PARAMS);
const commas = (n: number) => n.toLocaleString("en-US");

// ==================================================================== THE CHART
// held-out R² (blue, the hero — rises then plateaus) and task_A R² (red, muted — the
// forgetting trace, falls) vs rank, on evenly-spaced rank stops. The dashed green line
// is the full-FT ceiling; the dashed neutral line is the frozen floor. The elbow is the
// knee in the blue curve. SSR renders this verbatim — it is the JS-off view.
const CW = 560;
const CH = 288;
const PAD = { l: 52, r: 118, t: 26, b: 46 };
const PW = CW - PAD.l - PAD.r;
const PH = CH - PAD.t - PAD.b;
const Y_MIN = -2.55;
const Y_MAX = 1.12;
const cx = (i: number) => PAD.l + (i / (N - 1)) * PW;
const cy = (v: number) =>
  PAD.t + ((Y_MAX - Math.max(Y_MIN, Math.min(Y_MAX, v))) / (Y_MAX - Y_MIN)) * PH;

function Chart({ sel, showForget }: { sel: number; showForget: boolean }) {
  const heldPts = HELD_R2.map((v, i) => `${cx(i).toFixed(1)},${cy(v).toFixed(1)}`).join(" ");
  const taskPts = TASK_A_R2.map((v, i) => `${cx(i).toFixed(1)},${cy(v).toFixed(1)}`).join(" ");
  const y0 = cy(0);
  const ceilY = cy(FULL.held_out_r2);
  const floorY = cy(FROZEN.held_out_r2);
  const yTicks = [1, 0, -1, -2];
  const gPct = gainFrac(HEADLINE_IDX);

  return (
    <svg
      class="lr-chart"
      viewBox={`0 0 ${CW} ${CH}`}
      role="img"
      aria-label={
        `Rank sweep. Two curves versus LoRA rank r on stops ${RANKS.join(", ")}. The blue held-out skill ` +
        `fit, measured as R squared, starts at ${r2(HELD_R2[0])} for the frozen base at rank 0, rises steeply, ` +
        `and PLATEAUS: at rank ${HEADLINE_RANK} it is ${r2(HELD_R2[HEADLINE_IDX])}, about ${pct0(gPct)} of the way from ` +
        `the frozen floor to the full fine-tune ceiling at ${r2(FULL.held_out_r2)}, while training only ` +
        `${pct2(TRAIN_PCT[HEADLINE_IDX])} of the base weights. By rank 8 it has flattened onto the ceiling. That knee ` +
        `is the elbow: past it, more trainable parameters buy almost nothing. The muted red task_A trace, the ` +
        `in-distribution skill the frozen base already knew, FALLS from ${r2(TASK_A_R2[0])} toward ` +
        `${r2(TASK_A_R2[N - 1])} as the adapter grows — LoRA does not prevent forgetting.`
      }
    >
      {/* y grid + ticks */}
      {yTicks.map((v) => (
        <g>
          <line class="lr-grid" x1={PAD.l} y1={cy(v)} x2={CW - PAD.r} y2={cy(v)} />
          <text class="lr-tick" x={PAD.l - 7} y={cy(v) + 3} text-anchor="end">{v.toFixed(1)}</text>
        </g>
      ))}

      {/* full-FT ceiling (100% of params) — the goal the held-out fit plateaus onto */}
      <line class="lr-ceil" x1={PAD.l} y1={ceilY} x2={CW - PAD.r} y2={ceilY} />
      <text class="lr-ceil-lbl" x={CW - PAD.r + 4} y={ceilY + 3}>full FT ceiling</text>
      <text class="lr-ceil-sub" x={CW - PAD.r + 4} y={ceilY + 15}>100% of params</text>

      {/* frozen floor (r=0, zero-shot) — where a wholly frozen base sits */}
      <line class="lr-floor" x1={PAD.l} y1={floorY} x2={CW - PAD.r} y2={floorY} />
      <text class="lr-floor-lbl" x={CW - PAD.r + 4} y={floorY + 3}>frozen floor</text>
      <text class="lr-floor-sub" x={CW - PAD.r + 4} y={floorY + 15}>0% of params</text>

      {/* the selected-rank cursor (hydration-updated; SSR draws it at the headline rank) */}
      <line class="lr-cursor" x1={cx(sel)} y1={PAD.t} x2={cx(sel)} y2={y0} />

      {/* the elbow annotation at the headline rank — the SSR-visible hero */}
      <g class="lr-elbow">
        <circle cx={cx(HEADLINE_IDX)} cy={cy(HELD_R2[HEADLINE_IDX])} r={7} />
        <text class="lr-elbow-lbl" x={cx(HEADLINE_IDX)} y={cy(HELD_R2[HEADLINE_IDX]) - 12} text-anchor="middle">
          the elbow
        </text>
        <text class="lr-elbow-sub" x={cx(HEADLINE_IDX)} y={cy(HELD_R2[HEADLINE_IDX]) - 1} text-anchor="middle">
          r{HEADLINE_RANK} · {pct2(TRAIN_PCT[HEADLINE_IDX])} params · {pct0(gPct)}
        </text>
      </g>

      {/* x axis (zero line) + rank ticks */}
      <line class="lr-axis" x1={PAD.l} y1={y0} x2={CW - PAD.r} y2={y0} />
      {RANKS.map((r, i) => (
        <text class="lr-tick" x={cx(i)} y={CH - 26} text-anchor="middle">{r}</text>
      ))}
      <text class="lr-axis-cap" x={(PAD.l + CW - PAD.r) / 2} y={CH - 8} text-anchor="middle">
        LoRA rank r  ·  more trainable params →
      </text>
      <text class="lr-axis-cap lr-yaxis-cap" x={PAD.l - 40} y={PAD.t - 10}>held-out fit (R²)</text>

      {/* the forgetting trace — task_A fit falling as the adapter grows (muted) */}
      {showForget && <polyline class="lr-curve lr-curve--task" points={taskPts} />}
      {showForget &&
        TASK_A_R2.map((v, i) => (
          <circle class="lr-dot lr-dot--task" cx={cx(i)} cy={cy(v)} r={i === sel ? 4 : 2.6} />
        ))}

      {/* the hero — held-out fit rising then plateauing */}
      <polyline class="lr-curve lr-curve--held" points={heldPts} />
      {HELD_R2.map((v, i) => (
        <circle
          class="lr-dot lr-dot--held"
          data-sel={i === sel}
          cx={cx(i)}
          cy={cy(v)}
          r={i === sel ? 5 : 3}
        />
      ))}
    </svg>
  );
}

// ==================================================================== THE ISLAND
export default function LoraRankDialToy() {
  // default-interesting: the elbow (r=4). SSR renders here, so the JS-off view already
  // shows both readouts sitting at the knee — ~1.3% of the weights, ~84% of full's fit.
  const [sel, setSel] = useState(HEADLINE_IDX);
  const [showForget, setShowForget] = useState(true);

  const r = RANKS[sel];
  const tp = TRAIN_PCT[sel];
  const held = HELD_R2[sel];
  const taskA = TASK_A_R2[sel];
  const g = gainFrac(sel);
  const nParams = paramsOf(sel);

  const frozen = sel === 0;
  const plateaued = g >= 0.98;

  const announce = useMemo(() => {
    const climb =
      frozen
        ? `The base is fully frozen: 0% of the weights trained.`
        : `Training ${pct2(tp)} of the base weights (${commas(nParams)} of ${commas(BASE_PARAMS)}).`;
    const fit = frozen
      ? `Held-out fit is ${r2(held)} — the frozen base extrapolates the held-out skill wrongly.`
      : plateaued
        ? `Held-out fit is ${r2(held)}, ${pct0(g)} of full fine-tune's gain — it has plateaued onto the ceiling.`
        : `Held-out fit is ${r2(held)}, ${pct0(g)} of full fine-tune's gain.`;
    const forget = `The in-distribution task_A fit is ${r2(taskA)}` +
      (frozen ? ` — still intact, since nothing has trained yet.` : `, down from ${r2(TASK_A_R2[0])} frozen: the old skill is being forgotten as the adapter grows.`);
    return `Rank ${r}. ${climb} ${fit} ${forget}`;
  }, [sel]);

  return (
    <div class="lr">
      <header class="lr-head">
        <h3 class="lr-title">Turn the rank dial</h3>
        <p class="lr-sub">
          One knob: the <b>LoRA rank r</b>. As you turn it up, the <b>% of base weights trained</b> climbs — but the{" "}
          <b>held-out skill fit</b> rises then <b>plateaus</b>, flattening onto the full-fine-tune ceiling long before
          the params do. At <b>r{HEADLINE_RANK}</b> you train <b>~{pct2(TRAIN_PCT[HEADLINE_IDX])}</b> of the weights and
          recover <b>~{pct0(gainFrac(HEADLINE_IDX))}</b> of full-FT's gain. <b>Fewer params</b> is not <b>worse fit</b>.
        </p>
      </header>

      <Chart sel={sel} showForget={showForget} />

      <div class="lr-legend">
        <span class="lr-leg lr-leg--held"><b>held-out fit</b>&nbsp;the new skill (rises → plateaus)</span>
        <span class="lr-leg lr-leg--task"><b>task_A fit</b>&nbsp;the old skill (falls: forgetting)</span>
        <span class="lr-leg lr-leg--ceil"><b>full FT</b>&nbsp;100% of params (ceiling)</span>
      </div>

      {/* --- the one control: the rank dial (+ a toggle for the forgetting trace) --- */}
      <div class="lr-controls">
        <label class="lr-slider-lbl" for="lr-rank">rank r</label>
        <input
          id="lr-rank"
          class="lr-slider"
          type="range"
          min={0}
          max={N - 1}
          step={1}
          value={sel}
          list="lr-ticks"
          onInput={(e) => setSel(Number((e.target as HTMLInputElement).value))}
          aria-valuetext={
            `rank ${r} — ${frozen ? "0% weights (frozen)" : `${pct2(tp)} of weights`}, ` +
            `held-out fit ${r2(held)} (${pct0(g)} of full-FT's gain)`
          }
        />
        <datalist id="lr-ticks">
          {RANKS.map((rk, i) => <option value={i} label={String(rk)} />)}
        </datalist>
        <output class="lr-rank-out" for="lr-rank">
          r = <b>{r}</b>{r === HEADLINE_RANK ? " · the elbow" : frozen ? " · frozen" : ""}
        </output>
      </div>

      {/* the two synced readouts + the honest third (forgetting) */}
      <div class="lr-readouts" aria-hidden="true">
        <div class="lr-card lr-card--params">
          <span class="lr-card-k">% of base weights trained</span>
          <span class="lr-card-v">{pct2(tp)}</span>
          <div class="lr-meter" role="presentation">
            <div class="lr-meter-fill" style={`width:${Math.min(100, tp).toFixed(2)}%`} />
          </div>
          <span class="lr-card-sub">
            {frozen ? "nothing trained — the base is frozen" : `${commas(nParams)} of ${commas(BASE_PARAMS)} params · full FT trains 100%`}
          </span>
        </div>

        <div class="lr-card lr-card--fit" data-plateau={plateaued}>
          <span class="lr-card-k">held-out skill fit (R²)</span>
          <span class="lr-card-v">{r2(held)}</span>
          <div class="lr-meter lr-meter--fit" role="presentation">
            <div class="lr-meter-fill" style={`width:${Math.max(0, Math.min(100, g * 100)).toFixed(1)}%`} />
          </div>
          <span class="lr-card-sub">
            {frozen
              ? "frozen base — wrong on the held-out skill"
              : `${pct0(g)} of full-FT's gain${plateaued ? " · on the ceiling" : ""}`}
          </span>
        </div>

        <div class="lr-card lr-card--forget">
          <span class="lr-card-k">task_A fit (the old skill)</span>
          <span class="lr-card-v">{r2(taskA)}</span>
          <span class="lr-card-sub">
            {frozen ? "intact — nothing trained yet" : `was ${r2(TASK_A_R2[0])} frozen · forgetting as r grows`}
          </span>
        </div>
      </div>

      {/* forgetting-trace toggle — progressive enhancement; default on so SSR shows it */}
      <div class="lr-toggle-row">
        <button
          type="button"
          class="lr-toggle"
          data-on={showForget}
          aria-pressed={showForget}
          onClick={() => setShowForget((s) => !s)}
        >
          {showForget ? "hide" : "show"} the forgetting trace
        </button>
        <span class="lr-control-note">drag the dial · slider stops at each swept rank · poster reads with JS off</span>
      </div>

      {/* non-visual path to the same aha — the qualitative story, not per-step spam */}
      <div class="lr-sr" aria-live="polite">{announce}</div>

      {/* the honest framing — the load-bearing caveat */}
      <p class="lr-note">
        This teaches <b>parameter efficiency</b>, not free lunch. The elbow is real — <b>~{pct2(TRAIN_PCT[HEADLINE_IDX])}</b> of
        the weights recovers <b>~{pct0(gainFrac(HEADLINE_IDX))}</b> of full-FT's held-out gain, and the fit saturates
        onto the full-FT ceiling by <b>r8</b> while the trainable params keep climbing to <b>{pct2(TRAIN_PCT[N - 1])}</b>.
        But LoRA does <b>not</b> prevent forgetting: with the base W frozen, the <b>task_A</b> skill still degrades from{" "}
        <b>{r2(TASK_A_R2[0])}</b> to <b>{r2(TASK_A_R2[N - 1])}</b> as the adapter grows — the new skill snaps in for ~1%
        of the weights, and the old skill quietly leaves. Real per-rank numbers from lora.py (seed 0, default config,
        {" "}{commas(BASE_PARAMS)} base params); poster reads with JS off.
      </p>
    </div>
  );
}
