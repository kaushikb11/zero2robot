/**
 * FastCodecToy — ch5.5 "FAST: Turning Torques into Tokens" concept-toy
 * (demo id `fast_codec`). THE codec visualizer, made honest as a pure DATA-VIEWER
 * over a precomputed grid — no MuJoCo-WASM, no ONNX, no dynamic import. This chapter
 * trains NO policy, exports NO model, and runs NO sim: the artifact is a pure-numpy
 * CODEC, so there is nothing to *run*. It just reads demo/vizdata.json and draws the
 * DCT -> quantize -> BPE tradeoff in one figure.
 *
 * THE HERO. One smooth synthetic action chunk (H=24 x 3). LEFT: the original
 * trajectory with the current reconstruction overlaid, one lane per action dim —
 * watch the blue reconstruction snap onto the grey original as you keep more
 * coefficients. MIDDLE: the DCT spectrum (per-frequency energy) with a vertical
 * cutoff line at the "# coefficients kept" slider; the kept low-frequency bars hold
 * almost all the energy. RIGHT: a live token count (DCT+quant+BPE) and reconstruction
 * RMSE, next to the fixed naive per-step-per-dim baseline (72 tokens). Two sliders
 * drive it — keep_coeffs (over vizdata.keep_grid) and q_scale (over vizdata.scale_grid);
 * each pair indexes ONE precomputed entry in vizdata.settings, so the JS does NO
 * compute — it looks up {tokens, rmse, recon} and redraws. The line to feel: "keep a
 * handful of low-frequency coefficients and the whole smooth motion comes back — from
 * a FRACTION of the tokens per-step binning would spend."
 *
 * All numbers are REAL: read verbatim from fast.py's committed seed-0 vizdata.json
 * (a smooth synthetic chunk, where truncation is cheapest and the story is cleanest).
 * Nothing is mocked. HONEST CAVEAT (surfaced in the note): this dramatic compression
 * is the SMOOTH-chunk best case; real, phase-switching robot chunks compress ~2.2x
 * (per meta.yaml's measured reference run).
 *
 * Pure inline SVG + design tokens: theme-aware for free (light AND dark), and the
 * server-rendered default (both plots + the token/RMSE readout at keep=6, q=0.05) IS
 * the JS-off experience. Hydration only makes the two sliders draggable.
 *
 * Follows the FROZEN CONCEPT-TOY CONTRACT documented at the top of ../PlateIsland.tsx.
 */
import "./FastCodecToy.css";
import { useMemo, useState } from "preact/hooks";
// Real precomputed trajectory + DCT spectrum + {keep x scale} -> {tokens, rmse, recon}
// grid from fast.py's reference run (seed 0, default config) — committed small text
// (numeric grids), no binary. Same co-located-vizdata pattern the other viewer toys use.
import viz from "../../../../curriculum/phase5_practitioner/ch5.5_fast/demo/vizdata.json";

// ---------------------------------------------------------------- typed vizdata
const HORIZON: number = viz.horizon;        // 24 — timesteps per chunk
const ACT_DIM: number = viz.act_dim;        // 3  — action dims (kept small for legibility)
const NAIVE: number = viz.naive_tokens;     // 72 — per-step-per-dim baseline (H*act_dim)
const Q_MAX: number = viz.q_max;            // 127 — integer clip range for the quantizer
const TRAJ: number[][] = viz.trajectory as number[][];   // H x 3 — the original chunk
const DCT: number[][] = viz.dct_coeffs as number[][];    // H x 3 — spectrum (row 0 = DC)
const KEEP_GRID: number[] = viz.keep_grid as number[];   // [3, 6, 12, 24]
const SCALE_GRID: number[] = viz.scale_grid as number[]; // [0.025, 0.05, 0.1]
type Setting = { keep_coeffs: number; q_scale: number; tokens: number; rmse: number; recon: number[][] };
const SETTINGS: Setting[] = viz.settings as Setting[];

/** the precomputed entry for a (keep_coeffs, q_scale) pair — JS never computes, it
 *  just looks up {tokens, rmse, recon}. Matched by value so it is order-independent. */
function lookup(keep: number, scale: number): Setting {
  return (
    SETTINGS.find((s) => s.keep_coeffs === keep && s.q_scale === scale) ?? SETTINGS[0]
  );
}

// per-frequency spectral energy: the L2 magnitude across the 3 dims, so the 24 bars
// show where the chunk's energy lives (it piles into the first few low frequencies).
const FREQ_MAG: number[] = DCT.map((row) => Math.hypot(...row));
const MAX_MAG: number = Math.max(...FREQ_MAG);

// stable value-domain across every recon, so the trajectory axis never jumps when the
// slider changes (a lossy recon that overshoots still fits the frame).
const [VMIN, VMAX] = (() => {
  let lo = Infinity, hi = -Infinity;
  const scan = (grid: number[][]) => {
    for (const row of grid) for (const v of row) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  };
  scan(TRAJ);
  for (const s of SETTINGS) scan(s.recon);
  const pad = 0.06 * (hi - lo);
  return [lo - pad, hi + pad];
})();

// ------------------------------------------------------------ number formatting
const ratio = (v: number) => `${v.toFixed(1)}×`;
const rmseFmt = (v: number) => v.toFixed(3);

// ============================================================ THE TRAJECTORY PLOT
// Three stacked lanes (one per action dim) sharing the x-axis. Each lane draws the
// ORIGINAL as a grey ghost and the RECONSTRUCTION overlaid in signal blue — when the
// codec keeps enough coefficients the blue lands on the grey; when it is starved the
// blue smooths and lags. SSR renders this verbatim (it is the JS-off view).
const TW = 300, TH = 216;
const TPAD = { l: 10, r: 10, t: 8, b: 8 };
const LANE_GAP = 9;
const LANE_H = (TH - TPAD.t - TPAD.b - LANE_GAP * (ACT_DIM - 1)) / ACT_DIM;
const tx = (t: number) => TPAD.l + (t / (HORIZON - 1)) * (TW - TPAD.l - TPAD.r);
const laneTop = (d: number) => TPAD.t + d * (LANE_H + LANE_GAP);
const ty = (d: number, v: number) =>
  laneTop(d) + (1 - (v - VMIN) / (VMAX - VMIN)) * LANE_H;
const poly = (d: number, grid: number[][]) =>
  grid.map((row, t) => `${tx(t).toFixed(1)},${ty(d, row[d]).toFixed(1)}`).join(" ");

function Trajectory({ recon, keep, rmse }: { recon: number[][]; keep: number; rmse: number }) {
  const lanes = Array.from({ length: ACT_DIM }, (_, d) => d);
  return (
    <svg
      class="fc-traj"
      viewBox={`0 0 ${TW} ${TH}`}
      role="img"
      aria-label={
        `Action trajectory over ${HORIZON} timesteps, one lane per each of the ${ACT_DIM} action dimensions. ` +
        `In each lane the grey line is the original smooth chunk and the blue line is the reconstruction from ` +
        `${keep} kept DCT coefficients. The reconstruction ${rmse < 0.05 ? "lands almost exactly on" : rmse < 0.15 ? "tracks" : "smooths across and lags"} ` +
        `the original, at reconstruction RMSE ${rmseFmt(rmse)}.`
      }
    >
      <title>Reconstruction overlaid on the original chunk</title>
      {lanes.map((d) => (
        <g class="fc-lane">
          {/* lane baseline + frame */}
          <rect class="fc-lane-bg" x={TPAD.l} y={laneTop(d)} width={TW - TPAD.l - TPAD.r} height={LANE_H} rx={2} />
          <line class="fc-lane-zero" x1={TPAD.l} y1={ty(d, 0)} x2={TW - TPAD.r} y2={ty(d, 0)} />
          {/* the original — a grey ghost */}
          <polyline class="fc-orig" points={poly(d, TRAJ)} />
          {/* the reconstruction — signal blue, the thing the sliders move */}
          <polyline class="fc-recon" points={poly(d, recon)} />
          <text class="fc-lane-lbl" x={TPAD.l + 3} y={laneTop(d) + 10}>dim {d}</text>
        </g>
      ))}
    </svg>
  );
}

// =============================================================== THE DCT SPECTRUM
// 24 bars = per-frequency spectral energy (row 0 = DC, low freq on the left). A
// vertical cutoff at `keep` splits KEPT low-frequency bars (signal blue, they carry
// almost all the energy) from DISCARDED high-frequency bars (faded — they round to 0
// and BPE merges the zero-runs away). This is why a handful of coefficients suffice.
const SW = 300, SH = 216;
const SPAD = { l: 10, r: 10, t: 14, b: 22 };
const SPW = SW - SPAD.l - SPAD.r;
const BW = SPW / HORIZON;
const SY0 = SH - SPAD.b;
const barH = (mag: number) => (mag / MAX_MAG) * (SH - SPAD.t - SPAD.b);
const bx = (i: number) => SPAD.l + i * BW;

function Spectrum({ keep }: { keep: number }) {
  const cut = bx(keep); // the boundary: bars left of it are kept
  return (
    <svg
      class="fc-spec"
      viewBox={`0 0 ${SW} ${SH}`}
      role="img"
      aria-label={
        `DCT spectrum: per-frequency energy of the chunk across ${HORIZON} frequencies, lowest on the left ` +
        `(the DC term). Almost all the energy sits in the first few low-frequency coefficients. A cutoff line ` +
        `marks the ${keep} kept coefficients; the rest are discarded, round to zero, and are merged away by BPE.`
      }
    >
      <title>DCT spectrum with the coefficient cutoff</title>
      {/* baseline */}
      <line class="fc-spec-axis" x1={SPAD.l} y1={SY0} x2={SW - SPAD.r} y2={SY0} />
      {FREQ_MAG.map((mag, i) => {
        const h = barH(mag);
        const kept = i < keep;
        return (
          <rect
            class={`fc-bar ${kept ? "fc-bar--keep" : "fc-bar--drop"}`}
            x={bx(i) + 0.6}
            y={SY0 - h}
            width={Math.max(0.8, BW - 1.2)}
            height={h}
            rx={0.6}
          />
        );
      })}
      {/* the cutoff line — kept | discarded */}
      <line class="fc-cut" x1={cut} y1={SPAD.t - 2} x2={cut} y2={SY0} />
      <text class="fc-cut-lbl" x={cut + 3} y={SPAD.t + 6}>keep {keep}</text>
      {/* axis captions */}
      <text class="fc-spec-cap" x={SPAD.l} y={SH - 6}>← low freq</text>
      <text class="fc-spec-cap" x={SW - SPAD.r} y={SH - 6} text-anchor="end">high freq →</text>
    </svg>
  );
}

// ==================================================================== THE ISLAND
export default function FastCodecToy() {
  // default-interesting: keep 6 of 24 coefficients at q_scale 0.05 — a "handful" of
  // low frequencies that already reconstructs the smooth motion at tiny RMSE, from a
  // small fraction of the naive tokens. SSR renders at this setting, so it IS the
  // JS-off view; the sliders only add drag-to-change.
  const [keepIdx, setKeepIdx] = useState(1); // KEEP_GRID[1] = 6
  const [scaleIdx, setScaleIdx] = useState(1); // SCALE_GRID[1] = 0.05

  const keep = KEEP_GRID[keepIdx];
  const scale = SCALE_GRID[scaleIdx];
  const cur = lookup(keep, scale);
  const tokens = cur.tokens;
  const rmse = cur.rmse;
  const comp = NAIVE / tokens; // compression ratio vs the naive baseline

  const announce = useMemo(
    () =>
      `Keeping ${keep} of ${HORIZON} DCT coefficients at quantization step ${scale}: ` +
      `${tokens} tokens — ${ratio(comp)} fewer than the ${NAIVE}-token naive per-step-per-dim baseline — ` +
      `at reconstruction RMSE ${rmseFmt(rmse)}. ` +
      (rmse < 0.05
        ? "The reconstruction lands on the original."
        : "The reconstruction is visibly lossy: it smooths and lags the original."),
    [keep, scale, tokens, rmse, comp],
  );

  // token comparison mini-bars (fast vs naive), widths in percent of the naive bar
  const naivePct = 100;
  const fastPct = (tokens / NAIVE) * 100;

  return (
    <div class="fc">
      <header class="fc-head">
        <h3 class="fc-title">DCT → quantize → BPE, live</h3>
        <p class="fc-sub">
          One smooth action chunk. Keep a <b>handful of low-frequency DCT coefficients</b>, quantize, and let
          <b> BPE</b> merge the zero-runs: the whole motion comes back from a <b>fraction</b> of the tokens per-step
          binning would spend. Drag the two sliders — the reconstruction snaps onto the original as the token
          counter drops.
        </p>
      </header>

      <div class="fc-stage">
        {/* LEFT — the trajectory with the reconstruction overlaid */}
        <figure class="fc-cell">
          <figcaption class="fc-cell-cap">trajectory · reconstruction overlaid</figcaption>
          <Trajectory recon={cur.recon} keep={keep} rmse={rmse} />
          <div class="fc-legend" aria-hidden="true">
            <span class="fc-leg fc-leg--orig">original</span>
            <span class="fc-leg fc-leg--recon">reconstruction</span>
          </div>
        </figure>

        {/* MIDDLE — the DCT spectrum with the cutoff */}
        <figure class="fc-cell">
          <figcaption class="fc-cell-cap">DCT spectrum · energy per frequency</figcaption>
          <Spectrum keep={keep} />
          <div class="fc-legend" aria-hidden="true">
            <span class="fc-leg fc-leg--keep">kept</span>
            <span class="fc-leg fc-leg--drop">discarded → 0</span>
          </div>
        </figure>

        {/* RIGHT — the live token / RMSE readout vs the naive baseline */}
        <div class="fc-readout" aria-hidden="true">
          <div class="fc-tok">
            <span class="fc-tok-k">FAST tokens</span>
            <span class="fc-tok-v">{tokens}</span>
            <span class="fc-tok-sub">{ratio(comp)} fewer than naive</span>
          </div>

          <div class="fc-bars">
            <div class="fc-bars-row">
              <span class="fc-bars-k">naive</span>
              <span class="fc-bars-track"><span class="fc-bars-fill fc-bars-fill--naive" style={`width:${naivePct}%`} /></span>
              <span class="fc-bars-n">{NAIVE}</span>
            </div>
            <div class="fc-bars-row">
              <span class="fc-bars-k">FAST</span>
              <span class="fc-bars-track"><span class="fc-bars-fill fc-bars-fill--fast" style={`width:${fastPct.toFixed(1)}%`} /></span>
              <span class="fc-bars-n">{tokens}</span>
            </div>
          </div>

          <div class="fc-stat">
            <div class="fc-stat-row">
              <span class="fc-stat-k">reconstruction RMSE</span>
              <span class={`fc-stat-v ${rmse < 0.05 ? "fc-ok" : "fc-warn"}`}>{rmseFmt(rmse)}</span>
            </div>
            <div class="fc-stat-row">
              <span class="fc-stat-k">coefficients kept</span>
              <span class="fc-stat-v">{keep} / {HORIZON}</span>
            </div>
            <div class="fc-stat-row">
              <span class="fc-stat-k">quant step · clip</span>
              <span class="fc-stat-v">{scale} · ±{Q_MAX}</span>
            </div>
          </div>
        </div>
      </div>

      {/* --- the two controls: # coefficients kept + quantization step --- */}
      <div class="fc-controls">
        <div class="fc-slider-group">
          <label class="fc-slider-lbl" for="fc-keep"># coefficients kept</label>
          <input
            id="fc-keep"
            class="fc-slider"
            type="range"
            min={0}
            max={KEEP_GRID.length - 1}
            step={1}
            value={keepIdx}
            onInput={(e) => setKeepIdx(Number((e.target as HTMLInputElement).value))}
            aria-valuetext={`${keep} of ${HORIZON} coefficients kept — ${tokens} tokens, RMSE ${rmseFmt(rmse)}`}
          />
          <output class="fc-slider-out" for="fc-keep">{keep}</output>
        </div>

        <div class="fc-slider-group">
          <label class="fc-slider-lbl" for="fc-scale">quantization step</label>
          <input
            id="fc-scale"
            class="fc-slider"
            type="range"
            min={0}
            max={SCALE_GRID.length - 1}
            step={1}
            value={scaleIdx}
            onInput={(e) => setScaleIdx(Number((e.target as HTMLInputElement).value))}
            aria-valuetext={`quantization step ${scale} — coarser steps spend fewer tokens; ${tokens} tokens, RMSE ${rmseFmt(rmse)}`}
          />
          <output class="fc-slider-out" for="fc-scale">{scale}</output>
        </div>

        <span class="fc-control-note">drag either slider · coarser / fewer coeffs = fewer tokens · poster reads with JS off</span>
      </div>

      {/* non-visual path to the same aha — the qualitative story, not per-frame spam */}
      <div class="bk-sr" aria-live="polite">{announce}</div>

      {/* the honest framing — this is the SMOOTH-chunk best case */}
      <p class="fc-note">
        The DCT is <b>orthonormal</b>, so quantizing coefficients costs the same error <b>energy</b> as quantizing
        raw samples (Parseval) — but in the frequency domain that error <b>concentrates</b>: most coefficients round
        to exactly <b>0</b>, and BPE losslessly merges the zero-runs. On this <b>smooth synthetic</b> chunk the win
        looks enormous because the energy piles into a few low frequencies. On <b>real, phase-switching robot chunks</b>
        the motion has more high-frequency content, so FAST compresses a more honest <b>~2.2×</b> over per-step binning
        at comparable RMSE (measured across seeds 0–2 in meta.yaml). Real precomputed grid from fast.py
        (seed 0, {HORIZON}×{ACT_DIM} chunk, naive baseline {NAIVE} tokens); poster reads with JS off.
      </p>
    </div>
  );
}
