/**
 * QuantizeDialToy — ch5.7 "Quantize a Policy by Hand" concept-toy
 * (demo id `quantization_dial`). A pure DATA-VIEWER over precomputed measurements —
 * no MuJoCo-WASM, no ONNX, no dynamic import. This chapter exports NO model and runs
 * NO policy in the browser; the artifact emits a static measurement (three deployment
 * configs + the weight round-trip errors), so this is the cheapest embed tier: it just
 * reads demo/vizdata.json and draws three bars.
 *
 * THE HERO — the quantization dial. A 3-detent slider (FP32 → per-tensor INT8 →
 * per-channel INT8) drives three live bars:
 *   · SIZE          drops ~3.6–3.8× and holds  (the guaranteed WIN)
 *   · ACTION ERROR  ~0 at FP32, SPIKES at per-tensor INT8, then RECOVERS ~half of it
 *                   at per-channel INT8 — the whole lesson in one motion
 *   · LATENCY       RISES ~6× — the HONEST surprise: naive INT8 is SLOWER on a laptop
 *                   CPU (no fused low-precision kernel). The size win is guaranteed;
 *                   the latency win is a Scale Lab, NOT a laptop result.
 * Every bar carries faint ghost ticks for all three configs, so the full arc (drop /
 * spike-then-recover / rise) is legible in a single static frame — which is exactly
 * what the SSR poster renders. Hydration only makes the dial move.
 *
 * All numbers are REAL, read verbatim from quantize.py's committed seed-0 vizdata.json
 * (three configs with size_kb / action_mse / latency_ms / success + Wilson CI, plus the
 * naive-round zero-fraction and per-tensor vs per-channel weight round-trip errors).
 * Nothing is mocked.
 *
 * Pure inline SVG + design tokens: theme-aware for free (light AND dark), and the
 * server-rendered default (the three bars + the selected readout + the ledger + the
 * scale note + the honest latency framing) IS the JS-off experience.
 *
 * Follows the FROZEN CONCEPT-TOY CONTRACT documented at the top of ../PlateIsland.tsx.
 */
import "./QuantizeDialToy.css";
import { useMemo, useState } from "preact/hooks";
// Real precomputed deployment configs + weight round-trip errors from quantize.py's
// reference run (seed 0, default config) — committed small text (a handful of scalars),
// no binary. Same co-located-vizdata pattern the other data-viewer toys use.
import viz from "../../../../curriculum/phase5_practitioner/ch5.7_quantize/demo/vizdata.json";

// ---------------------------------------------------------------- typed vizdata
type Config = {
  label: string;        // "FP32" | "per-tensor INT8" | "per-channel INT8"
  size_kb: number;      // weight footprint on disk / in memory
  action_mse: number;   // mean-squared action error vs the FP32 reference (0 at FP32)
  latency_ms: number;   // per-call inference latency on a laptop CPU
  success: number;      // successful rollouts …
  n: number;            // … out of n
  success_rate: number; // success / n
  ci_lo: number;        // Wilson interval on the success rate
  ci_hi: number;
};
const CONFIGS: Config[] = viz.configs as Config[];
const FP32 = CONFIGS[0];
// the "scale is the idea" companion figure — rounding-with-no-scale vs a per-row scale
const NAIVE_ZERO: number = viz.naive_round_zero_frac;                 // 0.9542
const RT_PER_TENSOR: number = viz.weight_roundtrip_err.per_tensor;    // 0.00193…
const RT_PER_CHANNEL: number = viz.weight_roundtrip_err.per_channel;  // 0.00112…

// short slider-detent labels (the full labels live in CONFIGS[i].label)
const DETENTS = ["FP32", "per-tensor", "per-channel"];

// ------------------------------------------------------------ number formatting
const kb = (v: number) => v.toFixed(1);
const ms = (v: number) => v.toFixed(3);
const x1 = (v: number) => v.toFixed(1);
const pct = (v: number) => Math.round(v * 100);
// action error is O(1e-4); show it scaled to ×10⁻⁴ so the mono readout stays legible.
const mse4 = (v: number) => (v * 1e4).toFixed(2);
const sizeRatio = (c: Config) => FP32.size_kb / c.size_kb;   // ×smaller (≥1)
const latRatio = (c: Config) => c.latency_ms / FP32.latency_ms; // ×slower (≥1 for INT8)

// ============================================================ THE THREE BARS
// One shared 200×26 viewBox per metric. The fill runs to the SELECTED config's
// value; faint ghost ticks mark ALL three configs so the arc (drop / spike-then-
// recover / rise) reads in a single static frame. SSR renders these verbatim — they
// ARE the JS-off view; the dial only moves which tick the fill reaches.
const BAR_W = 200;
const BAR_H = 26;
const PAD = 3;
const TRACK_Y = 10;
const TRACK_H = 9;
const TRACK_W = BAR_W - 2 * PAD;

type MetricKey = "size" | "mse" | "lat";

function MetricBar({
  cls,
  values,
  sel,
  ariaLabel,
}: {
  cls: MetricKey;
  values: number[]; // one per config
  sel: number;      // selected config index
  ariaLabel: string;
}) {
  const max = Math.max(...values, 1e-12);
  const xOf = (v: number) => PAD + (v / max) * TRACK_W;
  const selX = xOf(values[sel]);
  const fillW = Math.max(0, selX - PAD);
  return (
    <svg class={`qz-bar qz-bar--${cls}`} viewBox={`0 0 ${BAR_W} ${BAR_H}`} role="img" aria-label={ariaLabel}>
      {cls === "lat" && (
        <defs>
          {/* a diagonal hatch marks the latency fill as the CAVEAT bar (the honest
              surprise), distinct from the solid action-error cost bar */}
          <pattern id="qz-lat-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line class="qz-hatch-line" x1={0} y1={0} x2={0} y2={5} />
          </pattern>
        </defs>
      )}
      <rect class="qz-track" x={PAD} y={TRACK_Y} width={TRACK_W} height={TRACK_H} rx={2} />
      <rect class="qz-fill" x={PAD} y={TRACK_Y} width={fillW} height={TRACK_H} rx={2} />
      {cls === "lat" && fillW > 0 && (
        <rect x={PAD} y={TRACK_Y} width={fillW} height={TRACK_H} rx={2} fill="url(#qz-lat-hatch)" />
      )}
      {/* ghost ticks for every config — the full arc, always visible */}
      {values.map((v, i) => (
        <line
          class="qz-ghost"
          data-sel={i === sel}
          x1={xOf(v)}
          y1={TRACK_Y - 4}
          x2={xOf(v)}
          y2={TRACK_Y + TRACK_H + 4}
        />
      ))}
      {/* a caret the eye lands on, riding the selected config's level */}
      <path class="qz-caret" d={`M${(selX - 3).toFixed(1)} ${TRACK_Y - 6} L${(selX + 3).toFixed(1)} ${TRACK_Y - 6} L${selX.toFixed(1)} ${TRACK_Y - 1} z`} />
    </svg>
  );
}

// ==================================================================== THE ISLAND
export default function QuantizeDialToy() {
  // default-interesting: per-tensor INT8 (index 1) — the SPIKE. A problem on screen
  // invites the fix: slide right and the per-channel scale recovers half the error.
  // SSR renders at this detent, so the JS-off poster already shows the spike + the
  // full ghost-tick arc.
  const [idx, setIdx] = useState(1);
  const sel = CONFIGS[idx];

  const sizes = CONFIGS.map((c) => c.size_kb);
  const mses = CONFIGS.map((c) => c.action_mse);
  const lats = CONFIGS.map((c) => c.latency_ms);

  const sizeAria =
    `Size bar. FP32 ${kb(FP32.size_kb)} kilobytes, per-tensor INT8 ${kb(CONFIGS[1].size_kb)}, ` +
    `per-channel INT8 ${kb(CONFIGS[2].size_kb)}. Showing ${sel.label} at ${kb(sel.size_kb)} kilobytes — ` +
    `${idx === 0 ? "the baseline" : `${x1(sizeRatio(sel))} times smaller than FP32`}. The size win is guaranteed.`;
  const mseAria =
    `Action-error bar, mean-squared error versus FP32. FP32 is the reference at zero, ` +
    `per-tensor INT8 spikes to ${mse4(CONFIGS[1].action_mse)} times ten-to-the-minus-four, ` +
    `per-channel INT8 recovers to ${mse4(CONFIGS[2].action_mse)}. Showing ${sel.label} at ` +
    `${sel.action_mse === 0 ? "zero — the reference" : `${mse4(sel.action_mse)} times ten-to-the-minus-four`}.`;
  const latAria =
    `Latency bar, milliseconds per call on a laptop CPU. FP32 ${ms(FP32.latency_ms)}, ` +
    `per-tensor INT8 ${ms(CONFIGS[1].latency_ms)}, per-channel INT8 ${ms(CONFIGS[2].latency_ms)}. ` +
    `Showing ${sel.label} at ${ms(sel.latency_ms)} milliseconds — ` +
    `${idx === 0 ? "the baseline" : `${x1(latRatio(sel))} times SLOWER than FP32`}. ` +
    `Naive INT8 is slower on a laptop CPU: there is no fused low-precision kernel.`;

  const announce = useMemo(() => {
    const errStory =
      idx === 0
        ? "this is the FP32 reference — zero action error"
        : idx === 1
          ? `the per-tensor scale spikes the action error to ${mse4(sel.action_mse)} times ten-to-the-minus-four`
          : `the per-channel scale recovers the accuracy the per-tensor scale threw away — ${mse4(sel.action_mse)} times ten-to-the-minus-four, about half the per-tensor error`;
    return (
      `${sel.label}. Size ${kb(sel.size_kb)} kilobytes` +
      `${idx === 0 ? " (the baseline)" : `, ${x1(sizeRatio(sel))} times smaller than FP32 — the guaranteed win`}. ` +
      `Action error: ${errStory}. ` +
      `Latency ${ms(sel.latency_ms)} milliseconds` +
      `${idx === 0 ? " (the baseline)." : `, ${x1(latRatio(sel))} times slower than FP32 — naive INT8 is slower on a laptop CPU, no fused kernel; the speedup is a Scale Lab, not a laptop result.`}`
    );
  }, [idx]);

  const spike = idx === 1;
  const recovered = idx === 2;

  return (
    <div class="qz">
      <header class="qz-head">
        <h3 class="qz-title">Quantize a policy by hand</h3>
        <p class="qz-sub">
          Slide the dial <b>FP32 → per-tensor INT8 → per-channel INT8</b> and watch three real measurements move
          together. <b>Size</b> drops ~{x1(sizeRatio(CONFIGS[1]))}× and holds. <b>Action error</b> is zero at FP32,{" "}
          <b>spikes</b> at per-tensor INT8, then <b>recovers</b> most of the way at per-channel INT8 — the{" "}
          <b>granularity of the scale IS the idea</b>. <b>Latency</b> goes <b>up</b>, not down (the honest surprise).
        </p>
      </header>

      {/* --- THE THREE BARS ---------------------------------------------------- */}
      <div class="qz-bars">
        <div class="qz-metric qz-metric--size">
          <div class="qz-metric-head" aria-hidden="true">
            <span class="qz-metric-name">size</span>
            <span class="qz-metric-val">
              {kb(sel.size_kb)} <span class="qz-metric-u">KB</span>
              <span class="qz-metric-delta qz-delta--good">
                {idx === 0 ? "baseline" : `${x1(sizeRatio(sel))}× smaller ✓`}
              </span>
            </span>
          </div>
          <MetricBar cls="size" values={sizes} sel={idx} ariaLabel={sizeAria} />
        </div>

        <div class="qz-metric qz-metric--mse">
          <div class="qz-metric-head" aria-hidden="true">
            <span class="qz-metric-name">action error vs FP32</span>
            <span class="qz-metric-val">
              {sel.action_mse === 0 ? "0" : mse4(sel.action_mse)} <span class="qz-metric-u">×10⁻⁴ MSE</span>
              <span class={`qz-metric-delta ${spike ? "qz-delta--bad" : recovered ? "qz-delta--good" : "qz-delta--mute"}`}>
                {idx === 0 ? "reference" : spike ? "spike ▲" : "recovers ~½ ✓"}
              </span>
            </span>
          </div>
          <MetricBar cls="mse" values={mses} sel={idx} ariaLabel={mseAria} />
        </div>

        <div class="qz-metric qz-metric--lat">
          <div class="qz-metric-head" aria-hidden="true">
            <span class="qz-metric-name">latency (laptop CPU)</span>
            <span class="qz-metric-val">
              {ms(sel.latency_ms)} <span class="qz-metric-u">ms/call</span>
              <span class={`qz-metric-delta ${idx === 0 ? "qz-delta--mute" : "qz-delta--bad"}`}>
                {idx === 0 ? "baseline" : `${x1(latRatio(sel))}× slower ▲`}
              </span>
            </span>
          </div>
          <MetricBar cls="lat" values={lats} sel={idx} ariaLabel={latAria} />
          <p class="qz-lat-flag" aria-hidden="true">
            not a bug — naive INT8 has no fused kernel on CPU. Size is the win here; speed is the Scale Lab.
          </p>
        </div>
      </div>

      {/* --- THE DIAL: a 3-detent slider (the one interactive handle) ----------- */}
      <div class="qz-controls">
        <label class="qz-slider-lbl" for="qz-dial">quantization dial</label>
        <input
          id="qz-dial"
          class="qz-slider"
          type="range"
          min={0}
          max={2}
          step={1}
          value={idx}
          list="qz-detents"
          onInput={(e) => setIdx(Number((e.target as HTMLInputElement).value))}
          aria-valuetext={
            `${sel.label}: ${kb(sel.size_kb)} KB` +
            `${idx === 0 ? "" : `, ${x1(sizeRatio(sel))}× smaller`}, action error ` +
            `${sel.action_mse === 0 ? "zero (reference)" : `${mse4(sel.action_mse)}e-4 MSE`}, ` +
            `latency ${ms(sel.latency_ms)} ms${idx === 0 ? "" : ` (${x1(latRatio(sel))}× slower)`}`
          }
        />
        <datalist id="qz-detents">
          <option value="0" label="FP32" />
          <option value="1" label="per-tensor INT8" />
          <option value="2" label="per-channel INT8" />
        </datalist>
        <div class="qz-detent-labels" aria-hidden="true">
          {DETENTS.map((d, i) => (
            <button
              type="button"
              class="qz-detent"
              data-on={i === idx}
              onClick={() => setIdx(i)}
              tabIndex={-1}
            >
              {d}
            </button>
          ))}
        </div>
      </div>

      {/* --- the exact numbers + the honest success-rate reading (Wilson CI) ---- */}
      <figure class="qz-ledger">
        <figcaption class="qz-ledger-cap">
          <span>three configs · seed 0</span>
          <b>n = {FP32.n} rollouts / config</b>
        </figcaption>
        <div class="qz-ledger-grid" role="table" aria-label="Per-config size, action error, latency and success rate with Wilson confidence interval">
          <span class="qz-lg-h" role="columnheader">config</span>
          <span class="qz-lg-h" role="columnheader">size</span>
          <span class="qz-lg-h" role="columnheader">err ×10⁻⁴</span>
          <span class="qz-lg-h" role="columnheader">latency</span>
          <span class="qz-lg-h" role="columnheader">success [95% CI]</span>
          {CONFIGS.map((c, i) => (
            <>
              <span class="qz-lg-name" data-sel={i === idx} role="cell">{c.label}</span>
              <span class="qz-lg-v" data-sel={i === idx} role="cell">{kb(c.size_kb)} KB</span>
              <span class="qz-lg-v" data-sel={i === idx} role="cell">{c.action_mse === 0 ? "0" : mse4(c.action_mse)}</span>
              <span class="qz-lg-v" data-sel={i === idx} role="cell">{ms(c.latency_ms)} ms</span>
              <span class="qz-lg-v" data-sel={i === idx} role="cell">
                {pct(c.success_rate)}% [{pct(c.ci_lo)}–{pct(c.ci_hi)}%]
              </span>
            </>
          ))}
        </div>
        <p class="qz-ledger-note">
          All three success-rate intervals <b>overlap FP32's</b> — the size/accuracy trade is real in the{" "}
          <b>weights</b> (the MSE spike is measurable), but the <b>task-success</b> difference is inside the noise
          at n = {FP32.n}. Ship the small model; verify success on your own eval, not on the MSE.
        </p>
      </figure>

      {/* --- the companion static figure: why a scale, and why a per-ROW scale -- */}
      <figure class="qz-scale">
        <figcaption class="qz-scale-cap">rounding vs scaling — the granularity is the idea</figcaption>
        <div class="qz-scale-grid">
          <div class="qz-scale-stat">
            <span class="qz-scale-v qz-scale-v--bad">{pct(NAIVE_ZERO)}%</span>
            <span class="qz-scale-k">of weights collapse to <b>0</b> if you round with <b>no scale</b></span>
          </div>
          <div class="qz-scale-stat">
            <span class="qz-scale-v">
              {RT_PER_TENSOR.toFixed(4)} <span class="qz-scale-arrow">→</span> {RT_PER_CHANNEL.toFixed(4)}
            </span>
            <span class="qz-scale-k">
              weight round-trip error, <b>per-tensor → per-row</b> scale — cut ~{Math.round((1 - RT_PER_CHANNEL / RT_PER_TENSOR) * 100)}%
            </span>
          </div>
        </div>
        <p class="qz-scale-note">
          A scale turns a dead cast into a working policy; a <b>per-row</b> scale makes it accurate. The per-channel
          scale in the dial above is the same idea, measured on the actions.
        </p>
      </figure>

      {/* non-visual path to the same aha — the qualitative story, not per-frame spam */}
      <div class="bk-sr" aria-live="polite">{announce}</div>

      {/* --- the honest framing note (size win guaranteed, latency win is Scale Lab) */}
      <p class="qz-note">
        <b>Honest bottom line.</b> The <b>size</b> win is <b>guaranteed</b> — ~{x1(sizeRatio(CONFIGS[1]))}× fewer bytes
        on disk and in memory. The <b>latency</b> win is <b>not</b>: this from-scratch INT8 runs{" "}
        <b>~{Math.round(latRatio(CONFIGS[2]))}× slower</b> than FP32 on a laptop CPU, because there is no fused
        low-precision kernel — each matmul dequantizes back to float first. Real INT8 speedups need fused kernels or
        accelerators (the <b>Scale Lab</b>), not this laptop path. On your machine, quantization buys you <b>size</b>,
        not <b>speed</b>. Real measured numbers from quantize.py (seed 0, cpu); poster reads with JS off.
      </p>
    </div>
  );
}
