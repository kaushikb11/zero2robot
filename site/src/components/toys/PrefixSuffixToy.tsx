/**
 * PrefixSuffixToy — ch5.4 "The Production VLA Shape" concept-toy
 * (demo id `prefix_suffix_attention`). THE block-attention-mask viewer, made honest
 * as a pure DATA-VIEWER over precomputed data — no MuJoCo-WASM, no ONNX, no dynamic
 * import. This chapter trains a policy but exports NO onnx and the PushT rollout
 * FLOORS at free-tier, so there is nothing to drive live: the artifact emits two
 * block-attention masks + the held-out flow-MSE numbers, and this is the cheapest
 * embed tier — it just reads demo/vizdata.json and draws a grid.
 *
 * THE HERO. We draw the full (P+H)×(P+H) attention mask as a heatmap: rows = query
 * token, cols = key token, labelled by labels_seq (the prefix block [vision, state,
 * tok0..tok11] then the suffix block [act0..act7]). Four quadrants are shaded and
 * named: prefix↔prefix (the VLM fusion), suffix→prefix (the cross-attention — the
 * hero), suffix↔suffix (the action chunk coordinating), and prefix→suffix (always
 * dark — the prefix never reads the actions, which is what makes it KV-cacheable).
 * A single toggle CUTS the suffix→prefix quadrant (mask_full → mask_cut) and, in
 * lockstep, swaps the held-out flow-MSE delta bars: the SAME trained expert, now
 * blind to the VLM, and the measured velocity fit collapses (1.57 → 2.54, +0.98).
 *
 * WHAT IS REAL, AND WHAT IS NOT. The reproducible claim is the flow-MSE routing gap
 * (+0.6..1.0 across seeds), read verbatim from vla_shape.py's committed seed-0
 * vizdata.json — nothing mocked. The recorded PushT rollout, by contrast, FLOORS
 * (0/8 success) for BOTH masks: a from-scratch action expert on a frozen-RANDOM
 * vision backbone cannot drive PushT at free-tier (that is the Scale Lab, and ch5.2's
 * aligned encoder is the upgrade). So this toy shows the MEASURED signal — the mask
 * cell and the flow-MSE collapse — and never implies the two-tower drives PushT.
 *
 * Pure inline SVG + design tokens: theme-aware for free (light AND dark), and the
 * server-rendered default (the full mask grid + both flow-MSE bars + the readout) IS
 * the JS-off experience. Hydration only adds the cut/restore toggle, per-cell
 * hover/keyboard inspection, and the aria-live readout.
 *
 * Follows the FROZEN CONCEPT-TOY CONTRACT documented at the top of ../PlateIsland.tsx.
 */
import "./PrefixSuffixToy.css";
import { useMemo, useState } from "preact/hooks";
// Real precomputed masks + held-out flow-MSE from vla_shape.py's reference run
// (seed 0, exercise_config, H=8) — committed small text (numeric grids), no binary.
// Same co-located-vizdata pattern the other data-viewer toys use.
import viz from "../../../../curriculum/phase5_practitioner/ch5.4_vla_shape/demo/vizdata.json";

// ---------------------------------------------------------------- typed vizdata
const P: number = viz.prefix_len;              // 14 — 1 vision + 1 state + 12 instruction tokens
const H: number = viz.horizon;                 // 8  — action-expert tokens (the chunk)
const N = P + H;                                // 22 — the full sequence length
const LABELS: string[] = viz.labels_seq as string[];
const MASK_FULL: number[][] = viz.mask_full as number[][];
const MASK_CUT: number[][] = viz.mask_cut as number[][];
// len N, avg action-token attn from the FINAL block only (full mask). This last layer
// self-concentrates WITHIN the action chunk (prefix keys ≈ 0), so the strip depicts within-chunk
// self-attention — NOT where the expert reads the prefix. The prefix-reading is proven by the
// +0.98 flow-MSE collapse when the cross-attention is cut (MSE_GAP below), not by these weights.
const SUFFIX_ATTN: number[] = viz.suffix_attention as number[];
const MSE_FULL: number = viz.meta.flow_mse_full;   // 1.566…
const MSE_CUT: number = viz.meta.flow_mse_cut;     // 2.543…
const MSE_GAP: number = viz.meta.flow_mse_gap;     // 0.977…
// The recorded PushT rollouts — kept only to state the honest floor (both fail).
// 8 eval episodes per mask; successes below (both 0 → the free-tier floor).
const FULL_SUCC: number = viz.full.success ? 1 : 0;    // 0
const CUT_SUCC: number = viz.cut.success ? 1 : 0;      // 0

// ------------------------------------------------------------ number formatting
const f2 = (v: number) => v.toFixed(2);

// the four attention blocks, by (query-row, key-col) quadrant.
type Quad = "pp" | "sp" | "ss" | "ps";
function quadOf(r: number, c: number): Quad {
  if (r < P) return c < P ? "pp" : "ps";       // prefix query: reads prefix (fusion) | never suffix (blocked)
  return c < P ? "sp" : "ss";                  // suffix query: reads prefix (cross-attn) | suffix (chunk)
}
const QUAD_NAME: Record<Quad, string> = {
  pp: "prefix ↔ prefix · VLM fusion",
  sp: "suffix → prefix · cross-attention",
  ss: "suffix ↔ suffix · action chunk",
  ps: "prefix → suffix · always blocked (KV-cacheable)",
};

// ===================================================================== THE GRID
// One shared viewBox: a left/top label gutter, then the N×N cell grid, then a thin
// strip under the columns for the recorded avg action-token attention. SSR renders
// this verbatim (default: the FULL mask) — it is the JS-off view.
const CELL = 13;
const LG = 52;                 // left gutter (row labels)
const TG = 52;                 // top gutter (col labels)
const GRID = N * CELL;         // 286
const STRIP_GAP = 6;
const STRIP_H = 20;
const VBW = LG + GRID + 2;
const VBH = TG + GRID + STRIP_GAP + STRIP_H + 26;
const cxCol = (c: number) => LG + c * CELL + CELL / 2;
const cyRow = (r: number) => TG + r * CELL + CELL / 2;

// max avg-attention, to normalise the faint suffix-attention strip
const ATTN_MAX = Math.max(...SUFFIX_ATTN, 1e-6);

// ==================================================================== THE ISLAND
export default function PrefixSuffixToy() {
  // default-interesting: the FULL mask — the intact routing is the baseline the cut
  // collapses from, and it is what SSR renders, so the JS-off view tells the story.
  const [cut, setCut] = useState(false);
  const [cur, setCur] = useState<[number, number] | null>(null); // hovered/inspected cell

  const mask = cut ? MASK_CUT : MASK_FULL;

  // MSE bar scale — a shared axis holding the larger (cut) bar with headroom.
  const MSE_MAX = useMemo(() => Math.max(MSE_FULL, MSE_CUT) * 1.14, []);
  const fullPct = (MSE_FULL / MSE_MAX) * 100;
  const cutPct = (MSE_CUT / MSE_MAX) * 100;

  const curQuad = cur ? quadOf(cur[0], cur[1]) : null;
  const curAllowed = cur ? mask[cur[0]][cur[1]] === 1 : false;

  const announce =
    `Attention mask: ${cut ? "cross-attention CUT" : "full (intact)"}. ` +
    `The action expert ${cut ? "can no longer read" : "reads"} the prefix — vision, state, and language. ` +
    `Held-out flow-MSE ${cut ? `rises to ${f2(MSE_CUT)}, a gap of +${f2(MSE_GAP)} above the full mask (${f2(MSE_FULL)})` : `is ${f2(MSE_FULL)}`}. ` +
    (cur
      ? `Inspecting query ${LABELS[cur[0]]} to key ${LABELS[cur[1]]}: ${curAllowed ? "allowed" : "blocked"} — ${QUAD_NAME[curQuad!]}.`
      : `Cutting one block of the mask severs the expert's only path to the state.`);

  const onKeyDown = (e: KeyboardEvent) => {
    const k = e.key;
    if (k === "c" || k === "C") { e.preventDefault(); setCut((v) => !v); return; }
    if (k === "Escape") { setCur(null); return; }
    const cur0 = cur ?? [P, 0]; // start in the cross-attention quadrant — the hero
    const step: Record<string, [number, number]> = {
      ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1],
    };
    if (k in step) {
      e.preventDefault();
      const [dr, dc] = step[k];
      setCur([
        Math.max(0, Math.min(N - 1, cur0[0] + dr)),
        Math.max(0, Math.min(N - 1, cur0[1] + dc)),
      ]);
    }
  };

  // grid cells — data-driven from the ACTIVE mask (never hardcoded lit/dark).
  const cells = [];
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      cells.push(
        <rect
          class="ps-cell"
          x={LG + c * CELL}
          y={TG + r * CELL}
          width={CELL}
          height={CELL}
          data-q={quadOf(r, c)}
          data-on={mask[r][c] === 1}
        />,
      );
    }
  }

  return (
    <div class="ps">
      <header class="ps-head">
        <h3 class="ps-title">The production VLA shape — one cell from blind</h3>
        <p class="ps-sub">
          The <b>block-attention mask</b> is the whole architecture: a{" "}
          <b>[ prefix | suffix ]</b> sequence where the action expert (suffix) reads the frozen VLM (prefix) through
          exactly <b>one quadrant</b> — the <b>suffix → prefix cross-attention</b>. Cut that block and the held-out
          <b> flow-MSE jumps {f2(MSE_FULL)} → {f2(MSE_CUT)}</b> (a <b>+{f2(MSE_GAP)}</b> routing gap — a large,
          positive collapse on every seed): the expert loses its only path to the state. Routing, not just
          parameters, is load-bearing.
        </p>
      </header>

      <div class="ps-stage">
        {/* ---- the mask heatmap (the hero) ---- */}
        <figure
          class="ps-fig"
          tabIndex={0}
          role="group"
          aria-label="Interactive block-attention mask. Press C to cut or restore the suffix-to-prefix cross-attention, and arrow keys to inspect individual query-key cells."
          onKeyDown={onKeyDown}
        >
          <svg
            class="ps-svg"
            viewBox={`0 0 ${VBW} ${VBH}`}
            role="img"
            data-cut={cut}
            aria-label={
              `Block-attention mask, ${N} by ${N}: rows are query tokens, columns are key tokens, ordered as the ` +
              `${P}-token prefix (vision, state, 12 language tokens) then the ${H}-token action suffix. A lit cell ` +
              `means the query may attend to the key. Four blocks: prefix-to-prefix (the VLM fusion) is lit, ` +
              `suffix-to-suffix (the action chunk) is lit, prefix-to-suffix is always dark (the prefix never reads the ` +
              `actions, which keeps it cacheable), and suffix-to-prefix is the cross-attention. ` +
              (cut
                ? `Here the cross-attention is CUT — the suffix-to-prefix block is dark, so the action expert cannot read ` +
                  `the prefix at all, and the held-out flow-MSE has collapsed to ${f2(MSE_CUT)}.`
                : `Here the mask is full — the suffix-to-prefix block is lit, so the action expert reads the state, and ` +
                  `the held-out flow-MSE is ${f2(MSE_FULL)}.`)
            }
          >
            <title>The block-attention mask ({cut ? "cross-attention cut" : "full"})</title>

            {/* column (key) labels — rotated up out of the top gutter */}
            {LABELS.map((lab, c) => (
              <text
                class="ps-lbl"
                x={cxCol(c)}
                y={TG - 4}
                text-anchor="start"
                data-suffix={c >= P}
                transform={`rotate(-90 ${cxCol(c)} ${TG - 4})`}
              >
                {lab}
              </text>
            ))}
            {/* row (query) labels */}
            {LABELS.map((lab, r) => (
              <text class="ps-lbl" x={LG - 5} y={cyRow(r) + 3} text-anchor="end" data-suffix={r >= P}>
                {lab}
              </text>
            ))}

            {/* the cells (data-driven from the active mask) */}
            <g class="ps-cells">{cells}</g>

            {/* block dividers between the prefix and suffix blocks */}
            <g class="ps-divide">
              <line x1={LG + P * CELL} y1={TG} x2={LG + P * CELL} y2={TG + GRID} />
              <line x1={LG} y1={TG + P * CELL} x2={LG + GRID} y2={TG + P * CELL} />
            </g>

            {/* the hero outline: the suffix→prefix cross-attention quadrant */}
            <rect
              class="ps-xregion"
              x={LG}
              y={TG + P * CELL}
              width={P * CELL}
              height={H * CELL}
              data-cut={cut}
            />

            {/* the inspected cell cursor */}
            {cur && (
              <rect
                class="ps-cursor"
                x={LG + cur[1] * CELL}
                y={TG + cur[0] * CELL}
                width={CELL}
                height={CELL}
              />
            )}

            {/* transparent per-cell hover targets (hydration-only inspection) */}
            <g class="ps-hit">
              {Array.from({ length: N }, (_, r) =>
                Array.from({ length: N }, (_, c) => (
                  <rect
                    x={LG + c * CELL}
                    y={TG + r * CELL}
                    width={CELL}
                    height={CELL}
                    onPointerEnter={() => setCur([r, c])}
                    onPointerLeave={() => setCur((h) => (h && h[0] === r && h[1] === c ? null : h))}
                  />
                )),
              )}
            </g>

            {/* the FINAL block's avg action-token attention, per key (full mask) — a faint strip.
                Honest: the last layer self-concentrates within the action chunk (prefix keys ≈ 0),
                so this is within-chunk self-attention, NOT prefix-reading. The routing is proven
                by the +MSE_GAP flow-MSE collapse when the cross-attention is cut, not these weights. */}
            <g class="ps-strip">
              {SUFFIX_ATTN.map((a, c) => {
                const h = (a / ATTN_MAX) * STRIP_H;
                const y = TG + GRID + STRIP_GAP + (STRIP_H - h);
                return <rect class="ps-strip-bar" x={LG + c * CELL + 1.5} y={y} width={CELL - 3} height={Math.max(0, h)} />;
              })}
              <text class="ps-strip-lbl" x={LG} y={TG + GRID + STRIP_GAP + STRIP_H + 10}>
                final-block self-attention: last layer self-concentrates on the action tokens
              </text>
              <text class="ps-strip-lbl" x={LG} y={TG + GRID + STRIP_GAP + STRIP_H + 19}>
                (prefix keys ≈ 0) — NOT prefix-reading; proven by the +{f2(MSE_GAP)} flow-MSE collapse
              </text>
            </g>
          </svg>

          <figcaption class="ps-cap" aria-hidden="true">
            rows = query · cols = key · lit = may attend · {cut ? "cross-attention CUT" : "full mask"}
          </figcaption>
        </figure>

        {/* ---- the readout: the flow-MSE routing gap (the reproducible claim) ---- */}
        <div class="ps-panel" aria-hidden="true">
          <div class="ps-mse">
            <div class="ps-mse-cap">
              <span class="ps-mse-k">held-out flow-MSE</span>
              <span class="ps-mse-note">lower fits the expert velocity better</span>
            </div>

            <div class="ps-bar" data-active={!cut}>
              <span class="ps-bar-k">full mask</span>
              <span class="ps-bar-track"><span class="ps-bar-fill ps-bar-fill--full" style={`width:${fullPct.toFixed(1)}%`} /></span>
              <span class="ps-bar-v">{f2(MSE_FULL)}</span>
            </div>

            <div class="ps-bar" data-active={cut}>
              <span class="ps-bar-k">cross-attn cut</span>
              <span class="ps-bar-track"><span class="ps-bar-fill ps-bar-fill--cut" style={`width:${cutPct.toFixed(1)}%`} /></span>
              <span class="ps-bar-v">{f2(MSE_CUT)}</span>
            </div>

            <div class="ps-gap">
              <span class="ps-gap-v">+{f2(MSE_GAP)}</span>
              <span class="ps-gap-k">routing gap · seed-robust — a large positive collapse across seeds 0/1/2</span>
            </div>
          </div>

          <div class="ps-inspect">
            {cur ? (
              <>
                <span class="ps-inspect-h">
                  <b>{LABELS[cur[0]]}</b> → <b>{LABELS[cur[1]]}</b>
                </span>
                <span class="ps-inspect-v" data-on={curAllowed}>{curAllowed ? "allowed" : "blocked"}</span>
                <span class="ps-inspect-sub">{QUAD_NAME[curQuad!]}</span>
              </>
            ) : (
              <span class="ps-inspect-hint">hover a cell — or focus the grid and use arrow keys — to inspect a query→key edge</span>
            )}
          </div>
        </div>
      </div>

      {/* ---- the one control: cut / restore the cross-attention ---- */}
      <div class="ps-controls">
        <button
          type="button"
          class="ps-cut-btn"
          data-cut={cut}
          aria-pressed={cut}
          onClick={() => setCut((v) => !v)}
        >
          {cut ? "restore the cross-attention" : "cut the suffix → prefix cross-attention"}
        </button>
        <span class="ps-legend">
          <span class="ps-leg ps-leg--pp">prefix↔prefix</span>
          <span class="ps-leg ps-leg--sp">suffix→prefix</span>
          <span class="ps-leg ps-leg--ss">suffix↔suffix</span>
          <span class="ps-leg ps-leg--off">blocked</span>
        </span>
        <span class="ps-control-note">flow-MSE is the measured signal · poster reads with JS off</span>
      </div>

      {/* non-visual path to the same aha — the qualitative story, not per-cell spam */}
      <div class="ps-sr" aria-live="polite">{announce}</div>

      {/* the honest framing — what is measured vs what floors */}
      <p class="ps-note">
        The reproducible result is the <b>flow-MSE routing gap</b> ({f2(MSE_FULL)} → {f2(MSE_CUT)},{" "}
        <b>+{f2(MSE_GAP)}</b>, a large positive collapse on every seed 0/1/2): severing the suffix→prefix block
        collapses the trained expert's held-out velocity fit toward the unconditional prior. The recorded PushT rollout,
        by contrast,{" "}
        <b>floors for both masks</b> ({FULL_SUCC}/8 full · {CUT_SUCC}/8 cut) — a from-scratch action expert on a{" "}
        <b>frozen-random</b> vision backbone can't drive PushT at free-tier, so the rollout is the <b>Scale Lab</b>, not
        the lesson. Real precomputed masks + MSE from vla_shape.py (seed 0, H={H}); the two-tower shape does <b>not</b>{" "}
        drive PushT here — <b>ch5.2's aligned encoder</b> is the upgrade that makes pixels load-bearing. Poster reads
        with JS off.
      </p>
    </div>
  );
}
