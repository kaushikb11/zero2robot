/**
 * VitAttentionToy — ch5.1 "Patches & Attention: A ViT From Scratch" concept-toy
 * (demo id `vit_attention_viewer`). THE attention-map viewer, made honest as a pure
 * DATA-VIEWER over precomputed grids — no MuJoCo-WASM, no ONNX, no dynamic import.
 * This chapter trains NO policy and exports NO model, so there is nothing to *run*:
 * the artifact emits a static representation + attention maps, and this is the
 * cheapest embed tier — it just reads demo/vizdata.json and draws an overlay.
 *
 * THE HERO. For each held-out frame we draw the 32x32 camera thumbnail (as SVG
 * pixels, no raster) with the 8x8 CLS-token attention-rollout overlaid as a heatmap,
 * one cell per patch. A TRAINED/RANDOM toggle swaps the trained encoder's attention
 * for a same-shape random-init encoder's: the trained map CONCENTRATES (~10-16x over
 * uniform) on the patches that contain the block; the random map is nearly FLAT
 * (~1.1x) — it washes across the whole frame. The one line to feel: "a ViT is a bag
 * of patches with position tags — training is what teaches its attention to look at
 * the object."
 *
 * All numbers are REAL: read verbatim from vit.py's committed seed-0 vizdata.json
 * (the attention grids are min-max normalized to [0,1] per frame; concentration is
 * reported as peak / mean, so uniform reads ~1x). Nothing is mocked.
 *
 * Pure inline SVG + design tokens: theme-aware for free (light AND dark), and the
 * server-rendered default (a full frame + its trained heatmap + the concentration
 * readout + the probe stat) IS the JS-off experience. Hydration only adds the
 * trained/random toggle, the frame stepper, and per-patch hover/keyboard inspection.
 *
 * Follows the FROZEN CONCEPT-TOY CONTRACT documented at the top of ../PlateIsland.tsx.
 */
import "./VitAttentionToy.css";
import { useMemo, useState } from "preact/hooks";
// Real precomputed frames + CLS attention-rollout grids from vit.py's reference run
// (seed 0, default config) — committed small text (numeric grids), no binary. Same
// co-located-vizdata pattern the other data-viewer toys use.
import viz from "../../../../curriculum/phase5_practitioner/ch5.1_vit/demo/vizdata.json";

// ---------------------------------------------------------------- typed vizdata
const GRID: number = viz.grid;        // 8 — patches per side (8x8 = 64 image tokens)
const PATCH: number = viz.patch;      // 8 — pixels per patch side
const IMG_HW: number = viz.img_hw;    // 64 — the full frame the ViT saw
const THUMB_HW: number = viz.thumb_hw; // 32 — frames[].image is emitted at 32x32
const QUADRANTS: string[] = viz.quadrants as string[]; // ["NE","NW","SW","SE"]
const PROBE_TRAINED: number = viz.probe_acc_trained;   // 0.891 — trained linear-probe acc
const PROBE_RANDOM: number = viz.probe_acc_random;     // 0.688 — random-init baseline
type Frame = {
  quadrant: number;         // index into QUADRANTS
  image: number[][][];      // 32 x 32 x 3 uint8
  attn_trained: number[][]; // 8 x 8, normalized [0,1]
  attn_random: number[][];  // 8 x 8, normalized [0,1]
};
const FRAMES: Frame[] = viz.frames as Frame[];

type Mode = "trained" | "random";

// the pixel size of one thumbnail pixel and one patch, in the shared IMG_HW viewBox.
const PXS = IMG_HW / THUMB_HW; // 2 — each 32x32 pixel covers a 2x2 block of the 64px frame
// (PATCH === IMG_HW / GRID === 8 — each patch cell covers an 8x8 block; the two grids align)

// ------------------------------------------------------------ number formatting
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

/** peak / mean of a normalized grid — "how many times uniform" the hottest patch is.
 *  Trained concentrates (~10-16x); random washes out flat (~1.1x). */
function concentration(grid: number[][]): number {
  let sum = 0;
  let max = 0;
  for (const row of grid) for (const v of row) { sum += v; max = Math.max(max, v); }
  const mean = sum / (GRID * GRID);
  return mean > 0 ? max / mean : 0;
}

/** (row, col) of the hottest patch — where the encoder looks most. */
function argmax(grid: number[][]): [number, number] {
  let br = 0, bc = 0, bv = -Infinity;
  for (let r = 0; r < grid.length; r++)
    for (let c = 0; c < grid[r].length; c++)
      if (grid[r][c] > bv) { bv = grid[r][c]; br = r; bc = c; }
  return [br, bc];
}

// =============================================================== THE HEATMAP SVG
// One shared IMG_HW x IMG_HW (64x64) viewBox. The thumbnail draws as 32x32 SVG
// pixels (2x2 each); the attention overlays as 8x8 patch cells (8x8 each), so the
// two grids line up exactly. SSR renders this verbatim — it is the JS-off view.

/** The camera frame as SVG pixels (no raster). Depends only on the frame, so it is
 *  memoized by the caller and does not re-diff when the mode/hover changes. */
function Thumbnail({ image }: { image: number[][][] }) {
  const rects = [];
  for (let i = 0; i < image.length; i++) {
    const row = image[i];
    for (let j = 0; j < row.length; j++) {
      const [r, g, b] = row[j];
      rects.push(
        <rect x={j * PXS} y={i * PXS} width={PXS} height={PXS} fill={`rgb(${r},${g},${b})`} />,
      );
    }
  }
  return <g class="vit-thumb">{rects}</g>;
}

/** The 8x8 attention heatmap. Opacity tracks the (normalized) attention weight, so a
 *  concentrated trained map shows one hot patch over a dark frame, while a flat random
 *  map shows an even blue wash across every patch. */
function Heatmap({ grid }: { grid: number[][] }) {
  const cells = [];
  for (let r = 0; r < grid.length; r++) {
    for (let c = 0; c < grid[r].length; c++) {
      const w = grid[r][c];
      cells.push(
        <rect
          class="vit-heat"
          x={c * PATCH}
          y={r * PATCH}
          width={PATCH}
          height={PATCH}
          style={`opacity:${(0.9 * w).toFixed(3)}`}
        />,
      );
    }
  }
  return <g>{cells}</g>;
}

// ==================================================================== THE ISLAND
export default function VitAttentionToy() {
  // default-interesting: frame 0, TRAINED — the concentrated map is the aha, and it
  // is what SSR renders, so the JS-off view already tells the story.
  const [frameIdx, setFrameIdx] = useState(0);
  const [mode, setMode] = useState<Mode>("trained");
  const [hover, setHover] = useState<[number, number] | null>(null);

  const frame = FRAMES[frameIdx];
  const grid = mode === "trained" ? frame.attn_trained : frame.attn_random;
  const conc = concentration(grid);
  const [pr, pc] = argmax(grid);
  const quad = QUADRANTS[frame.quadrant] ?? `#${frame.quadrant}`;

  // the thumbnail only depends on the frame; memoize so hover/mode changes don't
  // re-diff 1024 pixel rects.
  const thumb = useMemo(() => <Thumbnail image={frame.image} />, [frameIdx]);
  // the heatmap depends on frame + mode.
  const heat = useMemo(() => <Heatmap grid={grid} />, [frameIdx, mode]);

  const trained = mode === "trained";
  const hoverW = hover ? grid[hover[0]][hover[1]] : null;

  const svgLabel =
    `Attention heatmap over an ${GRID} by ${GRID} patch grid, overlaid on a ${THUMB_HW} by ${THUMB_HW} ` +
    `camera thumbnail of a block in the ${quad} quadrant. Encoder: ${mode}. ` +
    (trained
      ? `The trained CLS-token attention concentrates about ${conc.toFixed(0)} times over uniform on the ` +
        `patches that contain the block, at grid row ${pr + 1}, column ${pc + 1}.`
      : `The random-init CLS-token attention is nearly flat, about ${conc.toFixed(1)} times uniform — ` +
        `it washes across the whole frame and does not single out the block.`);

  const announce =
    `Frame ${frameIdx + 1} of ${FRAMES.length}, block in the ${quad} quadrant. ${mode} encoder. ` +
    (trained
      ? `Attention concentrates about ${conc.toFixed(0)}x over uniform on the block.`
      : `Attention is nearly flat, about ${conc.toFixed(1)}x uniform — no object focus.`) +
    (hover ? ` Inspecting patch row ${hover[0] + 1}, column ${hover[1] + 1}: weight ${(hoverW ?? 0).toFixed(2)}.` : "");

  const onKeyDown = (e: KeyboardEvent) => {
    const k = e.key;
    if (k === "t" || k === "T") { e.preventDefault(); setMode((m) => (m === "trained" ? "random" : "trained")); return; }
    if (k === "]" || k === "." || k === "PageDown") { e.preventDefault(); setFrameIdx((i) => (i + 1) % FRAMES.length); return; }
    if (k === "[" || k === "," || k === "PageUp") { e.preventDefault(); setFrameIdx((i) => (i - 1 + FRAMES.length) % FRAMES.length); return; }
    // arrow keys inspect patches (a keyboard path to the same hover readout)
    const cur = hover ?? [0, 0];
    const step: Record<string, [number, number]> = {
      ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1],
    };
    if (k in step) {
      e.preventDefault();
      const [dr, dc] = step[k];
      setHover([
        Math.max(0, Math.min(GRID - 1, cur[0] + dr)),
        Math.max(0, Math.min(GRID - 1, cur[1] + dc)),
      ]);
    } else if (k === "Escape") {
      setHover(null);
    }
  };

  return (
    <div class="vit">
      <header class="vit-head">
        <h3 class="vit-title">Where does the CLS token look?</h3>
        <p class="vit-sub">
          The <b>CLS-token attention-rollout</b> over the <b>{GRID}×{GRID} patch grid</b>, overlaid on the frame the
          ViT saw. The <b>trained</b> encoder concentrates its attention on the patches that hold the block; a{" "}
          <b>random-init</b> encoder of the same shape washes out flat. A ViT is a <b>bag of patches with position
          tags</b> — training is what teaches its attention to look at the object.
        </p>
      </header>

      <div class="vit-stage">
        <figure
          class="vit-fig"
          tabIndex={0}
          role="group"
          aria-label="Interactive ViT attention viewer. Press T to toggle trained versus random-init encoder, bracket keys to step through frames, and arrow keys to inspect individual patches."
          onKeyDown={onKeyDown}
        >
          <svg
            class="vit-svg"
            viewBox={`0 0 ${IMG_HW} ${IMG_HW}`}
            role="img"
            aria-label={svgLabel}
            data-mode={mode}
          >
            <title>CLS-token attention over the patch grid</title>
            {/* the camera frame — SVG pixels, no raster */}
            {thumb}
            {/* a scrim so the heatmap reads against a bright thumbnail */}
            <rect class="vit-scrim" x={0} y={0} width={IMG_HW} height={IMG_HW} />
            {/* the CLS-token attention heatmap */}
            {heat}
            {/* patch gridlines */}
            <g class="vit-gridlines">
              {Array.from({ length: GRID - 1 }, (_, i) => (i + 1) * PATCH).map((p) => (
                <>
                  <line x1={p} y1={0} x2={p} y2={IMG_HW} />
                  <line x1={0} y1={p} x2={IMG_HW} y2={p} />
                </>
              ))}
            </g>
            {/* the hottest patch — a ring the eye lands on (SSR-visible) */}
            <rect class="vit-peak" x={pc * PATCH} y={pr * PATCH} width={PATCH} height={PATCH} />
            {/* the hovered / keyboard-inspected patch */}
            {hover && (
              <rect class="vit-cursor" x={hover[1] * PATCH} y={hover[0] * PATCH} width={PATCH} height={PATCH} />
            )}
            {/* transparent hover targets, one per patch (hydration-only inspection) */}
            <g class="vit-hit">
              {Array.from({ length: GRID }, (_, r) =>
                Array.from({ length: GRID }, (_, c) => (
                  <rect
                    x={c * PATCH}
                    y={r * PATCH}
                    width={PATCH}
                    height={PATCH}
                    onPointerEnter={() => setHover([r, c])}
                    onPointerLeave={() => setHover((h) => (h && h[0] === r && h[1] === c ? null : h))}
                  />
                )),
              )}
            </g>
          </svg>

          <figcaption class="vit-cap" aria-hidden="true">
            frame {frameIdx + 1}/{FRAMES.length} · block in <b>{quad}</b> · {mode} encoder
          </figcaption>
        </figure>

        {/* the readout panel — concentration + the hovered patch */}
        <div class="vit-panel" aria-hidden="true">
          <div class="vit-conc" data-mode={mode}>
            <span class="vit-conc-k">peak attention</span>
            <span class="vit-conc-v">
              {trained ? `≈${conc.toFixed(0)}×` : `≈${conc.toFixed(1)}×`} <span class="vit-conc-u">uniform</span>
            </span>
            <span class="vit-conc-sub">
              {trained
                ? "concentrated on the block — training taught it where to look"
                : "nearly flat — an untrained encoder looks everywhere at once"}
            </span>
          </div>

          <div class="vit-hoverbox">
            {hover ? (
              <>
                <span class="vit-hover-k">patch (row {hover[0] + 1}, col {hover[1] + 1})</span>
                <span class="vit-hover-v">weight {(hoverW ?? 0).toFixed(3)}</span>
              </>
            ) : (
              <span class="vit-hover-hint">hover a patch — or focus the frame and use arrow keys</span>
            )}
          </div>

          <div class="vit-probe">
            <span class="vit-probe-k">linear-probe accuracy</span>
            <span class="vit-probe-row">
              <span class="vit-probe-hi">trained {pct(PROBE_TRAINED)}</span>
              <span class="vit-probe-lo">random-init {pct(PROBE_RANDOM)}</span>
            </span>
          </div>
        </div>
      </div>

      {/* --- controls: trained/random toggle + frame stepper (keyboard-accessible) --- */}
      <div class="vit-controls">
        <div class="vit-toggle" role="group" aria-label="Encoder: trained versus random-init">
          <button
            type="button"
            class="vit-tbtn"
            data-on={trained}
            aria-pressed={trained}
            onClick={() => setMode("trained")}
          >
            trained
          </button>
          <button
            type="button"
            class="vit-tbtn"
            data-on={!trained}
            aria-pressed={!trained}
            onClick={() => setMode("random")}
          >
            random-init
          </button>
        </div>

        <div class="vit-frames" role="group" aria-label="Select frame">
          {FRAMES.map((f, i) => (
            <button
              type="button"
              class="vit-fbtn"
              data-on={i === frameIdx}
              aria-pressed={i === frameIdx}
              aria-label={`Frame ${i + 1}, block in the ${QUADRANTS[f.quadrant] ?? `#${f.quadrant}`} quadrant`}
              onClick={() => setFrameIdx(i)}
            >
              {QUADRANTS[f.quadrant] ?? `#${f.quadrant}`}
            </button>
          ))}
        </div>

        <span class="vit-control-note">toggle trained/random · step through {FRAMES.length} frames · poster reads with JS off</span>
      </div>

      {/* non-visual path to the same aha — the qualitative story, not per-pixel spam */}
      <div class="vit-sr" aria-live="polite">{announce}</div>

      {/* the honest framing */}
      <p class="vit-note">
        The <b>same accuracy</b> hides the difference: a coarse scene fact like which quadrant holds the block is a
        <b> permutation-invariant bag-of-patches property</b>, so it survives even a scrambled patch order — the bug
        only shows in the <b>attention map</b>, never in the metric. Real precomputed grids from vit.py
        (seed 0, {GRID}×{GRID} patches, {IMG_HW}px frames); attention is min-max normalized per frame, concentration
        reported as peak ÷ mean. Poster reads with JS off.
      </p>
    </div>
  );
}
