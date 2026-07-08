/**
 * AlignRetrievalToy — ch5.2 "Why Aligned: Contrastive Vision-Language" concept-toy
 * (demo id `aligned_vs_random_retrieval`). A pure DATA-VIEWER over precomputed cosine
 * retrieval rankings — NO MuJoCo-WASM, NO onnx, no live compute, no dynamic import.
 *
 * THE LESSON, SIDE BY SIDE. Type (or pick) an instruction — "the block is near the
 * top left corner" — and watch which held-out scenes light up under two encoders:
 *   · the ALIGNED tower (symmetric InfoNCE) returns the top-5 scenes that ACTUALLY sit
 *     in that corner — language pulled the right scenes to the top of the ranking;
 *   · the RANDOM-INIT tower returns noise — the same few high-norm scenes recur for
 *     different instructions (frame 189 is in all four of its top-5 lists), scattered
 *     across corners. Contrastive pretraining built a shared image-text space from
 *     PAIRING ALONE; random init did not. That contrast is the whole chapter.
 *
 * WHY A DATA VIEWER, NOT A LIVE RETRIEVAL. The rankings are a property of the TRAINED
 * encoders over a fixed held-out gallery; recomputing them in the browser would need
 * both towers + the image features. So align.py precomputes the REAL top-5 lists (seed
 * 0, cpu, 249 held-out frames) into the co-located demo/vizdata.json, and this island
 * only reads them. Each gallery "frame" is a block POSITION (tee_xy from sim state),
 * rendered as a small top-down scene marker — there are NO camera-image binaries.
 *
 * SSR POSTER = JS-OFF EXPERIENCE. The component server-renders the DEFAULT instruction
 * ("top left") with BOTH galleries fully drawn and both hit-readouts filled — that
 * server output IS the complete JS-off experience for the default query. Hydration only
 * makes the instruction picker interactive (re-rank the two galleries). No heavy deps,
 * so it is theme-aware for free via the flipping design tokens.
 *
 * COLOUR by MEANING (a data panel, driven entirely by the flipping tokens): a retrieved
 * scene that really sits in the named corner is a HIT → --entity-target-ink (green); one
 * in the wrong corner is a MISS → --alert (red). The shaded target corner + the 249 faint
 * background scenes are neutral ink. Aligned lights up green IN the corner; random scatters
 * red across the plot. Follows the FROZEN CONCEPT-TOY CONTRACT in ../PlateIsland.tsx.
 */
import "./AlignRetrievalToy.css";
import { useMemo, useState } from "preact/hooks";
// Real precomputed cosine rankings + gallery block positions from align.py's reference
// run — see the file's `provenance`. Committed small text, no binary. Imported directly
// at build (same wiring as every other vizdata toy); nothing is copied into site/.
import viz from "../../../../curriculum/phase5_practitioner/ch5.2_align/demo/vizdata.json";

// ---------------------------------------------------------------- typed vizdata
interface Frame { idx: number; tee_x: number; tee_y: number; quadrant: number; near_far: number }
interface Query { quadrant: number; instruction: string }
const QUADRANT_WORDS: string[] = viz.quadrant_words;         // idx by quadrant 0..3
const GALLERY: Frame[] = viz.gallery as Frame[];             // 249 held-out scenes
const QUERIES: Query[] = viz.queries as Query[];             // the 4 canonical instructions
const ALIGNED_TOP5: number[][] = viz.aligned_top5 as number[][]; // idx by quadrant → top-5 frame idxs
const RANDOM_TOP5: number[][] = viz.random_top5 as number[][];
const N_GALLERY = GALLERY.length;

// gallery frames looked up by their idx (position == idx here, but stay robust)
const BY_IDX = new Map<number, Frame>();
GALLERY.forEach((f) => BY_IDX.set(f.idx, f));

// the query the demo boots on — "top left" (quadrant 2), per demo/embed.yaml.
const DEFAULT_SEL = Math.max(0, QUERIES.findIndex((q) => q.quadrant === 2));

/** how many of an encoder's top-5 for a query actually sit in the named corner. */
function hitsFor(top5: number[], quadrant: number): number {
  return top5.reduce((n, i) => n + (BY_IDX.get(i)?.quadrant === quadrant ? 1 : 0), 0);
}

// the honest aggregate, computed straight from the shown rankings (aligned 20/20 in-corner,
// random 6/20). This is the headline — derived from the SAME lists the galleries draw, not
// a separately-imported metric that could drift.
const TOTAL = QUERIES.length * 5;
const ALIGNED_HITS = QUERIES.reduce((n, q) => n + hitsFor(ALIGNED_TOP5[q.quadrant], q.quadrant), 0);
const RANDOM_HITS = QUERIES.reduce((n, q) => n + hitsFor(RANDOM_TOP5[q.quadrant], q.quadrant), 0);

/** map free text to the nearest canonical query by matching its direction words
 *  ("top"/"bottom"/"left"/"right"); null if nothing matches (keep current). We only
 *  HAVE four precomputed queries, so this is honest nearest-canonical, not live encoding. */
function matchQuery(text: string): number | null {
  const t = text.toLowerCase();
  let best = -1, bestScore = 0;
  QUERIES.forEach((q, i) => {
    let s = 0;
    for (const w of QUADRANT_WORDS[q.quadrant].split(" ")) if (t.includes(w)) s += 1;
    if (s > bestScore) { bestScore = s; best = i; }
  });
  return bestScore > 0 ? best : null;
}

// ------------------------------------------------------------------- scene plot
// top-down table square; block position (tee_x, tee_y) in metres, +y up (flip for SVG).
// A center cross splits the four quadrants; the query's target corner is shaded. The 249
// scenes are faint neutral dots; the encoder's top-5 are drawn large, green if the scene
// really sits in the corner (a hit) or red if not (a miss).
const E = 0.25;                       // ± half-extent (m) that holds every tee_xy
const SZ = 176, PAD = 16;
const V = SZ + PAD * 2;               // square viewBox
const CTR = PAD + SZ / 2;             // sx(0) == sy(0)
const sx = (x: number) => PAD + ((x + E) / (2 * E)) * SZ;
const sy = (y: number) => PAD + ((E - y) / (2 * E)) * SZ; // world +y up

/** SVG rect of a quadrant's corner (left/top flags derived from the quadrant word). */
function quadRect(q: number) {
  const left = q === 0 || q === 2;    // bottom-left, top-left
  const top = q === 2 || q === 3;     // top-left, top-right
  return { x: left ? PAD : CTR, y: top ? PAD : CTR, w: SZ / 2, h: SZ / 2 };
}

function Gallery({ kind, query }: { kind: "aligned" | "random"; query: Query }) {
  const top5 = (kind === "aligned" ? ALIGNED_TOP5 : RANDOM_TOP5)[query.quadrant];
  const inTop5 = new Set(top5);
  const word = QUADRANT_WORDS[query.quadrant];
  const hits = hitsFor(top5, query.quadrant);
  const r = quadRect(query.quadrant);
  const clustered =
    kind === "aligned"
      ? `the lit scenes cluster in the ${word} corner — language pulled the right scenes to the top`
      : `the lit scenes are scattered across corners — cosine over an untrained space is noise`;
  return (
    <svg
      class="ar-plot"
      viewBox={`0 0 ${V} ${V}`}
      role="img"
      aria-label={
        `${kind === "aligned" ? "Aligned contrastive" : "Random-init"} encoder, top-down table of ${N_GALLERY} ` +
        `held-out scenes plotted by block position. For the instruction "${query.instruction}", the ${word} corner ` +
        `is shaded and the encoder's top 5 cosine matches are drawn large: ${hits} of 5 sit inside the ${word} ` +
        `corner. ${clustered}.`
      }
    >
      {/* the shaded target corner — where the instruction points */}
      <rect class="ar-corner" x={r.x} y={r.y} width={r.w} height={r.h} />
      {/* table edge + the quadrant cross */}
      <rect class="ar-table" x={PAD} y={PAD} width={SZ} height={SZ} rx={4} />
      <line class="ar-cross" x1={CTR} y1={PAD} x2={CTR} y2={PAD + SZ} />
      <line class="ar-cross" x1={PAD} y1={CTR} x2={PAD + SZ} y2={CTR} />
      {/* corner words, faint */}
      {QUADRANT_WORDS.map((w, q) => {
        const rr = quadRect(q);
        return (
          <text
            class={`ar-corner-lbl${q === query.quadrant ? " is-target" : ""}`}
            x={rr.x + rr.w / 2}
            y={q === 2 || q === 3 ? rr.y + 12 : rr.y + rr.h - 5}
            text-anchor="middle"
          >
            {w}
          </text>
        );
      })}

      {/* all 249 scenes as faint background dots (skip the top-5, drawn large below) */}
      {GALLERY.filter((f) => !inTop5.has(f.idx)).map((f) => (
        <circle class="ar-scene" cx={sx(f.tee_x)} cy={sy(f.tee_y)} r={1.5} />
      ))}

      {/* the encoder's top-5 — hit (in the corner) green, miss red; rank inside */}
      {top5.map((i, rank) => {
        const f = BY_IDX.get(i);
        if (!f) return null;
        const hit = f.quadrant === query.quadrant;
        return (
          <g class={`ar-hit ${hit ? "is-hit" : "is-miss"}`}>
            <circle cx={sx(f.tee_x)} cy={sy(f.tee_y)} r={7} />
            <text x={sx(f.tee_x)} y={sy(f.tee_y) + 2.6} text-anchor="middle">{rank + 1}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ===================================================================== THE ISLAND
export default function AlignRetrievalToy() {
  // default-interesting: the "top left" instruction — aligned lights up 5/5 in the
  // corner, random only 1/5. SSR renders at this selection, so it IS the JS-off view.
  const [sel, setSel] = useState(DEFAULT_SEL);
  const [text, setText] = useState("");
  const query = QUERIES[sel];
  const word = QUADRANT_WORDS[query.quadrant];

  const alignedHits = hitsFor(ALIGNED_TOP5[query.quadrant], query.quadrant);
  const randomHits = hitsFor(RANDOM_TOP5[query.quadrant], query.quadrant);

  const announce = useMemo(
    () =>
      `Instruction: "${query.instruction}". Aligned contrastive encoder: ${alignedHits} of its top 5 scenes ` +
      `sit in the ${word} corner — the retrieved scenes cluster there. Random-init encoder: ${randomHits} of 5 ` +
      `— the retrieved scenes are scattered across corners. Contrastive pretraining built a shared image-text ` +
      `space from pairing alone; the untrained encoder did not.`,
    [sel],
  );

  const onText = (e: Event) => {
    const v = (e.target as HTMLInputElement).value;
    setText(v);
    const m = matchQuery(v);
    if (m !== null) setSel(m);
  };
  const matched = matchQuery(text);

  return (
    <div class="ar">
      <header class="ar-head">
        <h3 class="ar-title">Why aligned — one instruction, two encoders</h3>
        <p class="ar-sub">
          Pick an instruction and watch which held-out scenes light up by <b>cosine similarity</b>. The{" "}
          <b>aligned</b> tower (symmetric InfoNCE) returns the scenes that really sit in that corner; the{" "}
          <b>random-init</b> tower returns noise. Same rankings, side by side — that gap is what contrastive
          pretraining bought, from <b>pairing alone, no labels</b>.
        </p>
      </header>

      {/* --- the one control: pick or type an instruction --- */}
      <div class="ar-picker" role="group" aria-label="Choose an instruction">
        <label class="ar-text-lbl" for="ar-text">type an instruction</label>
        <input
          id="ar-text"
          class="ar-text"
          type="text"
          value={text}
          placeholder='e.g. "the block is near the top left corner"'
          autocomplete="off"
          spellcheck={false}
          onInput={onText}
          aria-describedby="ar-text-hint"
        />
        <span id="ar-text-hint" class="ar-text-hint" aria-live="polite">
          {text.trim() === ""
            ? "…or choose one below"
            : matched !== null
              ? `matched → ${QUERIES[matched].instruction}`
              : "no corner word yet — showing the current instruction"}
        </span>
        <div class="ar-chips">
          {QUERIES.map((q, i) => (
            <button
              type="button"
              class="ar-chip"
              aria-pressed={i === sel}
              onClick={() => { setSel(i); setText(""); }}
            >
              {QUADRANT_WORDS[q.quadrant]}
            </button>
          ))}
        </div>
      </div>

      <p class="ar-instruction">
        <span class="ar-inst-k">instruction</span>
        <span class="ar-inst-v">“{query.instruction}”</span>
      </p>

      {/* --- the two labelled result columns, side by side --- */}
      <div class="ar-galleries">
        <figure class="ar-col ar-col--aligned" aria-label={`Aligned encoder retrieval for "${query.instruction}"`}>
          <figcaption class="ar-col-cap">
            <b>Aligned encoder</b><span>contrastive · symmetric InfoNCE</span>
          </figcaption>
          <Gallery kind="aligned" query={query} />
          <div class="ar-readout" data-good={alignedHits >= 3}>
            <b>{alignedHits} / 5</b> top matches in the {word} corner
          </div>
        </figure>

        <figure class="ar-col ar-col--random" aria-label={`Random-init encoder retrieval for "${query.instruction}"`}>
          <figcaption class="ar-col-cap">
            <b>Random-init encoder</b><span>untrained · no shared space</span>
          </figcaption>
          <Gallery kind="random" query={query} />
          <div class="ar-readout" data-good={randomHits >= 3}>
            <b>{randomHits} / 5</b> top matches in the {word} corner
          </div>
        </figure>
      </div>

      <div class="ar-legend">
        <span class="ar-leg ar-leg--hit">in the named corner (a hit)</span>
        <span class="ar-leg ar-leg--miss">wrong corner (a miss)</span>
        <span class="ar-leg ar-leg--scene">the other {N_GALLERY - 5} held-out scenes</span>
      </div>

      {/* non-visual path to the same aha — the qualitative story, not per-dot spam */}
      <div class="bk-sr" aria-live="polite">{announce}</div>

      <p class="ar-note">
        Across all four instructions the aligned tower puts <b>{ALIGNED_HITS} / {TOTAL}</b> of its top matches in
        the named corner; the random tower only <b>{RANDOM_HITS} / {TOTAL}</b> — and it returns the <b>same few
        high-norm scenes</b> for different instructions (frame 189 is in all four of its top-5 lists). The rock is{" "}
        <b>aligned ≫ random</b>: contrastive learned a real shared image-text space from <b>pairing alone</b>, no
        labels. These are tiny towers over near-blank 64×64 sim frames — a mechanism demo, not a strong perception
        model; a web-scale CLIP/SigLIP is the reference. Real precomputed rankings from align.py (seed 0, cpu,{" "}
        {N_GALLERY} held-out scenes); poster reads with JS off.
      </p>
    </div>
  );
}
