/**
 * ch1.7 "Tokens Meet Torques (the data)" — the VLA DATASET BROWSER + LANGUAGE-LEAK
 * concept-toy (`demo: vla_dataset_browser`).
 *
 * Unlike the ch0/ch1.1 toys, this chapter trains NO policy and ships NO onnx — its
 * artifact is a DATASET. So this island is a pure DATA-INSPECTION panel: there is no
 * MuJoCo-WASM, no inference, no lazy heavy import. The data is a small, committed,
 * TEXT-ONLY JSON (curriculum/.../demo/vizdata.json), statically imported, so the WHOLE
 * toy server-renders — with JS off you get the first example + both leak bars, i.e.
 * the entire lesson, and JS only adds prev/next + the clean↔leaked toggle.
 *
 * It teaches two things, one control each:
 *   1. BROWSE the multi-task language+vision dataset. Step through real sampled rows
 *      (one per instruction template, both tasks). Each row shows the three inputs a
 *      VLA conditions on — the instruction (words + token ids), the frozen CNN's 64-D
 *      feature vector (VISION as a bar, NOT a decoded camera image — no frame binaries
 *      exist here), and the 10-D state (numbers) — plus the padded action it maps to.
 *   2. THE LEAK (the chapter's Break-It). Toggle clean↔leaked templates. Clean
 *      templates name the TASK ("push the t block onto the target") and the action is
 *      essentially UN-decodable from words (probe R² 0.006). Leaked templates append
 *      "moving <direction>", naming the MOVE every frame, and the action becomes
 *      linearly decodable from language alone (R² 0.71) — so a policy would learn to
 *      read the answer off the words and ignore its camera. The toggle relabels the
 *      instruction (the appended words light up) AND highlights the jumped R² bar.
 *
 * Provenance: every number is real, from vla_data.py at seed 0 on CPU (default config),
 * run twice (clean + --break leak) by site/scripts/vizdata/ch1.7_vla_data.py, which
 * asserts each scalar matches this chapter's meta.yaml reference_run before writing.
 *
 * a11y: prev/next + toggle are native buttons; the browser region also takes ← → to
 * step and "l" to toggle the leak. An aria-live region announces the current example
 * and the leak state (the visual instrument panels are aria-hidden to avoid spam).
 * Theme-aware: reads the page's design tokens (which flip in dark mode) with fallbacks;
 * the two task hues get a dark-mode brighten in the co-located CSS. Reduced-motion:
 * bar transitions are disabled under prefers-reduced-motion.
 */
import "./VlaBrowserToy.css";
import { useEffect, useRef, useState } from "preact/hooks";
import vizdataRaw from "../../../../curriculum/phase1_imitation/ch1.7_vla_data/demo/vizdata.json";

// ------------------------------------------------------------------- data shape
interface VbExample {
  index: number;
  task: string;
  task_id: number;
  act_dim: number;
  instruction_clean: string;
  instruction_leak: string;
  leak_direction: string;
  tokens_clean: number[];
  tokens_leak: number[];
  action: number[];
  state: number[];
  image_features: number[];
}
interface VbData {
  provenance: string;
  seed: number;
  meta: {
    num_examples: number;
    num_examples_pusht: number;
    num_examples_aloha: number;
    vocab_size: number;
    feature_dim: number;
    max_tokens: number;
    oov_rate: number;
  };
  vocab: string[];
  specials: { pad: number; bos: number; eos: number; unk: number };
  tasks: { id: number; name: string; act_dim: number; count: number; templates: string[] }[];
  examples: VbExample[];
  leak: {
    clean: { r2: number; r2_pusht: number; r2_aloha: number };
    leak: { r2: number; r2_pusht: number; r2_aloha: number };
  };
}
const DATA = vizdataRaw as unknown as VbData;

// ------------------------------------------------------------------- small helpers
const fmt = (v: number, d = 2) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const r2pct = (v: number) => `${(v * 100).toFixed(1)}%`;

/** The token chips for the active instruction: [BOS] words… [EOS], then a faded
 *  "+N padding" note. In leaked mode the "moving <direction>" tokens are flagged so
 *  the learner SEES which words carry the action. */
function Tokens({ ex, leaked }: { ex: VbExample; leaked: boolean }) {
  const ids = leaked ? ex.tokens_leak : ex.tokens_clean;
  const { vocab, specials } = DATA;
  const shown = ids.filter((id) => id !== specials.pad);
  const padCount = ids.length - shown.length;
  const isSpecial = (id: number) =>
    id === specials.bos || id === specials.eos || id === specials.unk;
  const isLeakWord = (id: number) =>
    leaked && (vocab[id] === "moving" || vocab[id] === ex.leak_direction);
  return (
    <div class="vb-tokens">
      {shown.map((id, i) => (
        <span
          key={i}
          class="vb-tok"
          data-special={isSpecial(id)}
          data-leak={isLeakWord(id)}
        >
          <span class="vb-tok-w">{vocab[id]}</span>
          <span class="vb-tok-id">{id}</span>
        </span>
      ))}
      {padCount > 0 && <span class="vb-tok-pad">+{padCount} ⟨pad⟩</span>}
    </div>
  );
}

/** The action vector as a diverging horizontal bar chart (one row per NATIVE dim).
 *  Actions are ~[-1, 1]; we hold a fixed ±1 scale so bars are comparable across rows. */
function ActionBars({ ex }: { ex: VbExample }) {
  return (
    <div class="vb-actbars">
      {ex.action.map((v, i) => {
        const w = clamp(Math.abs(v), 0, 1) * 50; // % of the half-width
        const pos = v >= 0;
        return (
          <div class="vb-actrow" key={i}>
            <span class="vb-actlab">a[{i}]</span>
            <div class="vb-acttrack">
              <span class="vb-actzero" />
              <span
                class="vb-actfill"
                data-pos={pos}
                style={`width:${w}%; ${pos ? "left:50%" : `right:50%`}`}
              />
            </div>
            <span class="vb-actval">{fmt(v)}</span>
          </div>
        );
      })}
    </div>
  );
}

/** The frozen CNN's 64-D feature vector, drawn as a diverging mini-sparkline of bars.
 *  This is what a policy CONDITIONS ON in place of pixels — a fixed nonlinear
 *  projection of the camera, NOT a decoded image (no frame binaries exist in this toy). */
function FeatureBars({ feats }: { feats: number[] }) {
  const maxAbs = Math.max(1e-6, ...feats.map((v) => Math.abs(v)));
  return (
    <div class="vb-feat" role="img" aria-label={`Frozen CNN feature vector, ${feats.length} dimensions, shown as a bar sparkline diverging around zero.`}>
      {feats.map((v, i) => {
        const h = (Math.abs(v) / maxAbs) * 50; // % of half-height
        return (
          <span
            class="vb-featbar"
            key={i}
            data-pos={v >= 0}
            style={`height:${h}%; ${v >= 0 ? "bottom:50%" : "top:50%"}`}
          />
        );
      })}
    </div>
  );
}

// ------------------------------------------------------------------- the island
function VlaBrowser() {
  const N = DATA.examples.length;
  const [idx, setIdx] = useState(0);
  const [leaked, setLeaked] = useState(false);
  const [announce, setAnnounce] = useState("");
  const browserRef = useRef<HTMLDivElement>(null);

  const ex = DATA.examples[idx];
  const clean = DATA.leak.clean;
  const leak = DATA.leak.leak;

  // Announce the current example + leak state for screen readers (the visual panels
  // are aria-hidden). Debounced to a single polite message per change.
  useEffect(() => {
    const instr = leaked ? ex.instruction_leak : ex.instruction_clean;
    const leakLine = leaked
      ? ` Leaked template: the words name the move direction “${ex.leak_direction}”, so the action is decodable from language alone — probe R squared ${leak.r2.toFixed(3)}.`
      : ` Clean template: the words name the task, not the move — the action is essentially undecodable from language, probe R squared ${clean.r2.toFixed(3)}.`;
    setAnnounce(
      `Example ${idx + 1} of ${N}. Task ${ex.task}. Instruction: “${instr}”.` + leakLine,
    );
  }, [idx, leaked]);

  const go = (d: number) => setIdx((i) => (i + d + N) % N);

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === "ArrowRight" || e.key === "ArrowDown") { e.preventDefault(); go(1); }
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") { e.preventDefault(); go(-1); }
    else if (e.key === "l" || e.key === "L") { e.preventDefault(); setLeaked((v) => !v); }
  };

  const instrBase = ex.instruction_clean;

  return (
    <div class="vb">
      {/* ============================= THE EXAMPLE BROWSER ===================== */}
      <div
        class="vb-browser"
        ref={browserRef}
        tabIndex={0}
        role="group"
        aria-label="VLA dataset example browser. Use the previous and next buttons, or the arrow keys while focused here, to step through sampled examples. Press L to toggle leaked templates."
        onKeyDown={onKeyDown}
      >
        <div class="vb-browser-head" aria-hidden="true">
          <span class="vb-count">
            example <b>{idx + 1}</b> / {N}
          </span>
          <span class="vb-badge" data-task={ex.task}>
            {ex.task} · {ex.act_dim}-D action
          </span>
        </div>

        {/* LANGUAGE — the instruction + its token ids */}
        <div class="vb-card" aria-hidden="true">
          <div class="vb-card-k">language · instruction</div>
          <p class="vb-instr">
            <span>{instrBase}</span>
            {leaked && (
              <span class="vb-instr-leak"> moving {ex.leak_direction}</span>
            )}
          </p>
          <div class="vb-card-k vb-card-k2">token ids · fixed length {DATA.meta.max_tokens}</div>
          <Tokens ex={ex} leaked={leaked} />
        </div>

        {/* VISION + STATE — the frozen features (bar) and proprioception (numbers) */}
        <div class="vb-grid2" aria-hidden="true">
          <div class="vb-card">
            <div class="vb-card-k">
              vision · frozen CNN features · {DATA.meta.feature_dim}-D
            </div>
            <FeatureBars feats={ex.image_features} />
            <div class="vb-card-note">a fixed projection of the camera — not a picture</div>
          </div>
          <div class="vb-card">
            <div class="vb-card-k">state · proprioception · {ex.state.length}-D</div>
            <div class="vb-state">
              {ex.state.map((v, i) => (
                <span class="vb-statev" key={i}>{fmt(v)}</span>
              ))}
            </div>
          </div>
        </div>

        {/* ACTION — the target the (words, pixels, numbers) map to */}
        <div class="vb-card" aria-hidden="true">
          <div class="vb-card-k">→ action · {ex.act_dim}-D (native, unpadded)</div>
          <ActionBars ex={ex} />
        </div>
      </div>

      {/* browser controls */}
      <div class="vb-controls">
        <button type="button" class="vb-btn" onClick={() => go(-1)} aria-label="Previous example">
          ← prev
        </button>
        <button type="button" class="vb-btn" onClick={() => go(1)} aria-label="Next example">
          next →
        </button>
        <span class="vb-note">
          {N} real rows · one per instruction template, both tasks · ← → to step
        </span>
      </div>

      {/* ============================= THE LEAK ============================== */}
      <div class="vb-leak">
        <div class="vb-leak-head">
          <div class="vb-leak-title" aria-hidden="true">
            does the instruction leak the action?
          </div>
          {/* THE one toggle — clean ↔ leaked templates (segmented, keyboard-native) */}
          <div class="vb-toggle" role="group" aria-label="Instruction template mode">
            <button
              type="button"
              class="vb-seg"
              data-on={!leaked}
              aria-pressed={!leaked}
              onClick={() => setLeaked(false)}
            >
              clean
            </button>
            <button
              type="button"
              class="vb-seg"
              data-on={leaked}
              aria-pressed={leaked}
              onClick={() => setLeaked(true)}
            >
              leaked
            </button>
          </div>
        </div>

        {/* two bars: action_from_language_r2, clean vs leaked. Both always visible
            (JS-off shows the whole lesson); the active mode's bar is emphasised. */}
        <div class="vb-r2" aria-hidden="true">
          <div class="vb-r2row" data-active={!leaked}>
            <span class="vb-r2lab">clean templates</span>
            <div class="vb-r2track">
              <div class="vb-r2fill vb-r2-clean" style={`width:${clamp(clean.r2, 0, 1) * 100}%`} />
            </div>
            <span class="vb-r2val">R² {clean.r2.toFixed(3)}</span>
          </div>
          <div class="vb-r2row" data-active={leaked}>
            <span class="vb-r2lab">leaked templates</span>
            <div class="vb-r2track">
              <div class="vb-r2fill vb-r2-leak" style={`width:${clamp(leak.r2, 0, 1) * 100}%`} />
            </div>
            <span class="vb-r2val">R² {leak.r2.toFixed(3)}</span>
          </div>
        </div>

        <p class="vb-leak-cap" aria-hidden="true">
          <b>action_from_language_r2</b> — how much of the action a linear probe reads
          from the <em>words alone</em>. Clean templates name the task, so the words
          barely move with the per-frame action ({r2pct(clean.r2)}). Leaked templates
          append the compass direction every frame, so the action is decodable from
          language ({r2pct(leak.r2)}) — <b>a policy trained on the right bar would learn
          to ignore its camera.</b>
        </p>
        <div class="vb-leak-sub" aria-hidden="true">
          per task — pusht {clean.r2_pusht.toFixed(3)} → {leak.r2_pusht.toFixed(3)} ·
          aloha {clean.r2_aloha.toFixed(3)} → {leak.r2_aloha.toFixed(3)}
          <span class="vb-prov"> · seed {DATA.seed}, measured from vla_data.py (matches meta.yaml)</span>
        </div>
      </div>

      {/* the non-visual path: one polite announcement per example / mode change */}
      <div class="bk-sr" aria-live="polite">{announce}</div>
    </div>
  );
}

// ============================================================================
// WIRE ME IN — PlateIsland.tsx (the shared dispatch; do NOT edit here):
//   import VlaBrowserToy from "./toys/VlaBrowserToy";           // with the other toy imports
//   if (demo === "vla_dataset_browser") return <VlaBrowserToy />; // ch1.7 VLA data browser + language-leak
// (No lazy/heavy deps: this toy is a static-JSON data panel, safe to import eagerly
//  like the other toy posters; it fully server-renders for the JS-off path.)
// ============================================================================
export default function VlaBrowserToy() {
  return <VlaBrowser />;
}
