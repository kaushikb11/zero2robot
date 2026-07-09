/**
 * QuickstartWinToy — ch0.0 "Lesson 0: Your First Robot Policy in a Few Minutes"
 * concept-toy (demo id `quickstart_first_win`).
 *
 * THE FAST WIN, made honest as a pure DATA-VIEWER over a precomputed rollout — no
 * MuJoCo-WASM, no ONNX, no dynamic import. This is the hero moment of the whole
 * course, shown BEFORE any mechanics: the tiny BC policy the learner just trained
 * in one sitting, driving the PushT T-block from a held-out start onto the target.
 * Unlike ch1.1's LIVE ONNX embed (visitor perturbs the block), this REPLAYS a
 * recorded, REAL rollout transcribed from quickstart.py --seed 0.
 *
 * WHAT IT SHOWS. A top-down replay of the trained policy's WINNING rollout: the
 * amber pusher gets behind the magenta T-block and walks it home; the block's whole
 * recorded path trails behind it; a green target T + the pos_tol success ring sit at
 * the origin. The block starts 0.20 m out and lands at 0.028 m — inside the 0.03 m
 * ring — and the caption flips to "reached ✓". A scrubber (+ play/replay) steps the
 * 115-frame rollout; the SSR default is the LANDED frame, so the JS-off view already
 * shows the win. Below, the honest headline: trained 12/25 vs random 0/25.
 *
 * All numbers + geometry are REAL: scene geometry (tee rects, pusher radius, pos_tol,
 * world extent) and the 115-frame rollout are read verbatim from the co-located
 * demo/vizdata.json, transcribed from quickstart.py's seed-0 run. Nothing is mocked.
 *
 * HONESTY (matches the chapter). The win is real but MODEST. This replays ONE
 * genuine success — the first of the seed-0 policy's 12/25 held-out reaches (48% at
 * this seed; 28–48% across seeds 0–3). The random-action floor is 0/25. The toy never
 * implies the policy is perfect: the headline shows the real fraction, and the note
 * says ch1.1 is the same method done right and reaches higher. A deliberately tiny
 * budget shown as a first taste, not a solved task.
 *
 * Pure inline SVG + design tokens: theme-aware for free (light AND dark), and the
 * server-rendered default (the landed reach + the success readout + the honest
 * headline) IS the JS-off experience. Hydration only adds play/replay + the scrubber.
 *
 * a11y: the scrub slider is a native range (screen readers get frame + pos-err via
 * aria-valuetext); play/prev/next are native buttons; the arena region takes ← → to
 * step and Home/End to jump. An aria-live region announces the win qualitatively
 * (not per-frame spam). Reduced-motion: the play/replay auto-advance is disabled (the
 * button is hidden), manual scrubbing always works; bar/path transitions are cut.
 *
 * Follows the FROZEN CONCEPT-TOY CONTRACT documented at the top of ../PlateIsland.tsx.
 */
import "./QuickstartWinToy.css";
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
// Real recorded seed-0 winning rollout + scene geometry + the held-out headline from
// quickstart.py's reference run — committed small text (numeric frames), no binary.
// Same co-located-vizdata pattern the other data-viewer toys use.
import viz from "../../../../curriculum/phase0_foundations/ch0.0_quickstart/demo/vizdata.json";

// ---------------------------------------------------------------- typed vizdata
interface Scene {
  target_pose: number[];          // [x, y, yaw] — the goal pose (origin)
  tee_centers: number[][];        // per-rect centre in the block frame (metres)
  tee_half_sizes: number[][];     // per-rect half-extent [hx, hy] (metres)
  pusher_radius: number;
  pos_tol: number;                // success ring radius (metres)
  ang_tol: number;
  world_half_extent: number;      // arena spans ±this (metres)
}
interface Headline {
  trained_success_rate: number;   // 0.48 — held-out reach fraction at seed 0
  random_success_rate: number;    // 0.0  — random-action floor
  eval_episodes: number;          // 25
  trained_successes: number;      // 12
  random_successes: number;       // 0
  expert_demos: number;
  expert_successes: number;
}
interface Rollout {
  seed: number;
  success: boolean;
  fps: number;
  frame_schema: string[];         // ["pusher_x","pusher_y","tee_x","tee_y","tee_yaw","pos_err"]
  frames: number[][];             // 115 frames of the winning rollout
}
interface VizData {
  provenance: Record<string, unknown>;
  headline: Headline;
  scene: Scene;
  rollout: Rollout;
}
const DATA = viz as unknown as VizData;
const SCENE = DATA.scene;
const HEAD = DATA.headline;
const ROLL = DATA.rollout;
const FRAMES = ROLL.frames;
const N = FRAMES.length;
const POS_TOL = SCENE.pos_tol;

// frame_schema column indices (read once, not hard-guessed)
const IX = {
  px: ROLL.frame_schema.indexOf("pusher_x"),
  py: ROLL.frame_schema.indexOf("pusher_y"),
  tx: ROLL.frame_schema.indexOf("tee_x"),
  ty: ROLL.frame_schema.indexOf("tee_y"),
  yaw: ROLL.frame_schema.indexOf("tee_yaw"),
  err: ROLL.frame_schema.indexOf("pos_err"),
};

// the first frame the block lands inside the tolerance ring (and holds) — the win.
const SUCCESS_FRAME = (() => {
  const i = FRAMES.findIndex((f) => f[IX.err] < POS_TOL);
  return i < 0 ? N - 1 : i;
})();
const FINAL_ERR = FRAMES[N - 1][IX.err];
const START_ERR = FRAMES[0][IX.err];

// ------------------------------------------------------------------ formatting
const m3 = (v: number) => `${v.toFixed(3)} m`;
const pct = (v: number) => `${Math.round(v * 100)}%`;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const RAD2DEG = 180 / Math.PI;

// ===================================================================== GEOMETRY
// World → SVG px: square viewBox, world +y up (SVG y is down, so subtract). One
// fixed frame for every rollout step, so scrubbing never reflows. SSR renders this
// verbatim — it is the JS-off view.
const V = 360;
const HALF = SCENE.world_half_extent;
const S = V / (2 * HALF);
const w2s = (x: number, y: number): [number, number] => [V / 2 + x * S, V / 2 - y * S];

/** A PushT "T" at world (x,y) rotated by yaw (radians), built from the REAL scene
 *  rects (tee_centers / tee_half_sizes). `cls` picks filled block vs dashed target. */
function Tee({ x, y, yaw, cls, reached }: { x: number; y: number; yaw: number; cls: string; reached?: boolean }) {
  const [cx, cy] = w2s(x, y);
  return (
    <g
      class={cls}
      data-reached={reached ? "true" : undefined}
      transform={`translate(${cx.toFixed(1)} ${cy.toFixed(1)}) rotate(${(-yaw * RAD2DEG).toFixed(1)})`}
    >
      {SCENE.tee_centers.map((c, i) => {
        const [hx, hy] = SCENE.tee_half_sizes[i];
        // world +y is up but this <g> is in SVG space (y down); flip the centre-y so
        // the stem hangs the way the recorded block is oriented.
        return (
          <rect
            key={i}
            x={(c[0] - hx) * S}
            y={-(c[1] + hy) * S}
            width={2 * hx * S}
            height={2 * hy * S}
            rx={2}
          />
        );
      })}
    </g>
  );
}

/** the block's whole recorded path as an SVG polyline point string (up to frame n). */
function trail(n: number): string {
  return FRAMES.slice(0, Math.max(1, n + 1))
    .map((f) => w2s(f[IX.tx], f[IX.ty]).map((v) => v.toFixed(1)).join(","))
    .join(" ");
}

// ==================================================================== THE ISLAND
export default function QuickstartWinToy() {
  // default-interesting: the LAST frame — the block landed inside the ring, reach done.
  // That is the aha, and it is what SSR renders, so the JS-off view already tells the
  // whole story (landed reach + success readout + the honest headline below).
  const [f, setF] = useState(N - 1);
  const [playing, setPlaying] = useState(false);
  const [announce, setAnnounce] = useState("");
  const figRef = useRef<HTMLElement>(null);

  const reduce = useMemo(
    () => typeof window !== "undefined" && window.matchMedia
      && window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    [],
  );

  const fr = FRAMES[clamp(f, 0, N - 1)];
  const posErr = fr[IX.err];
  const reached = f >= SUCCESS_FRAME;      // latched: once landed at frame 111 it holds
  const within = posErr < POS_TOL;
  const [ppx, ppy] = w2s(fr[IX.px], fr[IX.py]);

  // recorded replay: auto-advance from the current frame to the end at ~10 fps
  // (rollout.fps). Skipped entirely under reduced-motion (the button is hidden there);
  // manual scrubbing always works.
  useEffect(() => {
    if (!playing || reduce) return;
    const dt = 1000 / (ROLL.fps || 10);
    let raf = 0, last = performance.now();
    const step = (now: number) => {
      if (now - last >= dt) {
        last = now;
        setF((p) => {
          if (p >= N - 1) { setPlaying(false); return p; }
          return p + 1;
        });
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing, reduce]);

  // announce the qualitative win once it lands / once you scrub back before it — not
  // per-frame spam. The slider's aria-valuetext carries the live per-frame numbers.
  useEffect(() => {
    setAnnounce(
      reached
        ? `The trained policy drove the T-block home: it is ${m3(posErr)} from the target, `
          + `inside the ${POS_TOL} metre success ring — reached. This is one real, held-out success.`
        : `Replaying the trained policy's rollout. The block is ${m3(posErr)} from the target, `
          + `still outside the ${POS_TOL} metre ring.`,
    );
  }, [reached]);

  const togglePlay = () => {
    if (playing) { setPlaying(false); return; }
    if (f >= N - 1) setF(0);   // replay from the top if parked at the end
    setPlaying(true);
  };
  const onScrub = (e: Event) => {
    setPlaying(false);
    setF(parseInt((e.currentTarget as HTMLInputElement).value, 10));
  };
  const onKeyDown = (e: KeyboardEvent) => {
    const k = e.key;
    if (k === "ArrowRight" || k === "]" || k === ".") { e.preventDefault(); setPlaying(false); setF((p) => Math.min(N - 1, p + 1)); }
    else if (k === "ArrowLeft" || k === "[" || k === ",") { e.preventDefault(); setPlaying(false); setF((p) => Math.max(0, p - 1)); }
    else if (k === "Home") { e.preventDefault(); setPlaying(false); setF(0); }
    else if (k === "End") { e.preventDefault(); setPlaying(false); setF(N - 1); }
    else if (k === " " && !reduce) { e.preventDefault(); togglePlay(); }
  };

  const svgLabel =
    `Top-down PushT arena. A green dashed target T and its ${POS_TOL} metre success ring sit at the centre. ` +
    `The trained policy's recorded rollout drives the magenta T-block from ${m3(START_ERR)} out toward the ` +
    `target; a faint magenta trail shows the block's whole path and the amber pusher walks behind it. ` +
    `Currently at frame ${f + 1} of ${N}; the block is ${m3(posErr)} from the target` +
    (within ? `, inside the success ring — reached.` : `, still outside the ring.`);

  // headline bars — trained ≫ random, the real held-out numbers.
  const bars = [
    { key: "trained", label: "trained policy", rate: HEAD.trained_success_rate, n: HEAD.trained_successes, kind: "hero" as const },
    { key: "random", label: "random actions", rate: HEAD.random_success_rate, n: HEAD.random_successes, kind: "base" as const },
  ];

  return (
    <div class="qw">
      <header class="qw-head">
        <h3 class="qw-title">You'll train this, and it works</h3>
        <p class="qw-sub">
          The tiny <b>behavior-cloning</b> policy you train in this chapter, replayed on a{" "}
          <b>held-out start</b>: the amber pusher gets behind the block and walks it home. The block starts{" "}
          <b>{m3(START_ERR)}</b> from the target and lands at <b>{m3(FINAL_ERR)}</b>, inside the{" "}
          <b>{POS_TOL} m</b> ring. One <b>real</b> success, recorded from <code>quickstart.py --seed 0</code>.
        </p>
      </header>

      {/* the top-down replay — SSR renders the landed frame (the JS-off view) */}
      <figure
        class="qw-fig"
        ref={figRef}
        tabIndex={0}
        role="group"
        aria-label="Trained-policy PushT rollout replay. Use the left and right arrow keys to scrub the recorded rollout frame by frame, Home and End to jump to the first or last frame, and Space to play."
        onKeyDown={onKeyDown}
      >
        <svg class="qw-svg" viewBox={`0 0 ${V} ${V}`} role="img" aria-label={svgLabel} data-reached={reached}>
          <title>PushT — trained policy's winning rollout</title>

          {/* arena + faint graph paper */}
          <rect class="qw-arena" x={1} y={1} width={V - 2} height={V - 2} rx={6} />
          <g class="qw-grid">
            {Array.from({ length: 9 }, (_, i) => ((i + 1) * V) / 10).map((v) => (
              <>
                <line x1={v} y1={2} x2={v} y2={V - 2} />
                <line x1={2} y1={v} x2={V - 2} y2={v} />
              </>
            ))}
          </g>

          {/* the block's whole recorded path up to the current frame (the journey) */}
          <polyline class="qw-trail" points={trail(f)} />

          {/* the success ring (pos_tol) + the green dashed target T, both at the origin */}
          <circle class="qw-ring" cx={V / 2} cy={V / 2} r={POS_TOL * S} data-reached={reached} />
          <Tee x={SCENE.target_pose[0]} y={SCENE.target_pose[1]} yaw={SCENE.target_pose[2]} cls="qw-target" />
          <text class="qw-lab qw-lab-target" x={w2s(0.02, 0.05)[0]} y={w2s(0.02, 0.05)[1]}>target</text>

          {/* the block (tee_x, tee_y, tee_yaw) at the current frame */}
          <Tee x={fr[IX.tx]} y={fr[IX.ty]} yaw={fr[IX.yaw]} cls="qw-tee" reached={reached} />

          {/* the pusher (pusher_x, pusher_y) — the amber agent */}
          <g transform={`translate(${ppx.toFixed(1)} ${ppy.toFixed(1)})`}>
            <circle class="qw-pusher-ring" r={SCENE.pusher_radius * 1.9 * S} />
            <circle class="qw-pusher" r={SCENE.pusher_radius * S} />
          </g>
        </svg>

        <figcaption class="qw-cap" aria-hidden="true">
          frame {f + 1}/{N} · block → target{" "}
          <b data-within={within}>{m3(posErr)}</b> vs {POS_TOL} m ring{" "}
          {reached ? <b class="qw-ok">reached ✓</b> : <span class="qw-pending">closing in…</span>}
        </figcaption>
      </figure>

      {/* --- controls: play/replay + frame scrubber (keyboard-accessible) --- */}
      <div class="qw-controls">
        {!reduce && (
          <button
            type="button"
            class="qw-play"
            aria-pressed={playing}
            aria-label={playing ? "Pause the replay" : f >= N - 1 ? "Replay the rollout from the start" : "Play the rollout"}
            onClick={togglePlay}
          >
            {playing ? "❚❚ pause" : f >= N - 1 ? "↺ replay" : "▶ play"}
          </button>
        )}
        <div class="qw-scrub">
          <label class="qw-scrub-lbl" for="qw-frame">frame</label>
          <input
            id="qw-frame"
            class="qw-slider"
            type="range"
            min={0}
            max={N - 1}
            step={1}
            value={clamp(f, 0, N - 1)}
            onInput={onScrub}
            aria-valuetext={
              `frame ${f + 1} of ${N}, block ${m3(posErr)} from the target, `
              + `${within ? "inside" : "outside"} the ${POS_TOL} metre success ring`
            }
          />
          <output class="qw-scrub-out" for="qw-frame">{f + 1}/{N}</output>
        </div>
        <span class="qw-control-note">
          {reduce ? "scrub the recorded rollout" : "play or scrub the recorded rollout"} · poster reads with JS off
        </span>
      </div>

      {/* THE HEADLINE — the honest held-out numbers: trained ≫ random */}
      <figure class="qw-rates">
        <figcaption class="qw-rates-cap">
          held-out success · {HEAD.eval_episodes} random starts the policy never trained on
        </figcaption>
        <div class="qw-bars">
          {bars.map((b) => (
            <div class="qw-bar-row" data-kind={b.kind}>
              <span class="qw-bar-name">{b.label}</span>
              <div class="qw-bar-track">
                <div class="qw-bar-fill" data-kind={b.kind} style={`width:${Math.max(b.rate * 100, b.rate > 0 ? 2 : 0)}%`} />
              </div>
              <span class="qw-bar-val" data-kind={b.kind}>{b.n}/{HEAD.eval_episodes} · {pct(b.rate)}</span>
            </div>
          ))}
        </div>
        <p class="qw-rates-note">
          A <b>modest, real</b> win: this seed reaches <b>{pct(HEAD.trained_success_rate)}</b> of held-out starts
          ({HEAD.trained_successes} of {HEAD.eval_episodes}; 28–48% across seeds 0–3), against a{" "}
          <b>{pct(HEAD.random_success_rate)}</b> random floor. The replay above is <b>one</b> of those{" "}
          {HEAD.trained_successes} successes — not a solved task. <b>ch1.1</b> is the same method done right and
          reaches higher.
        </p>
      </figure>

      {/* non-visual path to the same aha — the qualitative win, not per-frame spam */}
      <div class="bk-sr" aria-live="polite">{announce}</div>

      {/* the honest framing note — straight from the chapter */}
      <p class="qw-note">
        Real recorded rollout from <code>quickstart.py --seed 0</code> (300 expert demos, 600 epochs, cpu). A
        deliberately <b>tiny budget</b> shown <b>before</b> the mechanics — the fast taste of the whole loop, not the
        best you can do. Scene geometry, the {N}-frame path, and the {HEAD.trained_successes}/{HEAD.eval_episodes} vs{" "}
        {HEAD.random_successes}/{HEAD.eval_episodes} headline are all read from the chapter's committed vizdata;
        poster reads with JS off.
      </p>
    </div>
  );
}
