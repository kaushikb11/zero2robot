/**
 * ch1.4 "Generative Policies I: Diffusion" — the MULTIMODALITY concept-toy
 * (`demo: diffusion_multimodality`). The chapter's rock, made visible.
 *
 * A deliberately multimodal target — a unit RING, every direction an equally-good
 * mode, the empty centre the one place NO data sits — and two ways to fit it:
 *   · DIFFUSION (blue): learn to denoise noise into a SAMPLE. It commits to one
 *     mode per draw, so the cloud covers all 8 angular modes and lands ON the ring.
 *   · a same-width MSE REGRESSOR (red): the squared-error optimum is the AVERAGE of
 *     every good action, which for a ring is the dead centre. It mode-collapses.
 * That contrast is the whole idea, and it needs no robot and no WASM — just a 2D
 * scatter of REAL points regenerated from the chapter's diffusion.py at seed 0
 * (site/scripts/vizdata/ch1.4_diffusion.py; provenance in vizdata.json).
 *
 * THE ONE CONTROL is denoising steps (2 / 10 / 100). At 100 the ring is crisp
 * (8/8 modes, radius ~0.87); forced down to 2 it under-denoises — samples get
 * pulled off the ring and a mode drops (7/8, radius ~0.70), the measured Break-It.
 * The regressor never depends on it: red stays collapsed at the centre.
 *
 * Built to the FROZEN CONCEPT-TOY CONTRACT at the top of ../PlateIsland.tsx:
 *   1. SSR render == the JS-off experience. The scatter is pure static SVG from a
 *      build-time-imported JSON; with JS off it is the whole figure (the crisp
 *      100-step contrast), no fetch, no WASM. Controls are inert-but-harmless.
 *   2. No heavy deps to hydrate — the data is tiny text, imported at build.
 *   3. Reuse the real numbers: the points ARE diffusion.py's output; nothing here
 *      is invented (root CLAUDE.md: numbers have provenance).
 *   4. Make the invisible visible: a live "mode coverage" readout (N/8) + the
 *      off-ring pull, both read straight off the regenerated clouds.
 *   5. ONE control (denoising steps), immediate feedback, default-interesting
 *      (boots crisp at 100 so stepping DOWN is the aha), keyboard + aria-live path.
 *   6. Colour discipline: neutral --ink-mute for the target ring (a map, not an
 *      entity), ONE --signal blue for the diffusion samples (the method that
 *      works + the live control), --alert red for the collapsed regressor.
 */
import "./diffusion-ring.css";
import { useEffect, useRef, useState } from "preact/hooks";
import vizRaw from "../../../../curriculum/phase1_imitation/ch1.4_diffusion/demo/vizdata.json";

// ------------------------------------------------------------------- viz data
interface StepSet {
  modes_covered: number;
  mean_radius: number;
  points: number[][]; // [x, y] in ring space (target radius 1)
}
interface VizData {
  target_radius: number;
  n_sectors: number;
  default_steps: number;
  step_counts: number[];
  target: { points: number[][] };
  diffusion: Record<string, StepSet>;
  regression: StepSet;
}

const viz = vizRaw as unknown as VizData;

/** The build-time import can only go missing if the generator was never run or the
 *  JSON was hand-broken; guard so the toy degrades to a captioned placeholder
 *  rather than throwing during hydration. (JS-off already shows the SSR scatter.) */
function dataOk(d: VizData | null | undefined): d is VizData {
  return !!d && !!d.diffusion && !!d.target?.points?.length && !!d.regression?.points?.length
    && Array.isArray(d.step_counts) && d.step_counts.every((s) => !!d.diffusion[String(s)]?.points?.length);
}

// ------------------------------------------------------------------- geometry
const V = 460;                 // square viewBox
const EXTENT = 1.5;            // world half-extent mapped to the view (ring r=1 fits with margin)
const S = V / 2 / EXTENT;      // world -> px scale
const CX = V / 2, CY = V / 2;
const w2s = (x: number, y: number): [number, number] => [CX + x * S, CY - y * S]; // world +y up
const R1 = S; // px radius of the unit ring
const SECTORS = 8;

// ------------------------------------------------------------------- rendering
/** A cloud of dots. `animate` staggers a draw-in (client-only, reduced-motion off). */
function Cloud({ points, cls, r, animate, drawKey }: {
  points: number[][]; cls: string; r: number; animate: boolean; drawKey: number;
}) {
  return (
    <g class={cls} key={drawKey}>
      {points.map((p, i) => {
        const [px, py] = w2s(p[0], p[1]);
        return (
          <circle
            cx={px.toFixed(1)}
            cy={py.toFixed(1)}
            r={r}
            class={animate ? "dr-dot dr-dot--draw" : "dr-dot"}
            style={animate ? `animation-delay:${((i % 60) * 6).toFixed(0)}ms` : undefined}
          />
        );
      })}
    </g>
  );
}

/** The static backdrop: the unit ring guide + the 8 sector spokes (the "modes"). */
function RingGuide() {
  const spokes = Array.from({ length: SECTORS }, (_, k) => {
    // sector boundaries at theta = -pi + 2pi*k/8 (matches ring_stats' binning)
    const a = -Math.PI + (2 * Math.PI * k) / SECTORS;
    const [x2, y2] = w2s(1.32 * Math.cos(a), 1.32 * Math.sin(a));
    return <line class="dr-spoke" x1={CX} y1={CY} x2={x2.toFixed(1)} y2={y2.toFixed(1)} />;
  });
  return (
    <>
      {spokes}
      <circle class="dr-ring" cx={CX} cy={CY} r={R1} />
      <circle class="dr-center" cx={CX} cy={CY} r={3.4} />
    </>
  );
}

// ------------------------------------------------------------------------- toy
function DiffusionRingToy() {
  const ok = dataOk(viz);
  const steps = ok ? viz.step_counts : [100];
  const [stepIdx, setStepIdx] = useState(() => {
    const d = ok ? viz.step_counts.indexOf(viz.default_steps) : 0;
    return d >= 0 ? d : (ok ? viz.step_counts.length - 1 : 0);
  });
  const [mounted, setMounted] = useState(false); // gates client-only entry animation
  const [reduce, setReduce] = useState(false);
  const [drawKey, setDrawKey] = useState(0);
  const groupRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    setReduce(!!mq?.matches);
    setMounted(true);
  }, []);

  if (!ok) {
    return (
      <div class="dr">
        <div class="dr-figure dr-figure--placeholder" role="img"
          aria-label="Diffusion ring toy — the precomputed sample data is unavailable in this build.">
          <p class="dr-placeholder-text">
            ring viz data unavailable — regenerate with
            <code> site/scripts/vizdata/ch1.4_diffusion.py</code>
          </p>
        </div>
      </div>
    );
  }

  const stepVal = steps[stepIdx];
  const diff = viz.diffusion[String(stepVal)];
  const reg = viz.regression;
  const animate = mounted && !reduce;

  const selectStep = (i: number) => {
    const clamped = Math.max(0, Math.min(steps.length - 1, i));
    setStepIdx(clamped);
    setDrawKey((k) => k + 1); // remount the diffusion cloud -> replay the draw-in
    // move focus to the newly-selected option (radiogroup convention)
    const btn = groupRef.current?.querySelectorAll<HTMLButtonElement>(".dr-seg-btn")[clamped];
    btn?.focus();
  };
  const onGroupKey = (e: KeyboardEvent) => {
    if (e.key === "ArrowRight" || e.key === "ArrowDown") { e.preventDefault(); selectStep(stepIdx + 1); }
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") { e.preventDefault(); selectStep(stepIdx - 1); }
    else if (e.key === "Home") { e.preventDefault(); selectStep(0); }
    else if (e.key === "End") { e.preventDefault(); selectStep(steps.length - 1); }
  };

  const crisp = diff.modes_covered >= viz.n_sectors;
  const modeText = `${diff.modes_covered}/${viz.n_sectors}`;

  return (
    <div class="dr">
      <figure class="dr-figure">
        <svg
          class="dr-svg"
          viewBox={`0 0 ${V} ${V}`}
          role="img"
          aria-label={
            "A 2D scatter over a grey unit ring with eight angular sectors. Blue dots are "
            + "diffusion samples; red dots are a same-width squared-error regressor. At the "
            + "default 100 denoising steps the blue dots cover all eight modes and sit on the "
            + "ring, while the red dots collapse into a single blob at the empty centre. Fewer "
            + "denoising steps pull the blue samples off the ring and drop a mode."
          }
        >
          <title>Diffusion vs a regressor on a multimodal ring target</title>
          <desc>
            The ring is a multimodal target: every direction is an equally-good mode and the
            centre is the one place no data sits. Diffusion samples cover all the modes; an
            MSE regressor averages them into the dead centre. Points are real samples
            regenerated from the chapter's diffusion.py at seed 0.
          </desc>

          <rect class="dr-arena" x={1} y={1} width={V - 2} height={V - 2} rx={6} />
          <RingGuide />

          {/* target ring points — the neutral map */}
          <Cloud points={viz.target.points} cls="dr-target" r={1.5} animate={false} drawKey={-1} />
          {/* the collapsed regressor — the failure, constant across steps */}
          <Cloud points={reg.points} cls="dr-regress" r={2.1} animate={false} drawKey={-2} />
          {/* the diffusion samples — the method that works; re-drawn on step change */}
          <Cloud points={diff.points} cls="dr-diffusion" r={2.1} animate={animate} drawKey={drawKey} />

          {/* legend */}
          <g class="dr-legend" transform={`translate(${V - 138} ${V - 58})`}>
            <rect class="dr-legend-bg" x={-10} y={-14} width={140} height={62} rx={5} />
            <circle class="dr-swatch dr-sw-diff" cx={0} cy={0} r={4} />
            <text class="dr-legend-t" x={12} y={4}>diffusion</text>
            <circle class="dr-swatch dr-sw-reg" cx={0} cy={20} r={4} />
            <text class="dr-legend-t" x={12} y={24}>regressor</text>
            <circle class="dr-swatch dr-sw-tgt" cx={0} cy={40} r={4} />
            <text class="dr-legend-t" x={12} y={44}>ring target</text>
          </g>
        </svg>

        {/* live readout — invisible-made-visible: mode coverage + off-ring pull */}
        <div class="dr-hud" aria-hidden="true">
          <div class="dr-hud-row">
            <span class="dr-k">diffusion · modes covered</span>
            <span class={`dr-v ${crisp ? "dr-ok" : "dr-bad"}`}>{modeText} {crisp ? "✓" : "▲"}</span>
          </div>
          <div class="dr-hud-row">
            <span class="dr-k">diffusion · mean radius</span>
            <span class="dr-v">{diff.mean_radius.toFixed(2)}</span>
          </div>
          <div class="dr-hud-row">
            <span class="dr-k">regressor · modes covered</span>
            <span class="dr-v dr-bad">{reg.modes_covered}/{viz.n_sectors} ▲</span>
          </div>
        </div>
      </figure>

      {/* Non-visual path to the same aha (the visual HUD is aria-hidden). */}
      <div class="bk-sr" aria-live="polite">
        {mounted
          ? `${stepVal} denoising steps: diffusion covers ${modeText} modes on the ring `
            + `(mean radius ${diff.mean_radius.toFixed(2)}), while the squared-error regressor `
            + `covers ${reg.modes_covered} of ${viz.n_sectors} and collapses to the empty centre.`
          : ""}
      </div>

      {/* THE one control — denoising steps (radiogroup: click or arrow keys) */}
      <div class="dr-controls">
        <div class="dr-seg-row">
          <span class="dr-seg-label" id="dr-steps-label">denoising steps</span>
          <div
            ref={groupRef}
            class="dr-seg"
            role="radiogroup"
            aria-labelledby="dr-steps-label"
            onKeyDown={onGroupKey}
          >
            {steps.map((s, i) => (
              <button
                type="button"
                class="dr-seg-btn"
                role="radio"
                aria-checked={i === stepIdx}
                tabIndex={i === stepIdx ? 0 : -1}
                data-active={i === stepIdx}
                onClick={() => selectStep(i)}
              >
                {s}
              </button>
            ))}
          </div>
          <button type="button" class="dr-btn" onClick={() => setDrawKey((k) => k + 1)}>
            replay draw
          </button>
        </div>
        <p class="dr-note">
          100 steps → crisp ring (8/8 modes) · 2 steps → under-denoised (a mode drops) ·
          the regressor collapses to the centre no matter what · scatter reads with JS off
        </p>
      </div>
    </div>
  );
}

export default function DiffusionRing() {
  return <DiffusionRingToy />;
}
