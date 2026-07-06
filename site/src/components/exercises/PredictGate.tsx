/**
 * PredictGate — the one interactive piece of the Practice section: predict-then-run
 * gating for a single predict-then-run exercise.
 *
 * ============================================================================
 * WHAT MAKES THIS HONEST (read before you touch it)
 * ----------------------------------------------------------------------------
 * 1. NO SPOILER PRE-COMMIT.  The answer, the reference metrics, and the run cell
 *    are NEVER rendered into the DOM until the learner commits a prediction.
 *    They arrive as props (so they live in the island's hydration payload, the
 *    Astro-endorsed channel — not as visible/readable text or in the a11y tree),
 *    and the reveal <div> is conditionally mounted only once `committed` is set.
 *    Grep this file: `answer`, `runCmd`, `refs` appear ONLY inside the
 *    `committed && (...)` reveal branch.
 * 2. SSR-SAFE.  No window/document/localStorage at module scope or in the
 *    initial render. The server renders the choices INERT (disabled radios +
 *    disabled commit button) — that is also the JS-off experience. All browser
 *    state (localStorage restore, commit writes) happens inside effects/handlers.
 * 3. THE CONTRACT KEY.  Committing writes the chosen option to localStorage under
 *    exactly `z2r:ex:<chapterSlug>:<exId>` — the key sibling completion tracking
 *    (F2) reads. Restored on mount so a returning learner sees their reveal.
 * 4. ANNOUNCED.  The commit outcome is announced through a visually-hidden
 *    aria-live region (the site's `.bk-sr`), so the reveal is not vision-only.
 * 5. NEVER PHONE HOME.  Nothing is executed or graded here. The reveal shows the
 *    recorded answer + the reference the local checks.py verifies against, and
 *    points the learner at the local run command. That is the whole loop.
 * ============================================================================
 */
import { useEffect, useState } from "preact/hooks";
import type { ExerciseRefs } from "../../lib/exercises.ts";
import "./exercises.css";

interface Props {
  chapterSlug: string;
  exId: string;
  choices: string[];
  answer: string;
  runCmd: string;
  refs?: ExerciseRefs | null;
}

/** The exact key sibling feature F2 (completion tracking) reads. Do not change. */
function storageKey(chapterSlug: string, exId: string): string {
  return `z2r:ex:${chapterSlug}:${exId}`;
}

/** Human labels for the reference metric keys we surface in the reveal. Anything
 *  not listed renders under its raw key — we never hide a recorded number. */
function refEntries(refs: ExerciseRefs | null | undefined): Array<[string, string]> {
  if (!refs) return [];
  return Object.entries(refs)
    .filter(([k]) => k !== "provenance")
    .map(([k, v]) => [k, Array.isArray(v) ? v.join(", ") : String(v)]);
}

export default function PredictGate({
  chapterSlug,
  exId,
  choices,
  answer,
  runCmd,
  refs,
}: Props) {
  // hydrated=false is the SSR / pre-hydration render: choices show, but inert.
  const [hydrated, setHydrated] = useState(false);
  const [chosen, setChosen] = useState<string | null>(null);
  const [committed, setCommitted] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const groupName = `pg-${chapterSlug}-${exId}`;
  const instrId = `${groupName}-instr`;

  // Mount-only: enable interaction and restore any committed prediction. This is
  // the ONLY place browser globals are touched.
  useEffect(() => {
    setHydrated(true);
    try {
      const saved = localStorage.getItem(storageKey(chapterSlug, exId));
      if (saved && choices.includes(saved)) {
        setChosen(saved);
        setCommitted(saved);
      }
    } catch {
      /* storage blocked (private mode) — the gate still works, just no restore */
    }
  }, [chapterSlug, exId]);

  const commit = () => {
    if (!chosen || committed) return;
    setCommitted(chosen);
    try {
      localStorage.setItem(storageKey(chapterSlug, exId), chosen);
    } catch {
      /* storage blocked — reveal still happens this session */
    }
  };

  const predictAgain = () => {
    setCommitted(null);
    setChosen(null);
    setCopied(false);
    try {
      localStorage.removeItem(storageKey(chapterSlug, exId));
    } catch {
      /* ignore */
    }
  };

  const copyCmd = async () => {
    try {
      await navigator.clipboard.writeText(runCmd);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard unavailable — the command is selectable text regardless */
    }
  };

  const isCorrect = committed === answer;

  return (
    <div class="ex-gate" data-committed={committed ? "true" : "false"}>
      {/* visually-hidden announcement of the commit outcome (post-commit only) */}
      <div class="bk-sr" role="status" aria-live="polite">
        {committed
          ? `Prediction committed: ${committed}. The measured answer is ${answer}. ` +
            (isCorrect
              ? "That matches the measurement."
              : "That differs from the measurement — worth re-running to see why.")
          : ""}
      </div>

      {!committed && (
        <fieldset class="ex-gate-fs" aria-describedby={instrId}>
          <legend class="ex-gate-legend">Predict, then commit</legend>
          <p id={instrId} class="ex-gate-instr">
            Pick the outcome you expect from the options above. The answer and the
            local run command reveal only after you commit — predicting after you
            know teaches nothing.
          </p>
          <div class="ex-gate-choices" role="radiogroup" aria-label="Your prediction">
            {choices.map((c) => (
              <label
                key={c}
                class="ex-choice"
                data-selected={chosen === c ? "true" : "false"}
              >
                <input
                  type="radio"
                  name={groupName}
                  value={c}
                  checked={chosen === c}
                  disabled={!hydrated}
                  onChange={() => setChosen(c)}
                />
                <span class="ex-choice-key">{c}</span>
              </label>
            ))}
          </div>
          <button
            type="button"
            class="ex-btn ex-btn--primary"
            disabled={!hydrated || !chosen}
            onClick={commit}
          >
            {hydrated ? "Commit prediction" : "Commit prediction (loads with JavaScript)"}
          </button>
        </fieldset>
      )}

      {committed && (
        <div class="ex-reveal">
          <p class="ex-reveal-verdict" data-correct={isCorrect ? "true" : "false"}>
            <span class="ex-reveal-mark" aria-hidden="true">
              {isCorrect ? "✓" : "→"}
            </span>
            You predicted <b>{committed}</b>. The measured answer is <b>{answer}</b>
            {isCorrect ? " — that matches." : " — worth re-running to see why."}
          </p>

          {refs && (refs.provenance || refEntries(refs).length > 0) && (
            <div class="ex-reveal-refs">
              <p class="ex-reveal-refs-head">
                The local checks verify against this measured reference:
              </p>
              {refEntries(refs).length > 0 && (
                <dl class="ex-refs-dl">
                  {refEntries(refs).map(([k, v]) => (
                    <div key={k} class="ex-refs-row">
                      <dt>{k}</dt>
                      <dd>{v}</dd>
                    </div>
                  ))}
                </dl>
              )}
              {refs.provenance && (
                <p class="ex-refs-prov">provenance · {refs.provenance}</p>
              )}
            </div>
          )}

          <div class="ex-runcell">
            <p class="ex-runcell-label">Now run it locally to see the measurement:</p>
            <div class="ex-cmd-line">
              <code class="ex-cmd">{runCmd}</code>
              <button
                type="button"
                class="ex-cmd-copy"
                aria-label="Copy the run command"
                onClick={copyCmd}
              >
                {copied ? "copied" : "copy"}
              </button>
            </div>
          </div>

          <button type="button" class="ex-link-btn" onClick={predictAgain}>
            predict again
          </button>
        </div>
      )}
    </div>
  );
}
