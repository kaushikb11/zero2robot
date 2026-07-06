/**
 * Achievements — the /progress page's milestone island.
 *
 * Every milestone is derived ONLY from real progress state (../../lib/progress):
 * chapters actually read, gated predictions actually committed, phases fully
 * completed, and overall completion. There are NO fake or vanity badges — if the
 * site cannot observe a real localStorage signal for something, there is no
 * achievement for it. (Notably: the toys never record "you drove a policy", so we
 * do NOT award a "first policy driven" badge — that would be dishonest. The
 * honest analogue is reading the chapter that teaches it, which the read signal
 * already covers.)
 *
 * SSR / hydration contract (mirrors ProgressOverview / CompletionBadge):
 *   · No window/localStorage at module scope or in the INITIAL render. First
 *     render (SSR + first paint) = every milestone LOCKED (computeAchievements
 *     with live=false), so the hydrated DOM matches the SSR'd DOM.
 *   · Real values are read in a post-mount effect and pushed via setState, kept
 *     live through subscribe().
 *   · An aria-live region announces a milestone the moment it UNLOCKS while the
 *     page is open (not the already-earned set on load — that would be noise).
 *   · Each milestone tile is focusable with a descriptive aria-label; the emerald
 *     "unlocked" state is reinforced with a ✓ glyph + text, never colour alone.
 */
import { useEffect, useRef, useState } from "preact/hooks";
import {
  chapterCompletion,
  overallCompletion,
  isRead,
  predictionsFor,
  subscribe,
} from "../../lib/progress";
import "./progress.css";

export interface AchievementChapter {
  slug: string;
  phaseKey: string;
  phaseLabel: string;
  exerciseIds?: string[];
}

interface Props {
  chapters: AchievementChapter[];
}

interface Achievement {
  id: string;
  title: string;
  blurb: string; // how it is earned (always honest)
  glyph: string; // decorative sigil (aria-hidden)
  unlocked: boolean;
}

/** Pure milestone model. `live` gates every localStorage read so the initial
 *  (SSR + first-paint) render is a deterministic all-locked shell. */
function computeAchievements(chapters: AchievementChapter[], live: boolean): Achievement[] {
  // per-chapter real signals
  const read = (slug: string) => (live ? isRead(slug) : false);
  const predictedIn = (ch: AchievementChapter): number => {
    if (!live) return 0;
    const ids = ch.exerciseIds ?? [];
    if (ids.length === 0) return 0;
    const committed = predictionsFor(ch.slug);
    return ids.filter((id) => committed.includes(id)).length;
  };

  const anyRead = live && chapters.some((c) => read(c.slug));
  const anyPrediction = live && chapters.some((c) => predictedIn(c) > 0);
  // a chapter with gated exercises where the learner read it AND answered them all
  const masteredChapters = live
    ? chapters.filter((c) => {
        const total = (c.exerciseIds ?? []).length;
        return total > 0 && read(c.slug) && predictedIn(c) === total;
      }).length
    : 0;
  const overall = live ? overallCompletion(chapters) : 0;

  const base: Achievement[] = [
    {
      id: "first-read",
      title: "Got started",
      blurb: "Read your first chapter end to end.",
      glyph: "▸",
      unlocked: anyRead,
    },
    {
      id: "first-prediction",
      title: "Skin in the game",
      blurb: "Commit your first predict-then-run answer before revealing it.",
      glyph: "◆",
      unlocked: anyPrediction,
    },
    {
      id: "chapter-mastery",
      title: "Full marks",
      blurb: "Read a chapter and answer every one of its predictions.",
      glyph: "★",
      unlocked: masteredChapters > 0,
    },
  ];

  // one milestone per phase — unlocked when every chapter in it is 100% complete
  const phaseMilestones: Achievement[] = [];
  for (const ch of chapters) {
    if (phaseMilestones.some((m) => m.id === `phase-${ch.phaseKey}`)) continue;
    const inPhase = chapters.filter((c) => c.phaseKey === ch.phaseKey);
    const complete =
      live && inPhase.every((c) => chapterCompletion(c.slug, c.exerciseIds ?? []) >= 1);
    phaseMilestones.push({
      id: `phase-${ch.phaseKey}`,
      title: `${ch.phaseLabel} complete`,
      blurb: `Finish every chapter in ${ch.phaseLabel}.`,
      glyph: "❖",
      unlocked: complete,
    });
  }

  const capstones: Achievement[] = [
    {
      id: "halfway",
      title: "Halfway there",
      blurb: "Reach 50% overall completion across the curriculum.",
      glyph: "◐",
      unlocked: overall >= 0.5,
    },
    {
      id: "graduate",
      title: "zero2robot",
      blurb: "Reach 100% — every chapter read, every prediction in.",
      glyph: "◉",
      unlocked: overall >= 1,
    },
  ];

  return [...base, ...phaseMilestones, ...capstones];
}

export default function Achievements({ chapters }: Props) {
  const [items, setItems] = useState<Achievement[]>(() =>
    computeAchievements(chapters, false),
  );
  const [announce, setAnnounce] = useState("");
  // the set of unlocked ids we have already seen, so we only announce NEW unlocks
  // (not the whole already-earned set on the first post-mount sync).
  const seenRef = useRef<Set<string> | null>(null);

  useEffect(() => {
    const update = () => {
      const next = computeAchievements(chapters, true);
      const unlocked = new Set(next.filter((a) => a.unlocked).map((a) => a.id));
      if (seenRef.current === null) {
        // first real read: adopt the baseline silently, no announcement.
        seenRef.current = unlocked;
      } else {
        const fresh = next.filter((a) => a.unlocked && !seenRef.current!.has(a.id));
        if (fresh.length) {
          setAnnounce(
            `Achievement unlocked: ${fresh.map((a) => a.title).join(", ")}.`,
          );
        }
        seenRef.current = unlocked;
      }
      setItems(next);
    };
    update();
    return subscribe(update);
    // chapters is SSR'd once per page load and stable for the island's life.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const earned = items.filter((a) => a.unlocked).length;

  return (
    <section class="pg-ach" aria-label="Achievements">
      <div class="pg-ach-head">
        <h2 class="pg-ach-title">Milestones</h2>
        <span class="pg-ach-count" aria-live="polite">
          {earned}/{items.length} earned
        </span>
      </div>
      <p class="pg-ach-note">
        Every milestone below is earned from real signals only — chapters you read
        and predictions you committed. Nothing here is awarded for just showing up.
      </p>

      {/* polite live region: announces only milestones unlocked while open */}
      <div class="bk-sr" aria-live="polite" role="status">
        {announce}
      </div>

      <ul class="pg-ach-grid">
        {items.map((a) => (
          <li
            class="pg-ach-item"
            key={a.id}
            data-unlocked={a.unlocked}
            tabIndex={0}
            aria-label={`${a.unlocked ? "Unlocked" : "Locked"}: ${a.title}. ${a.blurb}`}
          >
            <span class="pg-ach-glyph" aria-hidden="true">
              {a.glyph}
            </span>
            <span class="pg-ach-body">
              <span class="pg-ach-name">
                {a.title}
                <span class="pg-ach-state" aria-hidden="true">
                  {a.unlocked ? " ✓" : " · locked"}
                </span>
              </span>
              <span class="pg-ach-blurb">{a.blurb}</span>
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
