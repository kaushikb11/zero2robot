/**
 * progress.ts — the client-side progress & completion model for the site.
 *
 * ONE source of truth for "how far has this learner got", stored entirely in
 * localStorage (site/CLAUDE.md: progress is localStorage-first, no accounts, ever).
 * No build-time deps, no network — this module is import-safe in Node (SSR) and
 * only ever TOUCHES localStorage from inside functions, guarded by a typeof-window
 * check, so it can be imported at the top of an island without a hydration crash.
 *
 * Honesty rule (root CLAUDE.md invariant, restated for this feature): completion
 * counts ONLY real signals — a chapter the learner actually scrolled through
 * (`read`), and predictions they actually committed. We never infer, never round
 * up, never award completion for merely opening a page.
 *
 * ─── localStorage schema ────────────────────────────────────────────────────
 *   z2r:progress   (JSON)  →  { [slug]: { read: boolean, readAt?: number } }
 *                             the canonical per-chapter read state. One key.
 *   z2r:last       (string)→  the slug of the last chapter visited (for resume).
 *   z2r:read       (JSON)  →  LEGACY. The old inline ChapterLayout Set of read
 *                             slugs (string[]). Migrated into z2r:progress on the
 *                             first read; kept (not deleted) so a transitional
 *                             build that still writes it is picked up next read.
 *
 *   z2r:ex:<slug>:<exId>  (string)  →  OWNED BY FEATURE F1 (exercises). Each key
 *                             is ONE committed prediction: its value is the
 *                             learner's chosen answer string for exercise <exId>
 *                             in chapter <slug>. We only ever READ these keys
 *                             (scan by prefix) to count predictions-made; F1 is
 *                             the sole writer. A present, non-empty value ==
 *                             "prediction committed".
 * ────────────────────────────────────────────────────────────────────────────
 */

export interface ChapterProgress {
  read: boolean;
  readAt?: number; // epoch ms the chapter first crossed the read threshold
}

/** slug → per-chapter progress. Absent slug === untouched. */
export type ProgressState = Record<string, ChapterProgress>;

/** The minimal shape overallCompletion / a dashboard needs per chapter. */
export interface ChapterRef {
  slug: string;
  exerciseIds?: string[];
}

const PROGRESS_KEY = "z2r:progress";
const LAST_KEY = "z2r:last";
const LEGACY_READ_KEY = "z2r:read";
const EX_PREFIX = "z2r:ex:"; // F1 predictions: z2r:ex:<slug>:<exId>
/** Dispatched on window after every same-tab write, so islands re-render without
 *  waiting for the (cross-tab-only) native `storage` event. */
const CHANGE_EVENT = "z2r:progress-change";

function hasWindow(): boolean {
  return typeof window !== "undefined" && typeof localStorage !== "undefined";
}

function emitChange(): void {
  if (!hasWindow()) return;
  try {
    window.dispatchEvent(new CustomEvent(CHANGE_EVENT));
  } catch {
    /* CustomEvent unsupported — subscribers still get cross-tab storage events */
  }
}

/** Write the canonical progress object WITHOUT emitting (used for silent migration). */
function persist(state: ProgressState): void {
  if (!hasWindow()) return;
  try {
    localStorage.setItem(PROGRESS_KEY, JSON.stringify(state));
  } catch {
    /* quota / disabled storage — progress is best-effort, never fatal */
  }
}

/** Read the raw z2r:progress object (no migration, no side effects). */
function loadRaw(): ProgressState {
  if (!hasWindow()) return {};
  try {
    const raw = localStorage.getItem(PROGRESS_KEY);
    const parsed = raw ? (JSON.parse(raw) as ProgressState) : {};
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

/** Fold any legacy z2r:read Set into the new schema. Non-destructive: a slug
 *  already marked read in z2r:progress is never downgraded, and the legacy key
 *  is left in place. Returns whether the state changed (so we persist once). */
function migrateLegacy(state: ProgressState): boolean {
  if (!hasWindow()) return false;
  let changed = false;
  try {
    const raw = localStorage.getItem(LEGACY_READ_KEY);
    if (!raw) return false;
    const legacy = JSON.parse(raw) as unknown;
    if (!Array.isArray(legacy)) return false;
    for (const slug of legacy) {
      if (typeof slug !== "string") continue;
      if (!state[slug]?.read) {
        state[slug] = { read: true, readAt: state[slug]?.readAt };
        changed = true;
      }
    }
  } catch {
    /* malformed legacy value — ignore, nothing to migrate */
  }
  return changed;
}

/** The canonical progress map, with legacy z2r:read merged in. SSR → `{}`. */
export function getProgress(): ProgressState {
  const state = loadRaw();
  if (migrateLegacy(state)) persist(state); // silent: migration is not a user action
  return state;
}

/** Has this chapter been read? SSR-safe (false on the server). */
export function isRead(slug: string): boolean {
  return !!getProgress()[slug]?.read;
}

/** Mark a chapter read (idempotent). No-op on the server / if already read. */
export function markRead(slug: string): void {
  if (!hasWindow()) return;
  const state = getProgress();
  if (state[slug]?.read) return;
  state[slug] = { read: true, readAt: Date.now() };
  persist(state);
  emitChange();
}

/** The exercise ids in <slug> that have a committed F1 prediction. Scans
 *  z2r:ex:<slug>:<exId> keys; a present, non-empty value counts. SSR → []. */
export function predictionsFor(slug: string): string[] {
  if (!hasWindow()) return [];
  const prefix = `${EX_PREFIX}${slug}:`;
  const ids: string[] = [];
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (!key || !key.startsWith(prefix)) continue;
      const val = localStorage.getItem(key);
      if (val != null && val !== "") ids.push(key.slice(prefix.length));
    }
  } catch {
    /* storage disabled — treat as no predictions */
  }
  return ids;
}

/**
 * A chapter's completion in [0, 1], from REAL signals only.
 *  - No exercises           →  1 if read, else 0.
 *  - With exercises         →  read is worth half, predictions-committed the
 *                              other half (fraction of exerciseIds answered).
 * Only exercise ids actually passed in count — a stray z2r:ex key for a removed
 * exercise never inflates the total.
 */
export function chapterCompletion(slug: string, exerciseIds: string[] = []): number {
  const read = isRead(slug) ? 1 : 0;
  if (exerciseIds.length === 0) return read;
  const committed = predictionsFor(slug);
  const answered = exerciseIds.filter((id) => committed.includes(id)).length;
  const predFrac = answered / exerciseIds.length;
  return read * 0.5 + predFrac * 0.5;
}

/** Mean chapter completion across the whole curriculum, in [0, 1]. SSR → 0. */
export function overallCompletion(chapters: ChapterRef[]): number {
  if (chapters.length === 0) return 0;
  const sum = chapters.reduce(
    (acc, c) => acc + chapterCompletion(c.slug, c.exerciseIds ?? []),
    0,
  );
  return sum / chapters.length;
}

/** The slug of the last chapter visited (resume target). SSR → null. */
export function lastVisited(): string | null {
  if (!hasWindow()) return null;
  try {
    return localStorage.getItem(LAST_KEY);
  } catch {
    return null;
  }
}

/** Record the chapter currently being read, for "continue where you left off". */
export function setLastVisited(slug: string): void {
  if (!hasWindow()) return;
  try {
    localStorage.setItem(LAST_KEY, slug);
  } catch {
    /* best-effort */
  }
  emitChange();
}

/**
 * Subscribe to any progress change — same-tab writes (our CustomEvent) AND
 * cross-tab writes (the native storage event, filtered to our keys). Returns an
 * unsubscribe fn. No-op / no-listeners on the server.
 */
export function subscribe(cb: () => void): () => void {
  if (!hasWindow()) return () => {};
  const onStorage = (e: StorageEvent) => {
    if (
      e.key === null || // localStorage.clear()
      e.key === PROGRESS_KEY ||
      e.key === LAST_KEY ||
      e.key === LEGACY_READ_KEY ||
      e.key.startsWith(EX_PREFIX)
    ) {
      cb();
    }
  };
  const onCustom = () => cb();
  window.addEventListener("storage", onStorage);
  window.addEventListener(CHANGE_EVENT, onCustom);
  return () => {
    window.removeEventListener("storage", onStorage);
    window.removeEventListener(CHANGE_EVENT, onCustom);
  };
}
