// Build-time bridge into curriculum/. Runs at SSG time (Node), never in the
// browser. Shells out to site/scripts/curriculum_bridge.py so region hashes are
// byte-identical to the CI drift gate (both go through infra/ci/lib/regions.py).
//
// P1 engine: THIS is the region-extraction entry point. Do not reimplement the
// parser in TS — call extractRegions() so the site build and check_prose_code_drift
// can never disagree on a hash.

import { execFileSync } from "node:child_process";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { resolve, join } from "node:path";

// Astro bundles this module into dist/.prerender/chunks at build, so we cannot
// locate the repo from import.meta.url. The build always runs with cwd = site/;
// Z2R_REPO_ROOT overrides for out-of-tree invocations (CI).
const SITE_DIR = process.env.Z2R_SITE_DIR ?? process.cwd();
export const REPO_ROOT = process.env.Z2R_REPO_ROOT ?? resolve(SITE_DIR, "..");
const BRIDGE = resolve(SITE_DIR, "scripts", "curriculum_bridge.py");

// The venv interpreter carries pyyaml + the curriculum imports. Overridable so
// CI can point at its own interpreter.
const PYTHON = process.env.Z2R_PYTHON ?? resolve(REPO_ROOT, ".venv", "bin", "python");

export interface Region {
  name: string;
  text: string; // exact interior bytes, markers excluded, line endings preserved
  sha256: string; // === infra/ci/lib/regions.py region_sha256(text)
}

export interface WallclockRow {
  tier: string;
  minutes: number | null;
  line: string; // curriculum/common/wallclock.py render_line(...)
}

// One Astro build is a single Node process that renders every page, and the nav
// sidebar makes each page re-request all five chapters' meta. Memoize by args so
// the build spawns the python bridge once per distinct query, not once per page.
const bridgeCache = new Map<string, any>();

function runBridge(args: string[]): any {
  const key = JSON.stringify(args);
  const cached = bridgeCache.get(key);
  if (cached !== undefined) return cached;
  const stdout = execFileSync(PYTHON, [BRIDGE, "--root", REPO_ROOT, ...args], {
    encoding: "utf-8",
  });
  const parsed = JSON.parse(stdout);
  bridgeCache.set(key, parsed);
  return parsed;
}

/** Extract every include-by-region span from an artifact (path relative to repo root). */
export function extractRegions(artifactRelPath: string): Region[] {
  const data = runBridge(["regions", "--artifact", artifactRelPath]);
  if (data.error) throw new Error(`region parse failed: ${data.error}`);
  return Object.entries(data.regions).map(([name, v]: [string, any]) => ({
    name,
    text: v.text as string,
    sha256: v.sha256 as string,
  }));
}

/** The { region_name: sha256 } map the site build would write into meta.yaml. */
export function regionHashes(regions: Region[]): Record<string, string> {
  return Object.fromEntries(regions.map((r) => [r.name, r.sha256]));
}

/** Measured wall-clock rows for a chapter, rendered via the one CSV source. */
export function wallclock(chapterId: string, tiers: string[]): WallclockRow[] {
  const data = runBridge(["wallclock", "--chapter", chapterId, "--tiers", ...tiers]);
  return data.rows as WallclockRow[];
}

/** A chapter's "Read the real thing" pointer: after building e.g. ACT from
 *  scratch, this names the ORIGINAL production repo at a PINNED commit + what to
 *  read. Authored in meta.yaml (`rtrt:`), gated behind `read_the_real_thing:
 *  true`, surfaced verbatim by the bridge — the site never invents a repo or
 *  commit. Present only when a chapter declares one; null otherwise. */
export interface RTRT {
  repo: string; // upstream repo, "org/name"
  commit: string; // PINNED sha or tag — never a branch/HEAD (author-supplied)
  url: string; // canonical upstream URL (defaults to github.com/{repo})
  whatToRead: string[]; // author's 3–5 file/function anchors (may be empty)
}

export interface ChapterMeta {
  id: string;
  title: string;
  artifact: string;
  objectives: string[];
  // Additive, optional fields (may be null/undefined for chapters without them).
  // Existing consumers that destructure the four fields above are unaffected.
  demo?: string | null;
  task?: string | null;
  readTheRealThing?: boolean | null; // the gate signal in meta.yaml
  rtrt?: RTRT | null; // the pinned pointer, non-null only when declared
}

/** Chapter meta.yaml fields via the same loader the gates use (chapterDir relative to repo root). */
export function chapterMeta(chapterDirRelPath: string): ChapterMeta {
  return runBridge(["meta", "--chapter-dir", chapterDirRelPath]) as ChapterMeta;
}

// --- chapter discovery: the site engine's map of the curriculum -------------
// The site build owns routing/navigation, so it needs the ORDERED set of
// chapters. We discover them by scanning curriculum/ on disk (Node fs), then
// pull each chapter's canonical fields through chapterMeta() (the bridge) — the
// filesystem gives us structure, the bridge gives us byte-honest content.

/** A chapter as the routing + navigation layer sees it. Extends ChapterMeta
 *  with the derived fields the site needs (slug, phase grouping, artifact
 *  basename for code-panel captions). */
export interface ChapterInfo extends ChapterMeta {
  slug: string; // URL segment, === id (e.g. "ch1.1-bc")
  chapterDir: string; // repo-relative chapter directory
  artifactBasename: string; // e.g. "bc.py" — what the include directive names (meta.artifact is the basename)
  artifactPath: string; // repo-relative artifact path — pass to extractRegions()
  number: string; // "1.1" — for display/breadcrumbs
  phaseKey: string; // "phase1_imitation" — the curriculum subdir
  phaseLabel: string; // "Phase 1 · Imitation" — sidebar heading
}

const CURRICULUM_DIR = "curriculum";

function titleCase(word: string): string {
  return word.charAt(0).toUpperCase() + word.slice(1);
}

/** "phase1_imitation" -> "Phase 1 · Imitation". */
function phaseLabelFor(phaseKey: string): string {
  const m = phaseKey.match(/^phase(\d+)_(.+)$/);
  if (!m) return phaseKey;
  return `Phase ${m[1]} · ${titleCase(m[2])}`;
}

/** "ch1.1-bc" -> "1.1". */
function chapterNumberFor(id: string): string {
  const m = id.match(/^ch([\d.]+)/);
  return m ? m[1] : id;
}

/** Discover every chapter under curriculum/, ordered by phase then chapter
 *  number. Pure structure comes from the filesystem; content comes from the
 *  bridge (chapterMeta). Returns a stable, sorted list the router + sidebar
 *  both consume. */
export function discoverChapters(): ChapterInfo[] {
  const root = resolve(REPO_ROOT, CURRICULUM_DIR);
  const phaseKeys = readdirSync(root, { withFileTypes: true })
    .filter((d) => d.isDirectory() && /^phase\d+_/.test(d.name))
    .map((d) => d.name);

  const chapters: ChapterInfo[] = [];
  for (const phaseKey of phaseKeys) {
    const phaseDir = join(root, phaseKey);
    const chapterDirs = readdirSync(phaseDir, { withFileTypes: true })
      .filter((d) => d.isDirectory() && /^ch\d/.test(d.name))
      .filter((d) => existsSync(join(phaseDir, d.name, "meta.yaml")))
      .map((d) => d.name);

    for (const dir of chapterDirs) {
      const chapterDir = `${CURRICULUM_DIR}/${phaseKey}/${dir}`;
      const meta = chapterMeta(chapterDir);
      const artifactBasename = meta.artifact.split("/").pop() ?? meta.artifact;
      chapters.push({
        ...meta,
        slug: meta.id,
        chapterDir,
        artifactBasename,
        artifactPath: `${chapterDir}/${artifactBasename}`,
        number: chapterNumberFor(meta.id),
        phaseKey,
        phaseLabel: phaseLabelFor(phaseKey),
      });
    }
  }

  // Order by phase index, then by chapter number (numeric, dotted).
  const phaseIndex = (k: string) => parseInt(k.match(/^phase(\d+)/)?.[1] ?? "0", 10);
  const numKey = (n: string) => n.split(".").map((p) => parseInt(p, 10) || 0);
  chapters.sort((a, b) => {
    const pa = phaseIndex(a.phaseKey);
    const pb = phaseIndex(b.phaseKey);
    if (pa !== pb) return pa - pb;
    const [a0, a1] = numKey(a.number);
    const [b0, b1] = numKey(b.number);
    return a0 - b0 || (a1 ?? 0) - (b1 ?? 0);
  });
  return chapters;
}

/** Group an ordered chapter list by phase, preserving order — the sidebar's shape. */
export function chaptersByPhase(
  chapters: ChapterInfo[],
): { phaseKey: string; phaseLabel: string; chapters: ChapterInfo[] }[] {
  const groups: { phaseKey: string; phaseLabel: string; chapters: ChapterInfo[] }[] = [];
  for (const ch of chapters) {
    let g = groups.find((x) => x.phaseKey === ch.phaseKey);
    if (!g) {
      g = { phaseKey: ch.phaseKey, phaseLabel: ch.phaseLabel, chapters: [] };
      groups.push(g);
    }
    g.chapters.push(ch);
  }
  return groups;
}

/** Read a curriculum prose file verbatim (path relative to repo root). The site
 *  stores NO prose of its own — single source of truth stays in curriculum/. */
export function readProse(proseRelPath: string): string {
  return readFileSync(resolve(REPO_ROOT, proseRelPath), "utf-8");
}
