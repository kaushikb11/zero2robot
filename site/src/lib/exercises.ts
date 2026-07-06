// Build-time bridge into a chapter's SUGGESTED exercise candidates. Runs at SSG
// time (Node), never in the browser — it shells out to the same Python bridge
// (site/scripts/curriculum_bridge.py) the region/wallclock/meta loaders use, so
// the site reads exercises from curriculum/ with zero duplicated parsing.
//
// The site NEVER executes or auto-grades an exercise: this loader only surfaces
// the prompt, a local run command, and (for predict-then-run) the recorded
// answer + reference metrics that gate a post-prediction reveal. Answers and
// metrics come straight from checks.py / meta.yaml via the bridge — never
// invented here. "Exercises never phone home."
//
// Mirrors the style of lib/curriculum.ts (same PYTHON/BRIDGE resolution + a
// memoized runBridge so one build spawns the bridge once per distinct query).

import { execFileSync } from "node:child_process";
import { resolve } from "node:path";

// The build always runs with cwd = site/; Z2R_* env vars override for CI /
// out-of-tree invocations. Kept in lockstep with lib/curriculum.ts.
const SITE_DIR = process.env.Z2R_SITE_DIR ?? process.cwd();
const REPO_ROOT = process.env.Z2R_REPO_ROOT ?? resolve(SITE_DIR, "..");
const BRIDGE = resolve(SITE_DIR, "scripts", "curriculum_bridge.py");
const PYTHON = process.env.Z2R_PYTHON ?? resolve(REPO_ROOT, ".venv", "bin", "python");

/** One authored exercise archetype. */
export type ExerciseType =
  | "predict-then-run"
  | "bug-hunt"
  | "code-completion"
  | "hyperparameter-investigation"
  | (string & {}); // forward-compatible: an unknown type renders as a plain prompt

/** Reference metrics + provenance a chapter's meta.yaml records for an exercise
 *  (exercise-spec: seeded-run bands with provenance, no bare magic numbers).
 *  Shape is exercise-specific; always carries `provenance` when available. */
export interface ExerciseRefs {
  provenance?: string;
  [key: string]: unknown;
}

/** An exercise as the site surfaces it. Every field beyond id/type/title/prompt
 *  is best-effort and may be null for a half-authored candidate. */
export interface Exercise {
  id: string; // "ex1", "ex2", …
  num: number | null; // 1, 2, … (parsed from id)
  type: ExerciseType;
  title: string; // the exercise's self-describing first docstring line
  prompt: string; // the docstring body as markdown (title + gate lines removed)
  choices: string[] | null; // e.g. ["A","B","C"] for predict-then-run
  gate_before_run: boolean; // METADATA gate flag (the site only *acts* on it for predict-then-run)
  answer: string | null; // the recorded answer letter (predict-then-run only), from checks.py / meta.yaml
  refs: ExerciseRefs | null; // reference metrics + provenance, if the chapter records them
  run_cmd: string; // "run it locally" — the pytest checks command, targeted to this exercise
  file: string; // repo-relative path to the exercise candidate
}

// One Astro build renders every page in a single Node process; memoize by args
// so the bridge is spawned once per distinct chapter, not once per page.
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

/** Every SUGGESTED exercise candidate for a chapter, in ex-number order
 *  (chapterDir relative to repo root). Empty when a chapter has no exercises. */
export function chapterExercises(chapterDirRelPath: string): Exercise[] {
  const data = runBridge(["exercises", "--chapter-dir", chapterDirRelPath]);
  return (data.exercises ?? []) as Exercise[];
}

/** True when an exercise should get the interactive predict-then-run gate: it is
 *  a predict-then-run with a choice set AND a recorded answer to reveal. Any
 *  missing piece degrades to the plain prompt + local-run surface (honest). */
export function isGated(ex: Exercise): boolean {
  return (
    ex.type === "predict-then-run" &&
    Array.isArray(ex.choices) &&
    ex.choices.length > 0 &&
    typeof ex.answer === "string" &&
    ex.answer.length > 0
  );
}
