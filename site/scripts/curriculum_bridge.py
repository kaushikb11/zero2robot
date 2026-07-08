#!/usr/bin/env python3
"""curriculum_bridge — the site build's read-only bridge into curriculum/.

The site is a Node/Astro app; the region-extraction and wall-clock semantics
are Python and MUST stay byte-identical to what the CI drift gate uses. Rather
than porting (and risking hash drift), the Astro build shells out to THIS
script, which imports the canonical modules directly:

  - infra/ci/lib/regions.py        parse_regions + region_sha256  (the gate's own parser)
  - curriculum/common/wallclock.py render_line                    (the one measured-time source)

Everything here is read-only. Output is JSON on stdout.

Usage (from anywhere; paths resolved against --root or auto-detected repo root):
  python curriculum_bridge.py regions   --artifact curriculum/phase1_imitation/ch1.1_bc/bc.py
  python curriculum_bridge.py wallclock --chapter ch1.1-bc --tiers cpu-laptop mps t4 4090
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path


# The bridge's OWN repo (scripts/ lives at <code_root>/site/scripts/). The
# canonical Python modules (regions.py, wallclock.py, chapters.py) always live
# here — imports must resolve from CODE_ROOT, never from --root (which only says
# where DATA/artifacts are read). Keeping these separate means a caller can point
# --root at a scratch tree that has no infra/ and imports still resolve.
CODE_ROOT = Path(__file__).resolve().parents[2]


def _data_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return CODE_ROOT


def _wire_imports() -> None:
    # Reuse the gate's exact parser, chapter loader, and wall-clock renderer.
    sys.path.insert(0, str(CODE_ROOT / "infra" / "ci"))  # -> lib.regions, lib.chapters
    sys.path.insert(0, str(CODE_ROOT))                    # -> curriculum.common.wallclock


def cmd_regions(root: Path, artifact_rel: str) -> int:
    from lib.regions import RegionError, parse_regions, region_sha256

    artifact = (root / artifact_rel).resolve()
    text = artifact.read_text(encoding="utf-8")
    try:
        regions = parse_regions(text, source=artifact_rel)
    except RegionError as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    out = {
        name: {"text": body, "sha256": region_sha256(body)}
        for name, body in regions.items()
    }
    # order: preserve source order for stable rendering
    print(json.dumps({"artifact": artifact_rel, "regions": out}, ensure_ascii=False))
    return 0


def cmd_wallclock(root: Path, chapter_id: str, tiers: list[str]) -> int:
    from curriculum.common.wallclock import lookup, render_line

    rows = []
    for tier in tiers:
        rows.append({
            "tier": tier,
            "minutes": lookup(chapter_id, tier),
            "line": render_line(chapter_id, tier),
        })
    print(json.dumps({"chapter": chapter_id, "rows": rows}, ensure_ascii=False))
    return 0


def _rtrt(m: dict) -> dict | None:
    """The chapter's "Read the real thing" pointer, or None.

    Two honest gates, BOTH required — anything short of both returns None so the
    site renders nothing:
      1. the existing `read_the_real_thing:` boolean is truthy, AND
      2. an `rtrt:` mapping carries at least a non-empty `repo` + `commit`.

    Nothing is synthesized here. `repo`, `commit` (a PINNED sha or tag — the
    author's responsibility, never a branch), `url`, and `what_to_read` are read
    verbatim from meta.yaml; we only default `url` to the GitHub repo URL and
    normalize `what_to_read` to a list. The commit is emitted as-authored so the
    site can link the exact snapshot the prose was written against.

    Authoring schema (add to a chapter's meta.yaml when its RTRT segment exists):

        read_the_real_thing: true
        rtrt:
          repo: "org/name"                 # upstream repo, "org/name"
          commit: "<full-sha-or-tag>"       # PINNED — never a branch/HEAD
          url: "https://github.com/org/name" # optional; defaults to github.com/{repo}
          what_to_read:                      # optional 3-5 anchors (str or list)
            - "path/to/file.py — what production adds that our toy omits"
    """
    if not m.get("read_the_real_thing"):
        return None
    block = m.get("rtrt")
    if not isinstance(block, dict):
        return None
    repo = block.get("repo")
    commit = block.get("commit")
    if not (isinstance(repo, str) and repo and isinstance(commit, str) and commit):
        return None  # a pinned commit is mandatory; never invent one
    url = block.get("url")
    if not (isinstance(url, str) and url):
        url = f"https://github.com/{repo}"
    raw_what = block.get("what_to_read")
    if isinstance(raw_what, str):
        what_to_read = [raw_what]
    elif isinstance(raw_what, list):
        what_to_read = [str(x) for x in raw_what if isinstance(x, str) and x.strip()]
    else:
        what_to_read = []
    return {
        "repo": repo,
        "commit": commit,
        "url": url,
        "whatToRead": what_to_read,
    }


def _track(m: dict) -> str | None:
    """The chapter's curriculum track, or None.

    `track: elective` marks a chapter as OFF the main line (0→1→2→4→5) — optional
    *Depth* the capstone never requires (Phase 3's three self-contained electives
    declare it). Read VERBATIM from meta.yaml, never synthesized: surfaced only
    when the author writes a non-empty string, so a chapter without the key is a
    main-line chapter (null). The site keys its "optional · Depth" treatment off
    this value; it never hard-codes a phase number.
    """
    t = m.get("track")
    return t.strip() if isinstance(t, str) and t.strip() else None


def cmd_meta(root: Path, chapter_rel: str) -> int:
    from lib.chapters import load_chapter

    chapter = load_chapter((root / chapter_rel).resolve())
    m = chapter.meta
    # `demo` / `task` are additive, optional fields the P1 site engine uses for
    # the "See it work" hero island slot and breadcrumb context. `rtrt` (below)
    # is likewise additive — the "Read the real thing" pointer, null until an
    # author declares one. `track` is additive too — the main-line vs optional
    # Depth marker (see _track). None of these affect region hashing or the drift
    # gate; older callers ignore them.
    print(json.dumps({
        "id": chapter.id,
        "title": m.get("title"),
        "artifact": m.get("artifact"),
        "objectives": [_objective_str(o) for o in (m.get("objectives") or [])],
        "demo": m.get("demo"),
        "task": m.get("task"),
        "track": _track(m),
        "readTheRealThing": bool(m.get("read_the_real_thing")),
        "rtrt": _rtrt(m),
    }, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# exercises: surface the chapter's SUGGESTED exercise candidates for the site.
#
# READ-ONLY and PARSE-ONLY. The exercise files import mujoco/torch/lerobot and
# have __main__ runners; we must NEVER execute them, so everything is extracted
# statically with `ast` (docstring + the METADATA dict literal) plus a YAML read
# of the chapter meta.yaml. Every field is best-effort: a missing docstring,
# absent METADATA, unparseable answer key, or missing meta block degrades to
# null and is skipped cleanly — a half-authored exercise must never break the
# site build.
#
# The site never executes or auto-grades: it renders the prompt + a local run
# command, and (for predict-then-run only) gates a reveal behind a recorded
# prediction. Answers/reference metrics are pulled from checks.py / meta.yaml —
# never fabricated here.
# ---------------------------------------------------------------------------

# The four authored exercise archetypes. Type comes from METADATA["type"] when
# present; else it is read off the docstring's self-description first line
# ("... — predict-then-run, ch0.1."); else inferred from the filename token.
_TYPE_FROM_DOCSTRING = re.compile(r"—\s*([a-z][a-z-]+?)\s*,")
_FILENAME_TYPE_HINTS = (
    ("predict", "predict-then-run"),
    ("bughunt", "bug-hunt"),
    ("bug_hunt", "bug-hunt"),
    ("complete", "code-completion"),
    ("completion", "code-completion"),
    ("investigation", "hyperparameter-investigation"),
    ("gain", "hyperparameter-investigation"),
    ("curve", "hyperparameter-investigation"),
)
_EX_ID = re.compile(r"^(ex\d+)")
_CHOICE_LINE = re.compile(r"^\s{1,3}([A-Za-z])\)\s+(.*\S)\s*$")
# The internal predict-then-run gate directive. Strip the CLAUSE (not the whole
# line): some docstrings trail useful runtime context after "run this file"
# (e.g. ch1.1 ex4's "(several minutes on CPU …)"), which must survive.
_RECORD_CLAUSE = re.compile(r"record your answer.*?run this file[.,]?", re.IGNORECASE)


def _literal_assign(tree: ast.AST, name: str):
    """Return the ast.literal_eval'd value of a top-level `name = {...}` assign,
    or None if absent/not a literal. Never executes the module."""
    for node in getattr(tree, "body", []):
        targets = node.targets if isinstance(node, ast.Assign) else []
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                return None
    return None


def _infer_type(metadata: dict | None, docstring: str, filename: str) -> str:
    if isinstance(metadata, dict) and isinstance(metadata.get("type"), str):
        return metadata["type"]
    first = next((ln for ln in docstring.splitlines() if ln.strip()), "")
    m = _TYPE_FROM_DOCSTRING.search(first)
    if m:
        return m.group(1)
    for token, kind in _FILENAME_TYPE_HINTS:
        if token in filename:
            return kind
    return "exercise"


def _clean_prompt(docstring: str) -> str:
    """The docstring as the site prose body: drop the self-describing first line
    (it becomes `title`) and the internal "Record your answer …" gate line, and
    reformat the "  A) …" option lines into a markdown list so they render as
    distinct choices instead of collapsing into one run-on paragraph."""
    lines = docstring.splitlines()
    # drop leading blanks, then the first non-empty line (the title line)
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines):
        i += 1
    body = lines[i:]

    out: list[str] = []
    prev_was_choice = False
    for line in body:
        if _RECORD_CLAUSE.search(line):
            remainder = _RECORD_CLAUSE.sub("", line).strip()
            if not remainder:
                continue  # the whole line was just the gate directive
            line = remainder  # keep trailing runtime context
        choice = _CHOICE_LINE.match(line)
        if choice:
            if not prev_was_choice and (not out or out[-1].strip()):
                out.append("")  # blank line so the list starts a fresh block
            out.append(f"- **{choice.group(1)})** {choice.group(2)}")
            prev_was_choice = True
        else:
            out.append(line)
            prev_was_choice = False
    return "\n".join(out).strip("\n")


def _answer_key_from_checks(suggested_dir: Path) -> dict:
    """The `ANSWER_KEY = {"ex1": "C"}` dict from checks.py (kept out of the
    exercise files to avoid spoilers), or {} if absent/unparseable."""
    checks = suggested_dir / "checks.py"
    if not checks.is_file():
        return {}
    try:
        tree = ast.parse(checks.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}
    value = _literal_assign(tree, "ANSWER_KEY")
    return value if isinstance(value, dict) else {}


def _refs_for(ex_id: str, exercise_checks: dict) -> dict | None:
    """Per-exercise reference metrics + provenance from meta.yaml's
    `exercise_checks:` block. The `answer` sub-key is surfaced separately, so it
    is dropped from refs. Missing per-exercise provenance falls back to the
    block-level provenance."""
    block = exercise_checks.get(ex_id)
    if not isinstance(block, dict):
        return None
    refs = {k: v for k, v in block.items() if k != "answer"}
    if "provenance" not in refs and isinstance(exercise_checks.get("provenance"), str):
        refs["provenance"] = exercise_checks["provenance"]
    return refs or None


def _parse_exercise(path: Path, chapter_rel: str, answer_key: dict,
                    exercise_checks: dict) -> dict | None:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None  # unparseable candidate — skip, never crash the build

    id_match = _EX_ID.match(path.stem)
    if not id_match:
        return None
    ex_id = id_match.group(1)

    docstring = ast.get_docstring(tree) or ""
    title = next((ln.strip() for ln in docstring.splitlines() if ln.strip()), path.stem)
    metadata = _literal_assign(tree, "METADATA")
    metadata = metadata if isinstance(metadata, dict) else None

    ex_type = _infer_type(metadata, docstring, path.stem)
    choices = None
    gate_before_run = False
    if metadata:
        raw_choices = metadata.get("choices")
        if isinstance(raw_choices, list) and all(isinstance(c, str) for c in raw_choices):
            choices = raw_choices
        gate_before_run = bool(metadata.get("gate_before_run", False))

    # Answer only makes sense for the predict-then-run archetype; pull it from
    # checks.py's ANSWER_KEY first, then meta.yaml's per-exercise `answer:`.
    answer = None
    if ex_type == "predict-then-run":
        candidate = answer_key.get(ex_id)
        if candidate is None:
            block = exercise_checks.get(ex_id)
            if isinstance(block, dict):
                candidate = block.get("answer")
        if isinstance(candidate, str):
            answer = candidate

    return {
        "id": ex_id,
        "num": int(ex_id[2:]) if ex_id[2:].isdigit() else None,
        "type": ex_type,
        "title": title,
        "prompt": _clean_prompt(docstring),
        "choices": choices,
        "gate_before_run": gate_before_run,
        "answer": answer,
        "refs": _refs_for(ex_id, exercise_checks),
        "run_cmd": f"pytest {chapter_rel}/exercises/suggested/checks.py -k {ex_id}",
        "file": f"{chapter_rel}/exercises/suggested/{path.name}",
    }


def cmd_exercises(root: Path, chapter_rel: str) -> int:
    from lib.chapters import ChapterError, load_chapter

    chapter_dir = (root / chapter_rel).resolve()
    suggested = chapter_dir / "exercises" / "suggested"

    exercise_checks: dict = {}
    try:
        meta = load_chapter(chapter_dir).meta
        block = meta.get("exercise_checks")
        if isinstance(block, dict):
            exercise_checks = block
    except ChapterError:
        pass  # no/broken meta.yaml — still surface exercises, just without refs

    exercises: list[dict] = []
    if suggested.is_dir():
        answer_key = _answer_key_from_checks(suggested)
        for path in sorted(suggested.glob("ex*.py")):
            parsed = _parse_exercise(path, chapter_rel.rstrip("/"), answer_key, exercise_checks)
            if parsed is not None:
                exercises.append(parsed)

    print(json.dumps({"chapterDir": chapter_rel, "exercises": exercises}, ensure_ascii=False))
    return 0


def _objective_str(o):
    """An objective is a string. But an unquoted YAML objective containing ': '
    (e.g. '...the freedom a task needs: a planar joint') parses as a one-key
    mapping, not a string — reconstruct 'key: value' so the site never renders
    '[object Object]'. Robust to the colon footgun without requiring authors to
    quote every objective."""
    if isinstance(o, str):
        return o
    if isinstance(o, dict) and len(o) == 1:
        (k, v), = o.items()
        return f"{k}: {v}"
    return str(o)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="repo root (default: auto-detect)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_regions = sub.add_parser("regions")
    p_regions.add_argument("--artifact", required=True,
                           help="artifact path relative to repo root")

    p_wall = sub.add_parser("wallclock")
    p_wall.add_argument("--chapter", required=True)
    p_wall.add_argument("--tiers", nargs="+", required=True)

    p_meta = sub.add_parser("meta")
    p_meta.add_argument("--chapter-dir", required=True,
                        help="chapter directory relative to repo root")

    p_ex = sub.add_parser("exercises")
    p_ex.add_argument("--chapter-dir", required=True,
                      help="chapter directory relative to repo root")

    args = parser.parse_args(argv)
    root = _data_root(args.root)
    _wire_imports()

    if args.cmd == "regions":
        return cmd_regions(root, args.artifact)
    if args.cmd == "wallclock":
        return cmd_wallclock(root, args.chapter, args.tiers)
    if args.cmd == "meta":
        return cmd_meta(root, args.chapter_dir)
    if args.cmd == "exercises":
        return cmd_exercises(root, args.chapter_dir)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
