#!/usr/bin/env python3
"""check_draft_scaffolding — leftover-draft-residue gate.

Contract (curriculum/CLAUDE.md: prose/meta ship as finished, published text):
scaffolding an agent leaves behind while drafting must never survive into a
shipped chapter. C0's editorial reverify removed exactly three kinds of residue
by hand; this gate fails on each so it can never silently ship again. It scans
each chapter's rendered prose (prose/chapter.md) and meta.yaml for:

1. A DRAFT BANNER — an "AGENT DRAFT" / "AGENT-DRAFT" marker or a
   "not publishable" disclaimer. These are the author-handoff banners
   ("AGENT DRAFT — raw material for the author's rewrite; not publishable.").

2. An inline [measured] TAG — a square-bracket provenance placeholder like
   `[measured]` or `[measured 2026-07-04]`. In FINISHED prose the honest form is
   the parenthetical "(measured, t4)" cross-checked by check_wallclock_provenance;
   the square-bracket tag is draft residue that renders verbatim to the reader.

3. A SELF-CONTRADICTORY "intentionally omitted" META COMMENT — a comment claiming
   a key is "intentionally omitted" / "deliberately omitted" while that key is in
   fact present in the file. (A genuine omission — e.g. "# region_hashes
   intentionally omitted until the site build renders this chapter" with no
   region_hashes key — is correct and passes; only the contradiction fails.)

Pure text/stdlib, fast; fails (exit 1) listing every residue as file:line.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from lib.chapters import ChapterError, discover_chapters

BANNER_RE = re.compile(r"AGENT[-\s]?DRAFT|not\s+publishable", re.IGNORECASE)
# Square-bracket provenance placeholder; the legit form uses (parentheses).
MEASURED_TAG_RE = re.compile(r"\[measured\b[^\]]*\]", re.IGNORECASE)
# "<key> [is/are/was/were] intentionally|deliberately omitted" — capture the key.
OMITTED_RE = re.compile(
    r"([A-Za-z_]\w*)\s+(?:is\s+|are\s+|was\s+|were\s+)?"
    r"(?:intentionally|deliberately)\s+omitted",
    re.IGNORECASE,
)


def _key_present(text: str, key: str) -> bool:
    """True if `key:` appears as an actual mapping key (a non-comment line)."""
    key_line = re.compile(r"^\s*" + re.escape(key) + r"\s*:")
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        if key_line.match(line):
            return True
    return False


def scan_banner_and_tags(text: str, where: str) -> list[str]:
    """Draft banners and inline [measured] tags in a prose/meta file."""
    failures: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if BANNER_RE.search(line):
            failures.append(
                f"{where}:{lineno}: draft banner residue "
                f"({line.strip()[:60]!r}) — this ships as published text "
                "(curriculum/CLAUDE.md)"
            )
        for m in MEASURED_TAG_RE.finditer(line):
            failures.append(
                f"{where}:{lineno}: inline '{m.group(0)}' provenance tag renders "
                "verbatim — finished prose uses the '(measured, <tier>)' form "
                "cross-checked by check_wallclock_provenance"
            )
    return failures


def scan_omitted(text: str, where: str) -> list[str]:
    """Self-contradictory 'intentionally omitted' comments (key IS present)."""
    failures: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        comment = line.lstrip()
        if not comment.startswith("#"):
            continue
        for m in OMITTED_RE.finditer(comment):
            key = m.group(1)
            if _key_present(text, key):
                failures.append(
                    f"{where}:{lineno}: comment says '{key}' is intentionally "
                    f"omitted, but '{key}:' IS present — stale draft comment "
                    "(curriculum/CLAUDE.md)"
                )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail on leftover draft scaffolding (AGENT DRAFT banners, "
        "inline [measured] tags, contradictory 'intentionally omitted' "
        "comments) in chapter prose/meta (curriculum/CLAUDE.md)."
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2]),
        help="repo root (default: auto-detected from this script's location)",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    try:
        chapters = discover_chapters(root)
    except ChapterError as exc:
        print(f"FAIL check_draft_scaffolding: {exc}", file=sys.stderr)
        return 1

    if not chapters:
        print("check_draft_scaffolding: no chapters discovered — OK")
        return 0

    failures: list[str] = []
    checked = 0
    for chapter in chapters:
        prose = chapter.directory / "prose" / "chapter.md"
        meta = chapter.meta_path
        if prose.is_file():
            checked += 1
            text = prose.read_text(encoding="utf-8")
            where = prose.relative_to(root).as_posix()
            failures.extend(scan_banner_and_tags(text, where))
            failures.extend(scan_omitted(text, where))
        if meta.is_file():
            text = meta.read_text(encoding="utf-8")
            where = meta.relative_to(root).as_posix()
            failures.extend(scan_banner_and_tags(text, where))
            failures.extend(scan_omitted(text, where))

    if failures:
        print(
            f"FAIL check_draft_scaffolding: {len(failures)} draft residue(s) "
            "left in shipped chapter prose/meta (curriculum/CLAUDE.md):",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(
        f"check_draft_scaffolding: {checked} chapter prose file(s) + "
        f"{len(chapters)} meta.yaml clean of draft residue — OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
