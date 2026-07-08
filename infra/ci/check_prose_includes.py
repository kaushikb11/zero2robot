#!/usr/bin/env python3
"""check_prose_includes — the shipping-blocker gate for chapter prose includes.

Contract (site/CLAUDE.md; the renderer is site/src/lib/prose.ts): a chapter page
injects its real code panels at every `[include-by-region: <artifact>#<region>]`
directive. The renderer only recognizes the directive when it is wrapped in a
fenced code block EXACTLY as prose.ts INCLUDE_RE requires:

    ```lang
    [include-by-region: artifact.py#region]
    ```

(the language tag is optional; artifact matches `[\\w.]+`, region matches `\\w+`).

A BARE directive — one not wrapped in that exact fence — is NOT matched by the
renderer, so it survives into the page as literal broken text
`[include-by-region: ...]` where a code panel should be. C0's reverify found 45
such bare directives across 11 chapters. This gate fails on any directive not
covered by a fenced INCLUDE_RE match, listing file:line.

It also fails on any `[rtrt]` / `[include-rtrt]` directive: prose.ts has NO
handler for those, so they leak verbatim onto the page exactly like a bare
include.

Scope: the single canonical rendered prose file, prose/chapter.md, of every
discovered chapter (chapters without one are skipped, like the other prose
gates). This is a pure text check — fast, stdlib-only.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from lib.chapters import ChapterError, discover_chapters

# The EXACT fenced form the site renderer accepts (prose.ts INCLUDE_RE). A
# directive covered by one of these spans renders as a real code panel.
INCLUDE_FENCED_RE = re.compile(
    r"```[a-zA-Z0-9]*\n\[include-by-region:\s*[\w.]+#(\w+)\]\n```"
)
# EVERY include directive occurrence, fenced or not. Anything here that is not
# inside an INCLUDE_FENCED_RE span is a bare directive that renders literally.
ANY_INCLUDE_RE = re.compile(r"\[include-by-region:[^\]]*\]")
# rtrt / include-rtrt directives: prose.ts has no handler, so they leak verbatim.
RTRT_RE = re.compile(r"\[\s*(?:include-)?rtrt\b[^\]]*\]", re.IGNORECASE)


def _line_of(text: str, offset: int) -> int:
    """1-based line number of a character offset within text."""
    return text.count("\n", 0, offset) + 1


def scan_prose(text: str, where: str) -> list[str]:
    """Return failure messages for bare includes / rtrt directives in `text`."""
    failures: list[str] = []
    covered = [m.span() for m in INCLUDE_FENCED_RE.finditer(text)]

    def is_covered(pos: int) -> bool:
        return any(start <= pos < end for start, end in covered)

    for m in ANY_INCLUDE_RE.finditer(text):
        if not is_covered(m.start()):
            line = _line_of(text, m.start())
            failures.append(
                f"{where}:{line}: bare include directive '{m.group(0)}' is not "
                "wrapped in a ```fence``` — the site renderer (prose.ts "
                "INCLUDE_RE) leaves it as literal broken text instead of a code "
                "panel (site/CLAUDE.md)"
            )
    for m in RTRT_RE.finditer(text):
        line = _line_of(text, m.start())
        failures.append(
            f"{where}:{line}: '{m.group(0)}' directive has NO handler in "
            "prose.ts — it renders verbatim onto the page (site/CLAUDE.md)"
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail on bare include-by-region directives (unwrapped by a "
        "fence) and unhandled rtrt directives in chapter prose "
        "(site/CLAUDE.md renderer contract)."
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
        print(f"FAIL check_prose_includes: {exc}", file=sys.stderr)
        return 1

    if not chapters:
        print("check_prose_includes: no chapters discovered — OK")
        return 0

    failures: list[str] = []
    checked = 0
    for chapter in chapters:
        prose = chapter.directory / "prose" / "chapter.md"
        if not prose.is_file():
            continue  # no rendered prose to check
        checked += 1
        where = prose.relative_to(root).as_posix()
        failures.extend(scan_prose(prose.read_text(encoding="utf-8"), where))

    if failures:
        print(
            f"FAIL check_prose_includes: {len(failures)} directive(s) would "
            "render as literal broken text on the chapter page "
            "(site/CLAUDE.md):",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(
        f"check_prose_includes: {checked} chapter prose file(s), every "
        "include-by-region directive fenced and no unhandled rtrt — OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
