#!/usr/bin/env python3
"""check_loc_caps — LOC cap gate for chapter artifacts.

Contract (curriculum/CLAUDE.md doctrine #1): every chapter artifact is a single
file targeting <=400 total lines; 450 is the HARD cap enforced here.

- Any artifact over 450 lines: FAIL (exit 1), listing violators as
  `path:line-count`.
- Any artifact over 400 lines (but <=450): WARNING to stderr, exit 0.
- Lines are counted as total file lines, comments and blanks included.
- A chapter whose meta.yaml names a missing artifact, or names no artifact at
  all, is a layout violation and fails.

Session-time twin: infra/hooks/pedagogy_gate.py (blocks full-file writes over
the cap); this gate is the authoritative CI re-check on the resulting files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lib.chapters import ChapterError, discover_chapters

HARD_CAP = 450
WARN_AT = 400


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enforce the 450-line hard cap on chapter artifacts "
        "(curriculum/CLAUDE.md doctrine #1)."
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
        print(f"FAIL check_loc_caps: {exc}", file=sys.stderr)
        return 1

    if not chapters:
        print("check_loc_caps: no chapters discovered under curriculum/ — OK")
        return 0

    failures: list[str] = []
    warnings: list[str] = []
    for chapter in chapters:
        artifact = chapter.artifact_path
        if artifact is None:
            failures.append(
                f"{chapter.rel(root)}/meta.yaml: no 'artifact' key "
                "(chapter layout, curriculum/CLAUDE.md)"
            )
            continue
        if not artifact.is_file():
            failures.append(
                f"{chapter.rel(root)}: artifact "
                f"'{chapter.meta.get('artifact')}' named by meta.yaml does not "
                "exist (chapter layout, curriculum/CLAUDE.md)"
            )
            continue
        rel = artifact.relative_to(root).as_posix()
        count = len(artifact.read_text(encoding="utf-8").splitlines())
        if count > HARD_CAP:
            failures.append(f"{rel}:{count}")
        elif count > WARN_AT:
            warnings.append(
                f"WARNING: {rel}:{count} exceeds the {WARN_AT}-line target "
                f"(hard cap {HARD_CAP}) — simplify before it blocks a PR"
            )

    for warning in warnings:
        print(warning, file=sys.stderr)

    if failures:
        print(
            f"FAIL check_loc_caps: {len(failures)} chapter artifact(s) violate "
            f"the {HARD_CAP}-line hard cap or chapter layout "
            "(curriculum/CLAUDE.md doctrine #1). Simplify — splitting the "
            "artifact is not allowed.",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(
        f"check_loc_caps: {len(chapters)} chapter artifact(s) within the "
        f"{HARD_CAP}-line hard cap — OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
