#!/usr/bin/env python3
"""check_prose_code_drift — region checksum gate (site includes vs chapter code).

Contract (site/CLAUDE.md): the site renders code panels from the actual chapter
artifact via include-by-region markers (`# --- region: name ---` ...
`# --- endregion ---`), never pasted copies. When the site build renders a
chapter, it records `region_hashes: {region_name: sha256hex}` in the chapter's
meta.yaml. This gate recomputes each region's sha256 from the artifact's exact
text and fails on any mismatch (drift) or missing region.

- Chapters without `region_hashes` pass (not yet rendered by the site build).
- Malformed region markers (unclosed / nested / duplicate / stray endregion)
  fail for EVERY chapter, rendered or not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lib.chapters import ChapterError, discover_chapters
from lib.regions import RegionError, parse_regions, region_sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify site include-region checksums against chapter "
        "artifacts (site/CLAUDE.md prose-code drift rule)."
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
        print(f"FAIL check_prose_code_drift: {exc}", file=sys.stderr)
        return 1

    if not chapters:
        print("check_prose_code_drift: no chapters discovered — OK")
        return 0

    failures: list[str] = []
    checked_regions = 0
    for chapter in chapters:
        artifact = chapter.artifact_path
        rel_dir = chapter.rel(root)
        if artifact is None or not artifact.is_file():
            failures.append(
                f"{rel_dir}: artifact '{chapter.meta.get('artifact')}' missing "
                "— cannot verify regions (chapter layout, curriculum/CLAUDE.md)"
            )
            continue
        rel_artifact = artifact.relative_to(root).as_posix()
        try:
            regions = parse_regions(
                artifact.read_text(encoding="utf-8"), source=rel_artifact
            )
        except RegionError as exc:
            failures.append(f"malformed region markers: {exc}")
            continue

        recorded = chapter.meta.get("region_hashes")
        if recorded is None:
            continue  # not yet rendered by the site build — pass
        if not isinstance(recorded, dict):
            failures.append(
                f"{rel_dir}/meta.yaml: region_hashes must be a mapping "
                "{region_name: sha256hex} (site/CLAUDE.md)"
            )
            continue
        for name, expected in sorted(recorded.items()):
            checked_regions += 1
            if name not in regions:
                failures.append(
                    f"{rel_artifact}: region '{name}' recorded in meta.yaml "
                    "region_hashes but not present in the artifact — the site "
                    "include is broken (site/CLAUDE.md)"
                )
                continue
            actual = region_sha256(regions[name])
            if actual != str(expected).strip().lower():
                failures.append(
                    f"{rel_artifact}: region '{name}' drifted from the "
                    f"rendered site include: recorded {expected}, artifact is "
                    f"now {actual} — re-render the chapter page "
                    "(site/CLAUDE.md)"
                )

    if failures:
        print(
            f"FAIL check_prose_code_drift: {len(failures)} problem(s) — code "
            "panels must stay in sync with chapter artifacts "
            "(site/CLAUDE.md):",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(
        f"check_prose_code_drift: {len(chapters)} chapter(s), "
        f"{checked_regions} recorded region hash(es) verified — OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
