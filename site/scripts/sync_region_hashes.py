#!/usr/bin/env python3
"""sync_region_hashes — refresh every chapter's meta.yaml region_hashes.

The site-build step that keeps the prose-code drift gate honest. For each
discovered chapter it recomputes each include-region's sha256 with the CANONICAL
hasher (infra/ci/lib/regions.region_sha256 — the exact function
check_prose_code_drift.py verifies with) and writes the result into meta.yaml via
write_region_hashes.update_meta_text (comment-preserving block update).

Because the hashes come from the same regions.py the gate uses, a chapter whose
code changed and whose page was re-built lands with fresh, matching hashes; a
code edit that skips the build leaves stale hashes and the gate goes RED — the
intended tripwire. Idempotent: re-running with unchanged code rewrites nothing.

Run as the site build's prebuild step (npm run build) or by hand:

  .venv/bin/python site/scripts/sync_region_hashes.py            # write
  .venv/bin/python site/scripts/sync_region_hashes.py --check    # verify only (CI)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# scripts/ lives at <repo>/site/scripts/. The canonical Python modules live under
# infra/ci (lib.regions, lib.chapters) — wire them onto the path, same as the
# bridge does, so hashing can never diverge from the gate.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "infra" / "ci"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # -> write_region_hashes

from lib.chapters import discover_chapters  # noqa: E402
from lib.regions import parse_regions, region_sha256  # noqa: E402
from write_region_hashes import update_meta_text  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify meta.yaml is already in sync; exit 1 if any file would change",
    )
    args = parser.parse_args(argv)

    chapters = discover_chapters(REPO_ROOT)
    stale: list[str] = []
    total_regions = 0
    for chapter in chapters:
        artifact = chapter.artifact_path
        if artifact is None or not artifact.is_file():
            print(f"SKIP {chapter.rel(REPO_ROOT)}: artifact missing", file=sys.stderr)
            continue
        rel = artifact.relative_to(REPO_ROOT).as_posix()
        regions = parse_regions(artifact.read_text(encoding="utf-8"), source=rel)
        region_hashes = {name: region_sha256(body) for name, body in regions.items()}
        total_regions += len(region_hashes)

        existing = chapter.meta_path.read_text(encoding="utf-8")
        updated = update_meta_text(existing, region_hashes)
        if updated == existing:
            continue
        if args.check:
            stale.append(chapter.rel(REPO_ROOT))
            continue
        chapter.meta_path.write_text(updated, encoding="utf-8")
        print(f"synced {chapter.rel(REPO_ROOT)}: {len(region_hashes)} region hash(es)")

    if args.check and stale:
        print(
            "FAIL sync_region_hashes --check: meta.yaml out of sync in "
            f"{len(stale)} chapter(s) — run the site build (npm run build) or "
            "`.venv/bin/python site/scripts/sync_region_hashes.py`:",
            file=sys.stderr,
        )
        for name in stale:
            print(f"  {name}", file=sys.stderr)
        return 1

    verb = "in sync" if args.check else "synced"
    print(
        f"sync_region_hashes: {len(chapters)} chapter(s), "
        f"{total_regions} region hash(es) {verb}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
