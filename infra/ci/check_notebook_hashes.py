#!/usr/bin/env python3
"""check_notebook_hashes — notebook <-> chapter artifact hash-sync gate.

Contract (notebooks/CLAUDE.md): "A notebook and its chapter file share a
content hash recorded in meta.yaml; drift fails CI." Concretely:

- A chapter that has been notebook-generated carries `notebook_hash: <sha256>`
  in its meta.yaml, recorded by the notebook-tier-test skill as the sha256 of
  the chapter artifact file AT GENERATION TIME.
- This gate recomputes sha256 of the artifact and fails on mismatch (the
  notebook is stale — regenerate it; never hand-edit notebooks/).
- It also fails if notebooks/<chapter_id>.ipynb is missing while the hash is
  recorded.
- Additionally, the skill records `notebook_file_hash: <sha256>` — the sha256 of
  the GENERATED NOTEBOOK BYTES themselves. This gate recomputes sha256 of
  notebooks/<chapter_id>.ipynb and fails on mismatch: a mismatch means the
  notebook was hand-edited (which notebook_hash alone cannot catch, since it
  only hashes the chapter artifact).
- Chapters with neither `notebook_hash` nor `notebook_file_hash` are skipped
  with a note (not yet generated).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from lib.chapters import ChapterError, discover_chapters


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify notebook<->chapter content-hash sync "
        "(notebooks/CLAUDE.md)."
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
        print(f"FAIL check_notebook_hashes: {exc}", file=sys.stderr)
        return 1

    if not chapters:
        print("check_notebook_hashes: no chapters discovered — OK")
        return 0

    failures: list[str] = []
    checked = 0
    skipped = 0
    for chapter in chapters:
        recorded = chapter.meta.get("notebook_hash")
        recorded_nb = chapter.meta.get("notebook_file_hash")
        if recorded is None and recorded_nb is None:
            skipped += 1
            print(
                f"note: {chapter.id}: no notebook_hash/notebook_file_hash in "
                "meta.yaml — notebook not yet generated, skipping"
            )
            continue
        checked += 1
        notebook = root / "notebooks" / f"{chapter.id}.ipynb"

        # (1) notebook_hash: artifact-vs-notebook staleness (regenerate needed).
        if recorded is not None:
            artifact = chapter.artifact_path
            if artifact is None or not artifact.is_file():
                failures.append(
                    f"{chapter.rel(root)}: artifact "
                    f"'{chapter.meta.get('artifact')}' missing — cannot verify "
                    "notebook_hash (chapter layout, curriculum/CLAUDE.md)"
                )
            else:
                if not notebook.is_file():
                    failures.append(
                        f"{chapter.id}: meta.yaml records notebook_hash but "
                        f"notebooks/{chapter.id}.ipynb does not exist — "
                        "regenerate via the notebook-tier-test skill "
                        "(notebooks/CLAUDE.md)"
                    )
                actual = sha256_file(artifact)
                if actual != str(recorded).strip().lower():
                    failures.append(
                        f"{chapter.id}: artifact "
                        f"{artifact.relative_to(root).as_posix()} hash {actual} "
                        f"!= meta.yaml notebook_hash {recorded} — the notebook "
                        "is STALE; regenerate via the notebook-tier-test skill, "
                        "never hand-edit (notebooks/CLAUDE.md)"
                    )

        # (2) notebook_file_hash: the notebook bytes themselves must be
        # untouched since generation. A mismatch = a HAND-EDITED notebook.
        if recorded_nb is not None:
            if not notebook.is_file():
                failures.append(
                    f"{chapter.id}: meta.yaml records notebook_file_hash but "
                    f"notebooks/{chapter.id}.ipynb does not exist — regenerate "
                    "via the notebook-tier-test skill (notebooks/CLAUDE.md)"
                )
            else:
                actual_nb = sha256_file(notebook)
                if actual_nb != str(recorded_nb).strip().lower():
                    failures.append(
                        f"{chapter.id}: notebook "
                        f"notebooks/{chapter.id}.ipynb bytes hash {actual_nb} "
                        f"!= meta.yaml notebook_file_hash {recorded_nb} — the "
                        "notebook was HAND-EDITED; regenerate via the "
                        "notebook-tier-test skill, never hand-edit "
                        "(notebooks/CLAUDE.md)"
                    )

    if failures:
        print(
            f"FAIL check_notebook_hashes: {len(failures)} notebook sync "
            "problem(s) (notebooks/CLAUDE.md):",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(
        f"check_notebook_hashes: {checked} notebook(s) in sync, "
        f"{skipped} chapter(s) not yet generated — OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
