#!/usr/bin/env python3
"""check_wallclock_provenance — wallclock.csv provenance gate.

Contract (root CLAUDE.md: "Wall-clock claims in prose must come from
curriculum/common/wallclock.csv, never estimated"; schema per the shared repo
contract): chapter,tier,wallclock_min,config_hash,commit,date,status.

Validated here:
- Header is exactly the 7 columns above; every row has exactly 7 fields.
- status is PENDING or MEASURED.
- PENDING rows have EMPTY wallclock_min/config_hash/commit/date (a pending row
  may not carry invented numbers).
- MEASURED rows have all four non-empty; wallclock_min is a positive float;
  date is ISO YYYY-MM-DD.
- Every chapter with a meta.yaml has >=1 row (fail on missing — a chapter must
  ship with a wall-clock entry, curriculum/CLAUDE.md definition of done).
- A row whose chapter id matches no existing meta.yaml is a WARNING, not a
  failure: the CSV may legitimately lead the chapter by one PR while chapters
  are scaffolded. Once the chapter lands, id mismatches surface as the missing-
  row failure above.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import re
import sys
from pathlib import Path

from lib.chapters import ChapterError, discover_chapters

EXPECTED_HEADER = [
    "chapter",
    "tier",
    "wallclock_min",
    "config_hash",
    "commit",
    "date",
    "status",
]
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_measured_row(row: dict[str, str], where: str) -> list[str]:
    """Rule checks for a MEASURED row; returns failure messages."""
    failures = []
    empty = [
        col
        for col in ("wallclock_min", "config_hash", "commit", "date")
        if not row[col].strip()
    ]
    if empty:
        failures.append(
            f"{where}: MEASURED row missing {', '.join(empty)} — measured "
            "entries must carry full provenance (ci-lanes.md)"
        )
    minutes = row["wallclock_min"].strip()
    if minutes:
        try:
            value = float(minutes)
        except ValueError:
            value = None
        if value is None or value <= 0:
            failures.append(
                f"{where}: wallclock_min '{minutes}' is not a positive float"
            )
    date = row["date"].strip()
    if date:
        valid = bool(DATE_RE.match(date))
        if valid:
            try:
                datetime.date.fromisoformat(date)
            except ValueError:
                valid = False
        if not valid:
            failures.append(
                f"{where}: date '{date}' is not a valid ISO YYYY-MM-DD date"
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate curriculum/common/wallclock.csv schema and "
        "provenance (root CLAUDE.md wall-clock rule, ci-lanes.md)."
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2]),
        help="repo root (default: auto-detected from this script's location)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="wallclock CSV path (default: {root}/curriculum/common/wallclock.csv)",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    csv_path = Path(args.csv) if args.csv else root / "curriculum/common/wallclock.csv"

    if not csv_path.is_file():
        print(
            f"FAIL check_wallclock_provenance: {csv_path} does not exist — "
            "the wall-clock ledger is mandatory (root CLAUDE.md)",
            file=sys.stderr,
        )
        return 1

    try:
        chapters = discover_chapters(root)
    except ChapterError as exc:
        print(f"FAIL check_wallclock_provenance: {exc}", file=sys.stderr)
        return 1
    chapter_ids = {chapter.id for chapter in chapters}

    failures: list[str] = []
    warnings: list[str] = []
    csv_ids: set[str] = set()

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows or rows[0] != EXPECTED_HEADER:
        print(
            f"FAIL check_wallclock_provenance: {csv_path} header must be "
            f"exactly {','.join(EXPECTED_HEADER)} "
            f"(got: {','.join(rows[0]) if rows else '<empty file>'})",
            file=sys.stderr,
        )
        return 1

    for lineno, fields in enumerate(rows[1:], start=2):
        if not fields or (len(fields) == 1 and not fields[0].strip()):
            continue  # tolerate blank lines
        where = f"{csv_path.name}:{lineno}"
        if len(fields) != len(EXPECTED_HEADER):
            failures.append(
                f"{where}: expected {len(EXPECTED_HEADER)} fields, got "
                f"{len(fields)}"
            )
            continue
        row = dict(zip(EXPECTED_HEADER, fields))
        chapter_id = row["chapter"].strip()
        csv_ids.add(chapter_id)
        if chapter_id not in chapter_ids:
            warnings.append(
                f"WARNING {where}: chapter '{chapter_id}' has no meta.yaml yet "
                "— OK only if the chapter lands in a following PR"
            )
        status = row["status"].strip()
        if status == "PENDING":
            filled = [
                col
                for col in ("wallclock_min", "config_hash", "commit", "date")
                if row[col].strip()
            ]
            if filled:
                failures.append(
                    f"{where}: PENDING row must have empty "
                    f"wallclock_min/config_hash/commit/date, but has "
                    f"{', '.join(filled)} — pending rows carry no numbers "
                    "(root CLAUDE.md: never estimated)"
                )
        elif status == "MEASURED":
            failures.extend(validate_measured_row(row, where))
        else:
            failures.append(
                f"{where}: status '{status}' is not PENDING or MEASURED"
            )

    for chapter in sorted(chapters, key=lambda c: c.id):
        if chapter.id not in csv_ids:
            failures.append(
                f"chapter '{chapter.id}' ({chapter.rel(root)}) has no row in "
                f"{csv_path.name} — every chapter needs a wall-clock entry, "
                "even PENDING (curriculum/CLAUDE.md definition of done)"
            )

    for warning in warnings:
        print(warning, file=sys.stderr)

    if failures:
        print(
            f"FAIL check_wallclock_provenance: {len(failures)} problem(s) in "
            f"{csv_path}:",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(
        f"check_wallclock_provenance: {len(csv_ids)} chapter id(s) across "
        f"{len(rows) - 1} row(s) valid — OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
