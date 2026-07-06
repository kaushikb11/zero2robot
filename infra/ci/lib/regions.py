"""Include-by-region marker parsing (site/CLAUDE.md convention).

Contract: chapter artifacts mark site-includable code spans with comment lines

    # --- region: model ---
    ...code...
    # --- endregion ---

Rules enforced here:
- Regions are flat: opening a region inside another region is malformed.
- Region names are unique within a file.
- Every opened region must be closed; a stray endregion is malformed.
- A region's text is the EXACT byte-per-byte content of the lines between the
  markers (markers excluded, line endings preserved) — this is what the site
  build hashes into meta.yaml `region_hashes` and what check_prose_code_drift
  recomputes.
"""

from __future__ import annotations

import hashlib
import re

REGION_OPEN_RE = re.compile(r"^\s*#\s*---\s*region:\s*([A-Za-z0-9_.\-]+)\s*---\s*$")
REGION_CLOSE_RE = re.compile(r"^\s*#\s*---\s*endregion\s*---\s*$")


class RegionError(Exception):
    """Malformed region markers (unclosed / nested / duplicate / stray close)."""


def parse_regions(text: str, source: str = "<string>") -> dict[str, str]:
    """Extract {region_name: exact_text} from source text.

    Raises RegionError on malformed markers; `source` names the file in errors.
    """
    regions: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for lineno, line in enumerate(text.splitlines(keepends=True), start=1):
        open_match = REGION_OPEN_RE.match(line)
        if open_match:
            name = open_match.group(1)
            if current is not None:
                raise RegionError(
                    f"{source}:{lineno}: region '{name}' opened inside open region "
                    f"'{current}' — regions must be flat (site/CLAUDE.md)"
                )
            if name in regions:
                raise RegionError(
                    f"{source}:{lineno}: duplicate region name '{name}' — region "
                    "names must be unique within a file (site/CLAUDE.md)"
                )
            current = name
            buffer = []
            continue
        if REGION_CLOSE_RE.match(line):
            if current is None:
                raise RegionError(
                    f"{source}:{lineno}: endregion with no open region "
                    "(site/CLAUDE.md)"
                )
            regions[current] = "".join(buffer)
            current = None
            continue
        if current is not None:
            buffer.append(line)
    if current is not None:
        raise RegionError(
            f"{source}: region '{current}' is never closed — add "
            "'# --- endregion ---' (site/CLAUDE.md)"
        )
    return regions


def region_sha256(text: str) -> str:
    """sha256 hex digest of a region's exact text (utf-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
