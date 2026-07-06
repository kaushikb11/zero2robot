#!/usr/bin/env python3
"""write_region_hashes — inject site-build region hashes into a chapter meta.yaml.

This is the write side of the drift contract, run by the site build (in P4,
against the real meta.yaml). It reads the site build's emitted region_hashes
JSON and merges a `region_hashes:` mapping into the target meta.yaml, which
check_prose_code_drift.py then verifies.

The hashes themselves come straight from the JSON payload (produced by the site
build via curriculum_bridge.py -> infra/ci/lib/regions.region_sha256), so they
stay BYTE-IDENTICAL to what the CI drift gate recomputes. This script only owns
where they land in meta.yaml.

meta.yaml carries human-authored comments (objectives narration, exercise-check
provenance) that are part of the product. So rather than round-tripping the whole
file through pyyaml (which strips comments and mangles unicode), we preserve the
existing file verbatim and only manage the `region_hashes:` block: replace it in
place if present, else append it. The block itself is rendered by pyyaml so it is
always well-formed YAML. Idempotent — re-running with the same hashes is a no-op.

  python write_region_hashes.py --hashes site/.build/region_hashes.ch1.1-bc.json \
      --meta curriculum/phase1_imitation/ch1.1_bc/meta.yaml
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

# Matches a top-level `region_hashes:` key and every indented line that belongs
# to it, up to (but not including) the next top-level line or EOF. Used to strip
# a previously-written block before re-appending, so re-runs stay idempotent.
_BLOCK_RE = re.compile(
    r"(?ms)^region_hashes:[^\n]*\n(?:[ \t]+[^\n]*\n?|[ \t]*\n)*"
)


def render_block(region_hashes: dict[str, str]) -> str:
    """Render `region_hashes: {name: sha256}` as a clean top-level YAML block.

    Order and values come straight from the payload; pyyaml guarantees the block
    parses back to exactly this mapping (what the gate loads and verifies).
    """
    dumped = yaml.safe_dump(
        {"region_hashes": region_hashes}, sort_keys=False, default_flow_style=False
    )
    return dumped if dumped.endswith("\n") else dumped + "\n"


def update_meta_text(existing: str, region_hashes: dict[str, str]) -> str:
    """Return meta.yaml text with the region_hashes block set, comments preserved."""
    stripped = _BLOCK_RE.sub("", existing)
    # Normalize trailing whitespace so the appended block sits cleanly at the end
    # with exactly one blank line separating it from prior content.
    stripped = stripped.rstrip("\n")
    block = render_block(region_hashes)
    if not stripped:
        return block
    return f"{stripped}\n\n{block}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hashes", required=True, help="site build region_hashes JSON")
    parser.add_argument("--meta", required=True, help="target meta.yaml to update")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.hashes).read_text(encoding="utf-8"))
    region_hashes = payload["region_hashes"]

    meta_path = Path(args.meta)
    existing = meta_path.read_text(encoding="utf-8")
    updated = update_meta_text(existing, region_hashes)
    meta_path.write_text(updated, encoding="utf-8")
    print(f"wrote {len(region_hashes)} region hash(es) into {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
