"""Look up measured wall-clock times from curriculum/common/wallclock.csv.

Brand promise (style guide rule 6): every wall-clock number a learner sees
was MEASURED on real hardware by the wallclock-bench skill. This module never
guesses — a chapter/tier pair that is absent or still PENDING renders as
"not yet measured", full stop.

CSV schema: chapter,tier,wallclock_min,config_hash,commit,date,status
"""

import csv
from pathlib import Path

# Default CSV location. Tests monkeypatch this module attribute; production
# code never passes csv_path.
WALLCLOCK_CSV = Path(__file__).resolve().parent / "wallclock.csv"


def lookup(chapter_id: str, tier: str, csv_path: Path | None = None) -> float | None:
    """Measured minutes for (chapter_id, tier), or None if PENDING/absent.

    None means "we have not measured this yet" — callers must render that
    honestly (use render_line), never substitute an estimate.
    """
    path = Path(csv_path) if csv_path is not None else WALLCLOCK_CSV
    if not path.exists():
        return None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("chapter") != chapter_id or row.get("tier") != tier:
                continue
            if (row.get("status") or "").strip().upper() == "PENDING":
                return None
            raw = (row.get("wallclock_min") or "").strip()
            if not raw:
                return None
            return float(raw)
    return None


def render_line(chapter_id: str, tier: str, csv_path: Path | None = None) -> str:
    """Human line for banners and prose. Exactly one of two shapes:

    measured:     "expected wall-clock on {tier}: ~{X} min (measured)"
    not measured: "wall-clock on {tier}: not yet measured"
    """
    minutes = lookup(chapter_id, tier, csv_path)
    if minutes is None:
        return f"wall-clock on {tier}: not yet measured"
    return f"expected wall-clock on {tier}: ~{minutes:g} min (measured)"
