"""Submission schema, reproducible hashing, and percentile bands.

Divisions are a FIRST-CLASS schema concept, not a UI filter (grader/CLAUDE.md):
a submission self-declares its tier. Free-tier claims are spot-checked by
wall-clock plausibility elsewhere (human/ops) — this module only carries the
declaration and a light plausibility hook.

Reproducibility (grader/CLAUDE.md: "Every score is reproducible from the
submission hash"): `submission_hash` is a pure function of the ONNX bytes, the
declared config hash, the division, and the exact scoring-seed set. Same inputs
-> same hash -> (with the deterministic scorer) same score.

Percentile bands before ranks (leaderboard-spec: shown to first-time
submitters): `percentile_band` places a score within a cohort. The cohort
distribution is a data input, not hardcoded — an empty cohort yields
"unranked", which is the honest state before a season has entries.
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass
from pathlib import Path


class Division(str, enum.Enum):
    """The two always-on divisions (leaderboard-spec). Self-declared."""

    FREE = "free"   # completable on T4 / CPU (front-page default)
    OPEN = "open"   # anything: Scale Labs, rented GPUs, ensembles


@dataclass(frozen=True)
class Submission:
    """A leaderboard submission: ONNX policy + config hash + declared tier."""

    onnx_path: Path
    config_hash: str          # hash of the learner's training config (self-declared)
    division: Division
    declared_runtime_min: float | None = None  # for free-tier plausibility

    def submission_hash(self, seeds: list[int]) -> str:
        """Deterministic id binding the artifact, config, division, and the
        exact seed set it was scored on. Disputes re-run from this."""
        h = hashlib.sha256()
        h.update(Path(self.onnx_path).read_bytes())
        h.update(b"\x00")
        h.update(self.config_hash.encode())
        h.update(b"\x00")
        h.update(self.division.value.encode())
        h.update(b"\x00")
        h.update(",".join(str(s) for s in seeds).encode())
        return h.hexdigest()


# Free-tier wall-clock plausibility: a *heuristic* upper bound (minutes) on a
# credible free-tier training run. This is a spot-check hint for ops, NOT a hard
# gate — real enforcement is human review (leaderboard-spec). Tunable.
FREE_TIER_PLAUSIBLE_MAX_MIN = 180.0


def free_tier_declaration_is_plausible(sub: Submission) -> bool | None:
    """None if unknowable (open division or no declared runtime); else whether
    a declared free-tier runtime is within the plausibility bound."""
    if sub.division is not Division.FREE:
        return None
    if sub.declared_runtime_min is None:
        return None
    return 0.0 < sub.declared_runtime_min <= FREE_TIER_PLAUSIBLE_MAX_MIN


# Coarse bands, best-first. Shown BEFORE numeric rank to first-time submitters.
_BANDS = [
    (90.0, "top 10%"),
    (75.0, "top 25%"),
    (50.0, "top half"),
    (25.0, "top 75%"),
    (0.0, "bottom 25%"),
]


def percentile_band(score: float, cohort_scores: list[float]) -> str:
    """Place `score` within `cohort_scores` and return a coarse band label.

    The cohort is a data input (the current division's scored submissions), not
    a baked-in distribution. An empty cohort is honest as "unranked": there is
    nothing to be a percentile of yet.
    """
    if not cohort_scores:
        return "unranked (no cohort yet)"
    at_or_below = sum(1 for s in cohort_scores if s <= score)
    pct = 100.0 * at_or_below / len(cohort_scores)
    for threshold, label in _BANDS:
        if pct >= threshold:
            return label
    return _BANDS[-1][1]
