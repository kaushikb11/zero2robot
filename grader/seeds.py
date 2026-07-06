"""Seed sources for the scoring harness — and the hidden-seed integration seam.

Anti-overfitting integrity model (grader/CLAUDE.md, leaderboard-spec.md):
  PUBLIC seeds  — used for local dev and for the *practice* / self-declared
                  scoring you can reproduce offline. Fixed, shipped in-repo.
  HIDDEN seeds  — the real leaderboard scoring seeds. They ROTATE at season
                  boundaries and are NEVER co-located with agent-writable code.

  public eval seeds  !=  leaderboard scoring seeds.

Only the PUBLIC source lives here. The hidden source is HUMAN-OWNED and lives
in grader/hidden_seeds/, which does NOT exist in this tree and is hook-denied
to agents — an agent must never create it. `HiddenSeedSource` below is a
DOCUMENTED SEAM ONLY: it declares the interface the human implementation
satisfies and refuses to run, so wiring it up is an explicit human act.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class SeedSource(ABC):
    """A named, ordered, deterministic set of scoring seeds."""

    #: division/label this source scores, e.g. "public" or "s1-hidden".
    name: str

    @abstractmethod
    def seeds(self) -> list[int]:
        """Return the ordered scoring seeds. Deterministic: same call, same
        list, so `submission + seeds -> score` is reproducible."""


class PublicSeedSource(SeedSource):
    """The public PushT scoring seeds. Fixed and offline-reproducible.

    These deliberately live in a DIFFERENT range from the seeds a chapter
    trains/evals on. ch1.1 BC evals on `10_000 + seed + episode`; these public
    scoring seeds start at 900_000 so a submission is never scored on a start
    it was tuned against locally. The hidden set (human-owned) uses yet another
    disjoint range that rotates per season.
    """

    name = "public"
    BASE = 900_000

    def __init__(self, n: int = 50) -> None:
        if n <= 0:
            raise ValueError("need at least one scoring seed")
        self.n = n

    def seeds(self) -> list[int]:
        return [self.BASE + i for i in range(self.n)]


class HiddenSeedSource(SeedSource):
    """SEAM ONLY — the real implementation is HUMAN-OWNED (grader/hidden_seeds/).

    This class exists so the scorer's type contract is complete and so the
    wiring point is obvious in code review. It intentionally raises: an agent
    cannot and must not supply hidden seeds. A human implementation reads the
    rotating season seed file from grader/hidden_seeds/ (hook-denied to agents)
    and returns it here.
    """

    name = "hidden"

    def seeds(self) -> list[int]:  # pragma: no cover - human-owned seam
        raise NotImplementedError(
            "hidden scoring seeds are human-owned and live in "
            "grader/hidden_seeds/ (hook-denied to agents). Wire a human "
            "implementation here at season setup; never generate them in code."
        )
