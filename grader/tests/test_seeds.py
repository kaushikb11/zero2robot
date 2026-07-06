"""Public seed source is deterministic; the hidden seam refuses to run."""

from __future__ import annotations

import pytest

from grader.seeds import HiddenSeedSource, PublicSeedSource


def test_public_seeds_deterministic():
    assert PublicSeedSource(n=8).seeds() == PublicSeedSource(n=8).seeds()
    assert PublicSeedSource(n=3).seeds() == [900_000, 900_001, 900_002]
    assert PublicSeedSource().name == "public"


def test_public_seeds_requires_positive_n():
    with pytest.raises(ValueError):
        PublicSeedSource(n=0)


def test_hidden_seed_source_is_a_refusing_seam():
    # Hidden seeds are human-owned (grader/hidden_seeds/, hook-denied). The seam
    # must never silently produce seeds.
    with pytest.raises(NotImplementedError, match="human-owned"):
        HiddenSeedSource().seeds()
