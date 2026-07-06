"""Deterministic scoring: same submission + same seeds => identical score."""

from __future__ import annotations

from grader.policy import OnnxPolicy
from grader.scoring import score_onnx_file, score_policy
from grader.seeds import PublicSeedSource
from grader.submission import Division, Submission, percentile_band


def test_score_is_deterministic_twice(sample_onnx):
    seeds = PublicSeedSource(n=6)
    r1 = score_onnx_file(sample_onnx, seeds)
    r2 = score_onnx_file(sample_onnx, seeds)
    assert r1.score == r2.score
    assert r1.success_rate == r2.success_rate
    assert r1.mean_episode_return == r2.mean_episode_return
    # per-episode, not just the aggregate.
    assert [e.__dict__ for e in r1.episodes] == [e.__dict__ for e in r2.episodes]


def test_score_stable_across_fresh_policy_load(sample_onnx):
    seeds = PublicSeedSource(n=6)
    a = score_policy(OnnxPolicy(sample_onnx), seeds)
    b = score_policy(OnnxPolicy(sample_onnx), seeds)
    assert a.score == b.score
    assert [e.episode_return for e in a.episodes] == [e.episode_return for e in b.episodes]


def test_score_shape_and_range(sample_onnx):
    result = score_onnx_file(sample_onnx, PublicSeedSource(n=6))
    assert result.n_episodes == 6
    assert 0.0 <= result.score <= 100.0
    assert result.seed_source == "public"
    assert result.seeds == PublicSeedSource(n=6).seeds()
    assert all(e.steps > 0 for e in result.episodes)


def test_submission_hash_reproducible(sample_onnx):
    seeds = PublicSeedSource(n=6).seeds()
    sub = Submission(sample_onnx, config_hash="cfg123", division=Division.FREE)
    h1 = sub.submission_hash(seeds)
    h2 = sub.submission_hash(seeds)
    assert h1 == h2 and len(h1) == 64
    # a different seed set (the hidden set would differ) => a different hash.
    assert sub.submission_hash(seeds + [1]) != h1


def test_public_seeds_disjoint_from_training_range():
    # ch1.1 BC evals on 10_000+... ; public scoring starts at 900_000. Disjoint
    # by construction (anti-overfitting): never scored on a trained start.
    assert min(PublicSeedSource(n=50).seeds()) >= 900_000


def test_percentile_band_before_ranks(sample_onnx):
    result = score_onnx_file(sample_onnx, PublicSeedSource(n=6))
    assert percentile_band(result.score, []) == "unranked (no cohort yet)"
    # top of a cohort => top band.
    assert percentile_band(100.0, [0.0, 10.0, 50.0]) == "top 10%"
