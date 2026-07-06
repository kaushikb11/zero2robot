"""Deterministic scoring harness: run a validated ONNX policy on PushT seeds.

REUSES curriculum/common/envs/pusht (the shared scoring env + its success
metric) — the grader imports the task, never copies it (decision 004,
grader/CLAUDE.md). The rollout MIRRORS ch1.1 BC's eval loop exactly
(reset(seed) -> obs -> action -> step until done; count info["success"]),
so a leaderboard score is the same quantity a learner sees locally.

Determinism (grader/CLAUDE.md): PushT resets are bitwise-reproducible on CPU
and onnxruntime CPU inference is a pure function of the weights, so
`policy + seed set -> score` is reproducible. Same submission + same seeds ->
identical score, every time (test_scoring proves it twice-over).

No network anywhere on this path: the env loads a local MJCF, the policy is a
local ONNX file run through onnxruntime. The only I/O is reading those files.

Sandbox seam: this function is the intended entrypoint of the gVisor sandbox
(policy.yaml). It honors the wallclock cap in-process as a *guard* (raising if
exceeded) — but the guard is never an input to the score, so it cannot make a
score nondeterministic. Isolation/network/CPU/memory caps are enforced by the
container around this process, not here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from curriculum.common.envs.pusht import PushTEnv

from .policy import OnnxPolicy
from .sandbox import SandboxPolicy, load_policy
from .seeds import PublicSeedSource, SeedSource


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    success: bool
    episode_return: float
    steps: int


@dataclass(frozen=True)
class ScoreResult:
    """The deterministic outcome of scoring a policy over a seed set."""

    seed_source: str
    seeds: list[int]
    episodes: list[EpisodeResult] = field(default_factory=list)

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def successes(self) -> int:
        return sum(e.success for e in self.episodes)

    @property
    def success_rate(self) -> float:
        return self.successes / self.n_episodes if self.episodes else 0.0

    @property
    def mean_episode_return(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.episode_return for e in self.episodes) / self.n_episodes

    @property
    def score(self) -> float:
        """The headline leaderboard score: success rate as a percentage."""
        return round(100.0 * self.success_rate, 6)


def rollout(policy: OnnxPolicy, env: PushTEnv, seed: int) -> EpisodeResult:
    """One deterministic episode. Mirrors ch1.1 bc.py's eval rollout."""
    obs = env.reset(seed=seed)
    episode_return, done, info, steps = 0.0, False, {}, 0
    while not done:
        action = policy.act(obs)
        obs, reward, done, info = env.step(action)
        episode_return += reward
        steps += 1
    return EpisodeResult(
        seed=seed,
        success=bool(info.get("success", False)),
        episode_return=round(episode_return, 6),
        steps=steps,
    )


def score_policy(
    policy: OnnxPolicy,
    seed_source: SeedSource | None = None,
    *,
    sandbox_policy: SandboxPolicy | None = None,
) -> ScoreResult:
    """Score a loaded ONNX policy over a seed source. Deterministic."""
    seed_source = seed_source or PublicSeedSource()
    sandbox_policy = sandbox_policy or load_policy()
    seeds = seed_source.seeds()

    env = PushTEnv()
    deadline = time.monotonic() + sandbox_policy.wallclock_limit_s
    episodes: list[EpisodeResult] = []
    for seed in seeds:
        episodes.append(rollout(policy, env, seed))
        # Guard only — never feeds the score, so determinism is preserved.
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"scoring exceeded sandbox wallclock cap "
                f"{sandbox_policy.wallclock_limit_s}s after "
                f"{len(episodes)}/{len(seeds)} episodes"
            )
    return ScoreResult(seed_source=seed_source.name, seeds=seeds, episodes=episodes)


def score_onnx_file(
    onnx_path: str | Path,
    seed_source: SeedSource | None = None,
    *,
    sandbox_policy: SandboxPolicy | None = None,
) -> ScoreResult:
    """Convenience: load an ONNX file (onnxruntime-only) and score it.

    NOTE: this assumes the file already passed grader.contract.validate_submission
    (the fail-closed gate). The leaderboard path validates first, then scores.
    """
    policy = OnnxPolicy(onnx_path)
    return score_policy(policy, seed_source, sandbox_policy=sandbox_policy)
