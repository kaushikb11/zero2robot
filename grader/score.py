"""Leaderboard scoring CLI: validate an ONNX submission, then score it.

    python -m grader.score policy.onnx --config-hash abc123 --division free
    python -m grader.score policy.onnx --config-hash abc123 --division open --json

The full leaderboard path in one command: fail-closed contract validation
(grader.contract) THEN deterministic scoring on the PUBLIC seeds
(grader.scoring). Prints the score, the reproducible submission hash, and the
percentile band. Hidden-seed scoring is human-owned and NOT reachable here
(grader.seeds.HiddenSeedSource is a seam that refuses to run).

Offline by construction — reads only the local ONNX file and the local PushT
MJCF; no network on this path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from curriculum.common.envs.pusht import PushTEnv

from .contract import ContractError, validate_submission
from .scoring import score_onnx_file
from .seeds import PublicSeedSource
from .submission import (
    Division,
    Submission,
    free_tier_declaration_is_plausible,
    percentile_band,
)


def grade(
    onnx_path: str | Path,
    config_hash: str,
    division: Division,
    *,
    n_seeds: int = 50,
    declared_runtime_min: float | None = None,
    cohort_scores: list[float] | None = None,
) -> dict:
    """Validate + score a submission on the public seeds. Returns a report dict."""
    # 1. fail-closed contract gate against the scoring env's dims.
    validated = validate_submission(
        onnx_path,
        expected_obs_dim=PushTEnv.OBS_DIM,
        expected_act_dim=PushTEnv.ACT_DIM,
    )
    # 2. deterministic scoring on the public seed set.
    seed_source = PublicSeedSource(n=n_seeds)
    result = score_onnx_file(validated.onnx_path, seed_source)

    sub = Submission(
        onnx_path=Path(onnx_path),
        config_hash=config_hash,
        division=division,
        declared_runtime_min=declared_runtime_min,
    )
    return {
        "submission_hash": sub.submission_hash(result.seeds),
        "division": division.value,
        "config_hash": config_hash,
        "seed_source": result.seed_source,
        "n_episodes": result.n_episodes,
        "successes": result.successes,
        "success_rate": round(result.success_rate, 6),
        "mean_episode_return": round(result.mean_episode_return, 6),
        "score": result.score,
        "percentile_band": percentile_band(result.score, cohort_scores or []),
        "free_tier_plausible": free_tier_declaration_is_plausible(sub),
        "contract_version": validated.contract_version,
        "opset": validated.opset,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m grader.score",
        description="Validate and score an ONNX leaderboard submission.",
    )
    parser.add_argument("onnx_path", help="path to a contract-v1 ONNX policy")
    parser.add_argument("--config-hash", required=True, help="training-config hash")
    parser.add_argument(
        "--division", choices=[d.value for d in Division], default="free"
    )
    parser.add_argument("--n-seeds", type=int, default=50, help="public seeds to score")
    parser.add_argument(
        "--declared-runtime-min", type=float, default=None,
        help="declared training wall-clock (min) for free-tier plausibility",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    args = parser.parse_args(argv)

    try:
        report = grade(
            args.onnx_path,
            args.config_hash,
            Division(args.division),
            n_seeds=args.n_seeds,
            declared_runtime_min=args.declared_runtime_min,
        )
    except ContractError as exc:
        print(f"REJECTED (contract): {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"submission {report['submission_hash'][:16]}… ({report['division']})")
        print(f"  seeds       : {report['seed_source']} × {report['n_episodes']}")
        print(f"  success     : {report['successes']}/{report['n_episodes']} "
              f"(rate {report['success_rate']:.3f})")
        print(f"  score       : {report['score']}")
        print(f"  band        : {report['percentile_band']}")
        print(f"  mean return : {report['mean_episode_return']:.3f}")
        if report["free_tier_plausible"] is not None:
            print(f"  free-tier plausible: {report['free_tier_plausible']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
