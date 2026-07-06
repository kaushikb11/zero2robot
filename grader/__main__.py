"""`python -m grader <subcommand>` dispatcher.

    python -m grader check 0.1 [--json]
    python -m grader score policy.onnx --config-hash H --division free [--json]

Each subcommand also runs standalone (`python -m grader.check`,
`python -m grader.score`). This dispatcher is the single friendly entrypoint.
"""

from __future__ import annotations

import sys

from . import check, score

_SUBCOMMANDS = {"check": check.main, "score": score.main}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _SUBCOMMANDS:
        print("usage: python -m grader {check|score} ...", file=sys.stderr)
        print("  check <chapter>            run a chapter's exercise checks", file=sys.stderr)
        print("  score <policy.onnx> ...    validate + score a submission", file=sys.stderr)
        return 2
    return _SUBCOMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
