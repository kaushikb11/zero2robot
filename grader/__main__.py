"""`python -m grader <subcommand>` dispatcher.

    python -m grader check 0.1 [--json]

`check` also runs standalone (`python -m grader.check`). This dispatcher is the
single friendly entrypoint.
"""

from __future__ import annotations

import sys

from . import check

_SUBCOMMANDS = {"check": check.main}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _SUBCOMMANDS:
        print("usage: python -m grader check ...", file=sys.stderr)
        print("  check <chapter>            run a chapter's exercise checks", file=sys.stderr)
        return 2
    return _SUBCOMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
