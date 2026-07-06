#!/usr/bin/env python3
"""run_notebooks — headless execution of every generated notebook.

Contract (ci-lanes.md lane 2, .github/workflows/ci-notebook.yml): nightly,
execute ALL notebooks under notebooks/ headlessly via nbclient.

- --profile cpu-smoke sets Z2R_PROFILE=cpu-smoke in the kernel environment
  (kernels inherit this process's environment).
- --timeout N: per-notebook cell-execution timeout in seconds (default 600).
- --fail-fast true|false: stop at the first failure (remaining notebooks are
  reported as "skipped").
- --report PATH: JSON report {notebook: {status, error, seconds}} — always
  written (even on failure) so the issue-filing step can read it.
- Exit 1 if any notebook failed, AFTER writing the report.
- Zero notebooks is a success with a note (young repo).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def str2bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got '{value}'")


def discover_notebooks(notebooks_dir: Path) -> list[Path]:
    """All .ipynb under notebooks/ recursively, skipping checkpoints."""
    if not notebooks_dir.is_dir():
        return []
    return sorted(
        p
        for p in notebooks_dir.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in p.parts
    )


def execute_notebook(path: Path, timeout: int) -> None:
    """Execute one notebook in-place-in-memory; raise on any cell failure.

    Kept as a module-level seam so tests can exercise the runner's selection,
    reporting, and exit-code logic without spinning a real kernel.
    """
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    kernel_name = nb.metadata.get("kernelspec", {}).get("name", "python3")
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute all generated notebooks headlessly "
        "(ci-lanes.md lane 2)."
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2]),
        help="repo root (default: auto-detected from this script's location)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="execution profile exported to the kernel as Z2R_PROFILE "
        "(e.g. cpu-smoke)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="per-notebook cell timeout in seconds (default 600)",
    )
    parser.add_argument(
        "--fail-fast",
        type=str2bool,
        default=False,
        metavar="BOOL",
        help="stop at the first failing notebook (default false)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="path to write the JSON report {notebook: {status,error,seconds}}",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    notebooks_dir = root / "notebooks"

    if args.profile:
        os.environ["Z2R_PROFILE"] = args.profile

    notebooks = discover_notebooks(notebooks_dir)
    report: dict[str, dict] = {}
    failed = 0
    stopped_early = False

    for notebook in notebooks:
        rel = notebook.relative_to(root).as_posix()
        if stopped_early:
            report[rel] = {"status": "skipped", "error": None, "seconds": 0.0}
            continue
        start = time.monotonic()
        try:
            execute_notebook(notebook, timeout=args.timeout)
        except Exception as exc:  # nbclient raises many types; report them all
            seconds = round(time.monotonic() - start, 3)
            error = f"{type(exc).__name__}: {exc}"
            report[rel] = {"status": "fail", "error": error, "seconds": seconds}
            failed += 1
            print(f"FAIL {rel} ({seconds}s): {error}", file=sys.stderr)
            if args.fail_fast:
                stopped_early = True
        else:
            seconds = round(time.monotonic() - start, 3)
            report[rel] = {"status": "pass", "error": None, "seconds": seconds}
            print(f"pass {rel} ({seconds}s)")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"report written to {report_path}")

    if not notebooks:
        print("run_notebooks: no notebooks found under notebooks/ — OK")
        return 0

    if failed:
        print(
            f"run_notebooks: {failed}/{len(notebooks)} notebook(s) FAILED "
            "(ci-lanes.md lane 2). Notebooks are generated — fix the chapter "
            "or regenerate via notebook-tier-test, never hand-edit "
            "(notebooks/CLAUDE.md).",
            file=sys.stderr,
        )
        return 1

    print(f"run_notebooks: {len(notebooks)} notebook(s) executed clean — OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
