#!/usr/bin/env python3
"""run_notebooks — headless execution of every generated notebook.

Contract (ci-lanes.md lane 2, .github/workflows/ci-notebook.yml,
`make check-notebooks-exec`): execute ALL notebooks under notebooks/ headlessly
via nbclient and assert each completes clean within a free-tier time budget.
This is the EXECUTION lane — distinct from `check_notebook_hashes.py`, which only
proves a notebook is byte-in-sync with its chapter artifact (never runs it).
Needs an installed kernel — `ipykernel` is in the `notebooks` extra (decision 016).

- --profile cpu-smoke sets Z2R_PROFILE=cpu-smoke in the kernel environment
  (kernels inherit this process's environment). The artifact's run-config cell
  reads it and passes --smoke --no-rerun for a tiny hermetic pass.
- --timeout N: per-notebook cell-execution timeout in seconds (default 600).
- --fail-fast true|false: stop at the first failure (remaining notebooks are
  reported as "skipped").
- --report PATH: JSON report {notebook: {status, error, reason, seconds}} —
  always written (even on failure) so the issue-filing step can read it.
- SKIP-VS-FAIL: a failure that clearly names an ABSENT dataset/checkpoint (see
  skip_reason) is reported "skip" with a reason and does NOT fail the lane — a
  learner provisions those in an earlier chapter. Every other cell failure is a
  hard "fail". The cpu-smoke path is hermetic, so skips should be rare; the net
  is deliberately narrow so a real code bug is never masked as a skip.
- Exit 1 if any notebook FAILED (skips don't count), AFTER writing the report.
- Zero notebooks is a success with a note (young repo).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# A missing dataset/checkpoint (or the Hub artifact behind it) is not a code
# defect the execution lane should fail on. skip_reason() classifies such a
# failure as skip-with-reason; everything else stays a hard failure.
_MISSING_FILE_MARKERS = (
    "FileNotFoundError",
    "No such file or directory",
    "does not exist",
)
_ARTIFACT_PATH_HINTS = (
    "datasets/",
    "checkpoints/",
    "outputs/",
    ".parquet",
    ".safetensors",
    ".rrd",
    "pusht-demos",
)
_HUB_UNREACHABLE_MARKERS = (
    "HfHubHTTPError",
    "EntryNotFoundError",
    "RepositoryNotFoundError",
    "LocalEntryNotFoundError",
    "Max retries exceeded",
)


def skip_reason(error_text: str) -> str | None:
    """Return a skip reason if `error_text` is an absent-dataset/checkpoint
    failure, else None (a real failure).

    Conservative on purpose: only an error that BOTH looks like a missing file
    AND names a dataset/checkpoint/output artifact path — or a Hugging Face Hub
    fetch that could not reach the artifact — is a skip. A plain ValueError,
    AssertionError, shape mismatch, or any other cell exception is a hard fail.
    """
    if any(marker in error_text for marker in _HUB_UNREACHABLE_MARKERS):
        return "dataset/checkpoint not reachable (Hugging Face Hub)"
    if any(marker in error_text for marker in _MISSING_FILE_MARKERS) and any(
        hint in error_text for hint in _ARTIFACT_PATH_HINTS
    ):
        return "dataset/checkpoint not present locally"
    return None


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
    skipped = 0
    stopped_early = False

    for notebook in notebooks:
        rel = notebook.relative_to(root).as_posix()
        if stopped_early:
            report[rel] = {
                "status": "skipped", "error": None, "reason": None, "seconds": 0.0
            }
            continue
        start = time.monotonic()
        try:
            execute_notebook(notebook, timeout=args.timeout)
        except Exception as exc:  # nbclient raises many types; report them all
            seconds = round(time.monotonic() - start, 3)
            error = f"{type(exc).__name__}: {exc}"
            reason = skip_reason(error)
            if reason is not None:
                report[rel] = {
                    "status": "skip", "error": None, "reason": reason,
                    "seconds": seconds,
                }
                skipped += 1
                print(f"skip {rel} ({seconds}s): {reason}")
                continue
            report[rel] = {
                "status": "fail", "error": error, "reason": None,
                "seconds": seconds,
            }
            failed += 1
            print(f"FAIL {rel} ({seconds}s): {error}", file=sys.stderr)
            if args.fail_fast:
                stopped_early = True
        else:
            seconds = round(time.monotonic() - start, 3)
            report[rel] = {
                "status": "pass", "error": None, "reason": None, "seconds": seconds
            }
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

    skip_note = f", {skipped} skipped (absent dataset/checkpoint)" if skipped else ""

    if failed:
        print(
            f"run_notebooks: {failed}/{len(notebooks)} notebook(s) FAILED"
            f"{skip_note} (ci-lanes.md lane 2). Notebooks are generated — fix "
            "the chapter or regenerate via notebook-tier-test, never hand-edit "
            "(notebooks/CLAUDE.md).",
            file=sys.stderr,
        )
        return 1

    passed = len(notebooks) - skipped
    print(
        f"run_notebooks: {passed}/{len(notebooks)} notebook(s) executed clean"
        f"{skip_note} — OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
