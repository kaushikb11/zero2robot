#!/usr/bin/env python3
"""full_system_smoke — the clean-clone, whole-product "does it all still stand
up together" proof.

Runs the product end-to-end from the current tree and reports one pass/fail
summary. Built for `make full-system-smoke` locally AND a nightly CI lane; it is
deliberately NOT part of the fast default `make check`.

Stages (each shells out to the canonical single-purpose tool — no logic is
re-implemented here):

  gates          make check  — ruff + every pedagogy gate + unit tests
  chapter-smoke  smoke_chapters.py --all --seed 0 --verify-determinism
                 (every chapter artifact --smoke, twice, byte-compared)
  export-parity  check_export_parity.py — torch<->ONNX parity for every
                 export_*_onnx.py --smoke
  site-build     npm run build in site/ (SKIP-with-reason if node_modules or
                 npm is absent — a clean checkout without `npm ci` is not a
                 product failure)
  notebooks      run_notebooks.py --profile cpu-smoke — execute all notebooks
                 headless (HEAVY; opt-in, not in the default stage set)

Selection: default runs [gates, chapter-smoke, export-parity, site-build].
`--with-notebooks` adds the notebook lane; `--only A,B` / `--skip A,B` filter;
`--list` prints the stage registry. Each stage reports OK / FAIL / SKIP in the
house idiom; exit 1 iff any stage FAILED (skips never fail the run).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent

STAGES: dict[str, str] = {
    "gates": "make check — ruff + pedagogy gates + unit tests",
    "chapter-smoke": "every chapter artifact --smoke (2x determinism)",
    "export-parity": "torch<->ONNX parity for every export_*_onnx.py --smoke",
    "site-build": "npm run build in site/ (skips if node_modules/npm absent)",
    "notebooks": "execute all notebooks headless (cpu-smoke) — heavy",
}
DEFAULT_STAGES = ["gates", "chapter-smoke", "export-parity", "site-build"]


def resolve_python(root: Path, override: str | None) -> str:
    if override:
        return override
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return str(venv_python)
    return sys.executable


def _run(cmd: list[str], cwd: Path, timeout: int) -> tuple[str, str]:
    """Run a stage subprocess. Returns (status, detail) with status OK/FAIL."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return "FAIL", f"timed out after {timeout}s"
    except OSError as exc:
        return "FAIL", f"could not launch: {exc}"
    if result.returncode != 0:
        combined = (result.stdout or "") + (result.stderr or "")
        tail = combined.strip().splitlines()[-4:]
        return "FAIL", (
            f"exit {result.returncode}: " + " | ".join(tail)
            if tail else f"exit {result.returncode}"
        )
    return "OK", "ok"


def stage_gates(root: Path, python: str, timeout: int) -> tuple[str, str]:
    # `make check` is the single source of truth for the gate bundle; if make is
    # unavailable (unusual on CI/local), skip rather than falsely fail.
    if not _which("make"):
        return "SKIP", "`make` not on PATH"
    return _run(["make", "PY=" + python, "check"], root, timeout)


def stage_chapter_smoke(root: Path, python: str, timeout: int) -> tuple[str, str]:
    return _run(
        [python, str(CI_DIR / "smoke_chapters.py"), "--all", "--seed", "0",
         "--verify-determinism", "--python", python],
        root, timeout,
    )


def stage_export_parity(root: Path, python: str, timeout: int) -> tuple[str, str]:
    return _run(
        [python, str(CI_DIR / "check_export_parity.py"), "--python", python],
        root, timeout,
    )


def stage_site_build(root: Path, python: str, timeout: int) -> tuple[str, str]:
    # NOTE: `npm run build` runs site/'s `prebuild` (sync:hashes + fetch:models),
    # and sync:hashes REWRITES curriculum meta.yaml region_hashes in place. On a
    # clean/ephemeral CI checkout that is correct and harmless; run LOCALLY it can
    # dirty tracked meta.yaml (revert with `git restore curriculum/**/meta.yaml`
    # if the committed order differs from sync's). This is the real product build
    # command, so the lane runs it as-is rather than bypassing prebuild.
    site = root / "site"
    if not site.is_dir():
        return "SKIP", "no site/ directory"
    if not _which("npm"):
        return "SKIP", "npm not on PATH"
    if not (site / "node_modules").is_dir():
        return "SKIP", "site/node_modules absent (run `npm ci` in site/ first)"
    return _run(["npm", "run", "build"], site, timeout)


def stage_notebooks(root: Path, python: str, timeout: int) -> tuple[str, str]:
    report = CI_DIR / "reports" / "notebooks.json"
    return _run(
        [python, str(CI_DIR / "run_notebooks.py"), "--profile", "cpu-smoke",
         "--fail-fast", "false", "--report", str(report)],
        root, timeout,
    )


STAGE_FNS = {
    "gates": stage_gates,
    "chapter-smoke": stage_chapter_smoke,
    "export-parity": stage_export_parity,
    "site-build": stage_site_build,
    "notebooks": stage_notebooks,
}


def _which(exe: str) -> bool:
    import shutil

    return shutil.which(exe) is not None


def select_stages(args: argparse.Namespace) -> list[str]:
    if args.only:
        requested = [s.strip() for s in args.only.split(",") if s.strip()]
    else:
        requested = list(DEFAULT_STAGES)
        if args.with_notebooks and "notebooks" not in requested:
            requested.append("notebooks")
    skip = {s.strip() for s in (args.skip or "").split(",") if s.strip()}
    return [s for s in requested if s not in skip]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean-clone full-system smoke (the whole-product proof)."
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2]),
        help="repo root (default: auto-detected from this script's location)",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated stages to run (overrides the default set)",
    )
    parser.add_argument(
        "--skip", default=None, help="comma-separated stages to skip"
    )
    parser.add_argument(
        "--with-notebooks",
        action="store_true",
        help="add the heavy notebook-execution stage to the default set",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="per-stage timeout in seconds (default 3600)",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="interpreter for the Python stages (default: {root}/.venv/bin/"
        "python, else this interpreter)",
    )
    parser.add_argument(
        "--list", action="store_true", help="print the stage registry and exit"
    )
    args = parser.parse_args(argv)

    if args.list:
        print("full_system_smoke stages:")
        for name, desc in STAGES.items():
            default = " (default)" if name in DEFAULT_STAGES else ""
            print(f"  {name.ljust(14)} {desc}{default}")
        return 0

    root = Path(args.root).resolve()
    python = resolve_python(root, args.python)

    stages = select_stages(args)
    unknown = [s for s in stages if s not in STAGE_FNS]
    if unknown:
        print(
            f"FAIL full_system_smoke: unknown stage(s): {', '.join(unknown)} "
            f"(known: {', '.join(STAGES)})",
            file=sys.stderr,
        )
        return 1
    if not stages:
        print("full_system_smoke: no stages selected — OK (nothing to run)")
        return 0

    print(f"full_system_smoke: running {len(stages)} stage(s): {', '.join(stages)}")
    print(f"  python: {python}\n  root:   {root}\n")

    results: list[tuple[str, str, str, float]] = []
    for name in stages:
        print(f"==> {name}: {STAGES[name]}")
        start = time.monotonic()
        status, detail = STAGE_FNS[name](root, python, args.timeout)
        seconds = round(time.monotonic() - start, 1)
        results.append((name, status, detail, seconds))
        print(f"<== {name}: {status} ({seconds}s)\n")

    width = max(len(name) for name, _, _, _ in results)
    print("=" * 60)
    print("full-system-smoke summary")
    print("=" * 60)
    print(f"{'stage'.ljust(width)}  result  time     detail")
    for name, status, detail, seconds in results:
        shown = detail if status != "OK" else "ok"
        print(f"{name.ljust(width)}  {status.ljust(4)}    {str(seconds)+'s':<8} {shown}")

    failed = [name for name, status, _, _ in results if status == "FAIL"]
    skipped = [name for name, status, _, _ in results if status == "SKIP"]
    ok = [name for name, status, _, _ in results if status == "OK"]
    print("=" * 60)
    print(f"{len(ok)} OK, {len(skipped)} skipped, {len(failed)} failed")

    if failed:
        print(
            f"\nFAIL full_system_smoke: {', '.join(failed)} failed — the product "
            "does not fully stand up. See the per-stage detail above.",
            file=sys.stderr,
        )
        return 1
    note = f" ({len(skipped)} skipped: {', '.join(skipped)})" if skipped else ""
    print(f"\nfull_system_smoke: all {len(ok)} stage(s) passed{note} — OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
