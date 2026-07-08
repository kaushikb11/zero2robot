#!/usr/bin/env python3
"""check_export_parity — torch<->ONNX parity smoke for every export script.

Contract (03-engineering/artifact-pipeline.md, tensor contracts v1/v2 in
curriculum/common/export_onnx.py): a policy that fails torch<->onnxruntime
parity must never reach the browser playground. Each chapter that ships an ONNX
policy carries a standalone `export_*_onnx.py` whose `--smoke` mode builds a
random-init net, exports it, and asserts parity — proving the SERIALIZATION path
without any trained checkpoint. This gate runs every such `--smoke` and reports
a clean pass/skip/fail table.

- Discovery: curriculum/phase*/ch*/export_*_onnx.py (sorted).
- A script that advertises a `--smoke` flag is RUN: `python <script> --smoke`
  from a throwaway cwd (its outputs land in the tmp dir, never the repo). Exit 0
  AND "parity" in its output -> PASS; anything else -> FAIL (tail shown).
- A script with NO `--smoke` path is SKIPPED with a reason (e.g. the diffusion /
  flow sampler-export scripts read a trained run dir; there is no hermetic
  random-init smoke to run standalone). Skips never fail the gate.
- The chapter ARTIFACTS that export inline under their own --smoke (e.g. bc.py)
  are covered by smoke_chapters.py; this gate is the standalone export scripts.
- House output + exit codes: exit 1 iff any script FAILED.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def resolve_python(root: Path, override: str | None) -> str:
    """Interpreter to run export scripts (mirrors smoke_chapters.resolve_python)."""
    if override:
        return override
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return str(venv_python)
    return sys.executable


def discover_export_scripts(root: Path) -> list[Path]:
    """Every curriculum/phase*/ch*/export_*_onnx.py, sorted."""
    curriculum = root / "curriculum"
    if not curriculum.is_dir():
        return []
    return sorted(curriculum.glob("phase*/ch*/export_*_onnx.py"))


def has_smoke_flag(script: Path) -> bool:
    """True if the script's argparse advertises a --smoke flag (its hermetic,
    checkpoint-free parity path)."""
    try:
        text = script.read_text(encoding="utf-8")
    except OSError:
        return False
    return '"--smoke"' in text or "'--smoke'" in text


def run_export_smoke(
    python: str, script: Path, timeout: int
) -> tuple[str, str]:
    """Run one export script's --smoke. Returns (status, detail).

    status is one of "PASS", "SKIP", "FAIL". Runs in a throwaway cwd so the
    default relative `outputs/...` the scripts write to never touches the repo;
    imports resolve via each script's own __file__-based sys.path insert, so cwd
    is irrelevant to importing curriculum.common.
    """
    if not has_smoke_flag(script):
        return "SKIP", "no --smoke path (needs a trained run dir/checkpoint)"
    with tempfile.TemporaryDirectory(prefix="z2r-export-") as tmp:
        cmd = [python, str(script), "--smoke"]
        try:
            result = subprocess.run(
                cmd,
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return "FAIL", f"timed out after {timeout}s"
        except OSError as exc:
            return "FAIL", f"could not launch: {exc}"
    combined = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        tail = combined.strip().splitlines()[-3:]
        return "FAIL", (
            f"exit {result.returncode}: " + " | ".join(tail)
            if tail else f"exit {result.returncode}"
        )
    if "parity" not in combined.lower():
        return "FAIL", "exit 0 but no 'parity' line — did it assert parity?"
    # Surface the measured delta line for the table (last 'parity' line).
    parity_lines = [ln for ln in combined.splitlines() if "parity" in ln.lower()]
    return "PASS", parity_lines[-1].strip() if parity_lines else "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="torch<->ONNX parity smoke for every export_*_onnx.py."
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2]),
        help="repo root (default: auto-detected from this script's location)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="per-script timeout in seconds (default 300)",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="interpreter for export scripts (default: {root}/.venv/bin/python, "
        "else this interpreter)",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    python = resolve_python(root, args.python)

    scripts = discover_export_scripts(root)
    if not scripts:
        print("check_export_parity: no export_*_onnx.py found — OK")
        return 0

    results: list[tuple[str, str, str]] = []
    for script in scripts:
        rel = script.relative_to(root).as_posix()
        status, detail = run_export_smoke(python, script, args.timeout)
        results.append((rel, status, detail))

    width = max(len(rel) for rel, _, _ in results)
    print(f"\n{'export script'.ljust(width)}  result  detail")
    for rel, status, detail in results:
        print(f"{rel.ljust(width)}  {status.ljust(4)}    {detail}")

    failed = [rel for rel, status, _ in results if status == "FAIL"]
    skipped = sum(1 for _, status, _ in results if status == "SKIP")
    passed = sum(1 for _, status, _ in results if status == "PASS")
    skip_note = f", {skipped} skipped" if skipped else ""

    if failed:
        print(
            f"\nFAIL check_export_parity: {len(failed)}/{len(results)} export "
            f"script(s) failed parity smoke{skip_note}: {', '.join(failed)}",
            file=sys.stderr,
        )
        return 1

    print(
        f"\ncheck_export_parity: {passed}/{len(results)} export script(s) "
        f"passed{skip_note} — OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
