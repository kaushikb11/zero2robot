#!/usr/bin/env python3
"""gen_notebook — deterministic Colab notebook generator for chapter artifacts.

This is the WRITE side of the offline learning path: the notebooks under
notebooks/ are generated from chapter artifacts, never hand-edited
(notebooks/CLAUDE.md). Regenerate via this script (the notebook-tier-test
skill drives it); the ci-notebook lane executes the result headlessly and
check_notebook_hashes.py fails CI if the notebook drifts from its artifact.

Contract (notebook-tier-test SKILL.md):
1. Parse the chapter's {artifact}.py regions into notebook cells in canonical
   order: title+objectives (meta.yaml) -> pinned installs -> detect_device +
   seed -> data/model/train/eval -> rerun viewing instructions.
2. Inject the honest wall-clock cell: reads curriculum/common/wallclock.csv and
   prints render_line for the detected tier (PENDING renders "not yet
   measured", never a guess).
3. Write the source artifact's content hash into meta.yaml (`notebook_hash`),
   plus the generated notebook's own byte hash (`notebook_file_hash`) so a
   hand-edited notebook is caught too.

Determinism: cells carry stable sequential ids, no execution timestamps, no
embedded outputs. Same artifact -> byte-identical notebook, so notebook_hash /
notebook_file_hash are reproducible and the drift gate is meaningful.

Region parsing reuses infra/ci/lib/regions.parse_regions (BYTE-IDENTICAL to the
prose-code drift gate); nothing here re-implements marker parsing.

  python infra/ci/gen_notebook.py ch0.4-record ch1.1-bc
  python infra/ci/gen_notebook.py --all      # every already-generated chapter
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# Run as a loose script from infra/ci/ (like the sibling gates), so lib/ is
# importable both as `python infra/ci/gen_notebook.py` and from the ci dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.chapters import Chapter, discover_chapters  # noqa: E402
from lib.regions import parse_regions  # noqa: E402

# Pinned runtime deps a Colab kernel needs to run any chapter (pyproject.toml
# [project].dependencies). Kept in lockstep with the pins by hand — a version
# bump is an upstream-pin-check PR, which will re-run this generator.
PINNED_DEPS = [
    "mujoco==3.10.0",
    "torch==2.10.0",
    "numpy==2.4.6",
    "rerun-sdk==0.26.2",
    "lerobot==0.4.4",
    "onnx~=1.17",
    "onnxruntime~=1.20",
]

# Chapter-specific extra deps. A chapter that opts into the ONE jax excursion
# (meta.yaml `allow_mjx: true`, decisions/015) needs the jax stack on Colab —
# the base PINNED_DEPS are torch-only, so without this its setup region's
# `import jax` would fail. Mirrors pyproject.toml
# [project.optional-dependencies].mjx, kept in lockstep by hand (a version bump
# is an upstream-pin-check PR, which re-runs this generator). CPU-by-default:
# jax installs a CPU jaxlib on Colab; the GPU Scale Lab adds jax[cuda12] itself.
MJX_EXTRA_DEPS = [
    "jax~=0.10",
    "mujoco-mjx==3.10.0",
    "flax~=0.12",
    "optax~=0.2",
]

# GitHub repo the Colab setup cell clones when it is not already inside a
# checkout. Placeholder org matches datasets/checkpoints (HF `zero2robot/`).
REPO_URL = "https://github.com/kaushikb11/zero2robot"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lines(text: str) -> list[str]:
    """nbformat `source` convention: list of lines, each keeping its newline
    except the last. Deterministic and diff-friendly."""
    if text == "":
        return []
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _lines(source.rstrip("\n")),
    }


def _markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(source.rstrip("\n")),
    }


def title_markdown(chapter: Chapter) -> str:
    """Title + objectives + honest framing, from meta.yaml (never invented)."""
    title = str(chapter.meta.get("title") or chapter.id)
    lines = [f"# {chapter.id} — {title}", ""]
    objectives = chapter.meta.get("objectives") or []
    if objectives:
        lines.append("**By the end of this notebook you will be able to:**")
        lines.append("")
        for obj in objectives:
            lines.append(f"- {obj}")
        lines.append("")
    lines += [
        "This notebook is **generated** from the chapter's single source file",
        f"(`{chapter.meta.get('artifact')}`) — do not edit it by hand; "
        "regenerate it",
        "with `infra/ci/gen_notebook.py` (the `notebook-tier-test` skill). The "
        "code",
        "cells below are the artifact's own regions, verbatim, so what you run "
        "here is",
        "exactly what the chapter teaches.",
        "",
        "> **Free-tier floor.** Set the environment variable `Z2R_PROFILE="
        "cpu-smoke`",
        "> (nightly CI does) for a tiny hermetic pass that finishes in minutes "
        "on a CPU.",
        "> Leave it unset for the full run on a Colab T4.",
    ]
    return "\n".join(lines)


def setup_cell(chapter: Chapter) -> str:
    # artifact_relpath from repo root, so the cell can locate/clone the repo.
    rel = chapter.artifact_path
    # Compute path relative to the repo root (…/curriculum/…). The chapter dir
    # is two-or-more levels under the root; find the "curriculum" anchor.
    parts = rel.parts
    idx = parts.index("curriculum")
    artifact_relpath = "/".join(parts[idx:])
    # A jax-excursion chapter (allow_mjx) appends the mjx extra so the Colab
    # install cell has the jax stack its setup region imports; every other
    # chapter's install cell is byte-unchanged (extra is empty).
    extra = MJX_EXTRA_DEPS if chapter.meta.get("allow_mjx") else []
    deps = " ".join(f'"{d}"' for d in PINNED_DEPS + extra)
    return f'''# --- generated setup cell (Colab): pinned installs + repo on path ---
# On Colab this installs the PINNED deps and clones the repo; inside an existing
# checkout (CI / local) it reuses what is here (no install, no clone). Either
# way it puts the repo root on sys.path and chdirs to the REPO ROOT (how the
# chapters are canonically invoked, and what their subprocess children use as
# cwd); the artifact's own `Path(__file__).resolve().parents[3]` still resolves.
import os
import subprocess
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
ARTIFACT_RELPATH = "{artifact_relpath}"

if IN_COLAB:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", {deps}],
        check=True,
    )
    if not Path("zero2robot").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "{REPO_URL}", "zero2robot"],
            check=True,
        )
    REPO_ROOT = Path("zero2robot").resolve()
else:
    REPO_ROOT = Path.cwd().resolve()
    while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "pyproject.toml").exists():
        REPO_ROOT = REPO_ROOT.parent

ARTIFACT = REPO_ROOT / ARTIFACT_RELPATH
assert ARTIFACT.is_file(), f"artifact not found: {{ARTIFACT}}"
sys.path.insert(0, str(REPO_ROOT))          # so `curriculum.common` imports resolve
os.chdir(REPO_ROOT)                          # run FROM REPO ROOT: matches the chapters' canonical invocation and their subprocess children's cwd=ROOT (chapters resolve their own paths via __file__, not cwd, so this is safe)
__file__ = str(ARTIFACT)                     # the artifact's setup region reads this
print(f"repo root: {{REPO_ROOT}}")
print(f"artifact:  {{ARTIFACT}}")'''


def wallclock_cell(chapter: Chapter) -> str:
    return f'''# --- generated wall-clock cell (honest: measured or "not yet measured") ---
# Every wall-clock number a learner sees was MEASURED on real hardware
# (curriculum/common/wallclock.csv). A tier we have not benchmarked yet prints
# "not yet measured" — never a guess (style-guide rule 6).
from curriculum.common import wallclock
from curriculum.common.device import detect_device

_TIER = {{"cuda": "t4", "mps": "mps", "cpu": "cpu-laptop"}}
_tier = _TIER[detect_device()]
print(wallclock.render_line("{chapter.id}", _tier))'''


def runconfig_cell(chapter: Chapter) -> str:
    prog = str(chapter.meta.get("artifact"))
    return f'''# --- generated run-config cell: the artifact parses these as CLI args ---
# The artifact below is the real chapter script; it reads its options from
# sys.argv. CI sets Z2R_PROFILE=cpu-smoke for a tiny hermetic pass (--smoke pins
# a few steps + CPU + no rerun .rrd). Unset it and edit this cell for a full run.
import os
import sys

_smoke = os.environ.get("Z2R_PROFILE") == "cpu-smoke"
sys.argv = ["{prog}", "--seed", "0"]
if _smoke:
    sys.argv += ["--smoke", "--no-rerun"]
print("argv:", sys.argv)'''


def closing_markdown(chapter: Chapter) -> str:
    return "\n".join(
        [
            "## Viewing the run in rerun",
            "",
            "Every chapter logs to [rerun.io](https://rerun.io) — training "
            "curves, the",
            "sim, the policy's actions. A full (non-smoke) run writes a `.rrd` "
            "under the",
            "chapter's `outputs/` directory; download it and open it locally "
            "with:",
            "",
            "```bash",
            "rerun outputs/<chapter>/<name>.rrd",
            "```",
            "",
            "The `--smoke` / `Z2R_PROFILE=cpu-smoke` path skips the `.rrd` (CI "
            "does not",
            "need a viewer). For the full experience, run without the smoke "
            "profile.",
            "",
            "---",
            "",
            "_Generated from the chapter artifact by "
            "`infra/ci/gen_notebook.py`. Do not hand-edit — regenerate._",
        ]
    )


def build_notebook(chapter: Chapter) -> dict:
    """Assemble the deterministic notebook node for one chapter."""
    artifact_path = chapter.artifact_path
    if artifact_path is None or not artifact_path.is_file():
        raise SystemExit(
            f"gen_notebook: {chapter.id}: artifact "
            f"'{chapter.meta.get('artifact')}' missing"
        )
    text = artifact_path.read_text(encoding="utf-8")
    regions = parse_regions(text, source=str(artifact_path))

    cells: list[dict] = [
        _markdown_cell(title_markdown(chapter)),
        _code_cell(setup_cell(chapter)),
        _code_cell(wallclock_cell(chapter)),
        _code_cell(runconfig_cell(chapter)),
    ]
    # The artifact's own regions, verbatim, in file order (parse_regions
    # preserves insertion order). setup region carries detect_device + seed.
    for region_text in regions.values():
        cells.append(_code_cell(region_text))
    cells.append(_markdown_cell(closing_markdown(chapter)))

    # Stable sequential ids (nbformat 4.5 requires cell ids) — no randomness,
    # no timestamps: byte-deterministic output.
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i}"

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def notebook_bytes(nb: dict) -> bytes:
    """Serialize deterministically: indent=1 (nbformat style), unicode kept,
    trailing newline. No timestamps anywhere, so bytes are reproducible."""
    return (
        json.dumps(nb, indent=1, ensure_ascii=False, sort_keys=False) + "\n"
    ).encode("utf-8")


def update_meta_text(existing: str, notebook_hash: str, file_hash: str) -> str:
    nb_line = (
        f"notebook_hash: {notebook_hash}"
        "        # sha256 of the artifact at generation; drift => regenerate "
        "(notebook-tier-test)"
    )
    fh_line = (
        f"notebook_file_hash: {file_hash}"
        "   # sha256 of the generated notebook bytes; mismatch => hand-edited"
    )
    # If our keys already exist, replace them IN PLACE so a re-run is a true no-op
    # and never relocates the block (idempotent + comment-safe, mirroring
    # write_region_hashes.py). Only append when they're absent.
    if re.search(r"(?m)^notebook_hash:", existing):
        out = re.sub(r"(?m)^notebook_hash:.*$", nb_line, existing, count=1)
        out = re.sub(r"(?m)^notebook_file_hash:.*$", fh_line, out, count=1)
        return out
    stripped = existing.rstrip("\n")
    block = f"{nb_line}\n{fh_line}\n"
    return f"{stripped}\n\n{block}" if stripped else block


def generate(chapter: Chapter, root: Path, notebooks_dir: Path) -> str:
    nb = build_notebook(chapter)
    data = notebook_bytes(nb)
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    nb_path = notebooks_dir / f"{chapter.id}.ipynb"
    nb_path.write_bytes(data)

    artifact_hash = sha256_bytes(chapter.artifact_path.read_bytes())
    file_hash = sha256_bytes(data)
    meta_text = chapter.meta_path.read_text(encoding="utf-8")
    chapter.meta_path.write_text(
        update_meta_text(meta_text, artifact_hash, file_hash), encoding="utf-8"
    )
    return (
        f"generated {nb_path.relative_to(root).as_posix()} "
        f"({len(nb['cells'])} cells); "
        f"notebook_hash={artifact_hash[:12]}… "
        f"notebook_file_hash={file_hash[:12]}…"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "chapters",
        nargs="*",
        help="chapter id(s) to generate (e.g. ch0.4-record). Omit with --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="regenerate every chapter that already carries notebook_hash",
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2]),
        help="repo root (default: auto-detected)",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    notebooks_dir = root / "notebooks"

    by_id = {ch.id: ch for ch in discover_chapters(root)}

    if args.all:
        targets = [
            ch for ch in by_id.values() if ch.meta.get("notebook_hash") is not None
        ]
        if not targets:
            print("gen_notebook: --all: no chapters carry notebook_hash yet")
            return 0
    else:
        if not args.chapters:
            parser.error("give chapter id(s) or use --all")
        targets = []
        for cid in args.chapters:
            if cid not in by_id:
                print(f"gen_notebook: unknown chapter '{cid}'", file=sys.stderr)
                return 1
            targets.append(by_id[cid])

    for chapter in targets:
        print(generate(chapter, root, notebooks_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
