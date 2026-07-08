#!/usr/bin/env python3
"""check_demo_assets — dangling-model-asset gate for demo embeds.

Contract (site/CLAUDE.md; curriculum/CLAUDE.md #5 "no binaries in git"): each
chapter's demo/embed.yaml declares the model artifact(s) the browser embed loads
(`policy: bc_policy.onnx`, `raw: bc_policy_raw.onnx`, `hil: serl_actor.onnx`, …).
The .onnx/.pt binaries are git-ignored provisioned outputs, so a demo can quietly
reference a model that NOTHING produces — a silent broken embed the reader hits at
runtime. Every such reference must therefore be TRACEABLE: it must either resolve
to a real, produced/provisioned model, or be explicitly marked PENDING.

A model reference RESOLVES when any of these hold:
  1. it is marked PENDING on its line (comment contains "PENDING"), e.g.
     `policy: sac_actor.onnx      # PENDING export`;
  2. a file with that basename exists in the working tree (a provisioned model
     under outputs/, site/…/models, playground/…/models, or the chapter dir);
  3. the chapter's own code (any .py under the chapter directory) writes/names
     that basename — i.e. an exporter produces it;
  4. it is a variant of a resolved base model: stripping a trailing `_<qualifier>`
     from the stem yields a name that resolves by (2) or (3). Demos routinely show
     A/B variants (raw vs curated) of one exported model family; the variant is
     produced by re-running the same exporter, so the family base resolving is
     sufficient evidence.

FAIL (exit 1) on any model reference that resolves by none of the above and is
not marked PENDING — a dangling asset, i.e. a silent broken demo. Recordings
(.rrd), scenes, and metrics.json are regenerated on every run and are out of
scope; this gate is specifically about deployable MODEL artifacts.

SECOND RESPONSIBILITY — the model-hosting TRIANGLE. A deployed site provisions
its live-demo policies at build time from checkpoints/models.yaml (the in-repo
pointer) via site/scripts/fetch_models.py. For that to work, every model a live
toy hard-codes (`const MODEL_URL = "/models/<name>.onnx"`) must trace all the way
through: (site toy ref) -> (manifest entry serving that path) -> (an exporter
that produces the .onnx). A toy referencing a model with NO manifest entry, or a
manifest entry whose exporter is missing or does not produce the file, FAILS —
that demo could never come alive on a fresh checkout. See check_triangle().
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from lib.chapters import Chapter, ChapterError, discover_chapters

# Deployable model artifact extensions the browser embed loads.
MODEL_EXT_RE = r"(?:onnx|ts\.pt|pt|safetensors)"
# A model reference is the VALUE of a `key: value` mapping line whose value is a
# bare model filename. Restricting to real key:value lines (not folded-scalar
# `notes:` prose) is what keeps an in-passing mention like "(bc_policy_break.onnx,
# trained with --break …)" inside a notes block out of scope.
ASSET_LINE_RE = re.compile(
    r"^\s*[\w-]+:\s*(?P<asset>[\w./-]+\." + MODEL_EXT_RE + r")\s*$"
)
PENDING_RE = re.compile(r"pending", re.IGNORECASE)
# Directories that never hold curriculum-provisioned models — skip when indexing.
IGNORE_DIRS = {".venv", ".git", "node_modules", "__pycache__"}

# --- the model-hosting TRIANGLE (checkpoints/models.yaml) --------------------
# A live site toy loads a policy by hard-coding its served path, e.g.
#   const MODEL_URL = "/models/bc_policy.onnx";
# For that demo to work on a deployed/fresh checkout, three things must agree:
#   (site toy ref) -> (manifest entry) -> (exporter that produces the .onnx)
# The manifest (checkpoints/models.yaml) is the in-repo POINTER; the build-time
# fetch (site/scripts/fetch_models.py) provisions the git-ignored .onnx from the
# Hub against it. A toy that references a model with no manifest entry, or an
# entry whose exporter is missing / does not produce the file, is a demo that can
# never come alive — so it FAILS this gate.
MANIFEST_REL = "checkpoints/models.yaml"
# Where site toys live and the /models/<name>.onnx references we hold them to.
SITE_SRC_REL = "site/src"
SITE_SRC_SUFFIXES = {".ts", ".tsx", ".astro", ".js", ".jsx"}
MODEL_URL_RE = re.compile(r"/models/(?P<file>[\w.-]+\.onnx)")


def _working_tree_basenames(root: Path) -> set[str]:
    """Every file basename in the working tree (minus vendored/VCS dirs)."""
    names: set[str] = set()
    for path in root.rglob("*"):
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        if path.is_file():
            names.add(path.name)
    return names


def _chapter_code_text(chapter: Chapter) -> str:
    """Concatenated text of every .py under the chapter dir (exporters live here)."""
    chunks: list[str] = []
    for py in chapter.directory.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            chunks.append(py.read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n".join(chunks)


def _resolves(basename: str, present: set[str], code: str, *, family: bool) -> bool:
    """A model basename resolves if it is a provisioned file or produced by code;
    with family=True, a trailing `_<qualifier>` may be stripped and re-tested."""
    if basename in present or basename in code:
        return True
    if not family:
        return False
    stem, dot, ext = basename.partition(".")
    while "_" in stem:
        stem = stem.rsplit("_", 1)[0]
        if _resolves(f"{stem}{dot}{ext}", present, code, family=False):
            return True
    return False


def scan_embed(
    embed: Path, present: set[str], code: str, where: str
) -> list[str]:
    """Failure messages for dangling (unresolved, un-PENDING) model refs."""
    failures: list[str] = []
    for lineno, line in enumerate(
        embed.read_text(encoding="utf-8").splitlines(), start=1
    ):
        code_part = line.split("#", 1)[0]
        m = ASSET_LINE_RE.match(code_part)
        if not m:
            continue
        asset = m.group("asset")
        basename = asset.rsplit("/", 1)[-1]
        if PENDING_RE.search(line):
            continue  # explicitly marked pending — not yet wired, and honest
        if _resolves(basename, present, code, family=True):
            continue
        failures.append(
            f"{where}:{lineno}: model '{asset}' is dangling — no provisioned "
            "file, no exporter in the chapter code, and not marked PENDING; the "
            "embed would silently fail to load it (site/CLAUDE.md). Mark it "
            "`# PENDING export` or wire the exporter."
        )
    return failures


def _load_manifest(root: Path) -> tuple[dict[str, dict] | None, str | None]:
    """Return ({site_path -> entry}, error). error is a message if the manifest is
    absent/unparseable, in which case the mapping is None."""
    manifest = root / MANIFEST_REL
    if not manifest.is_file():
        return None, f"no model manifest at {MANIFEST_REL}"
    try:
        data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return None, f"{MANIFEST_REL} does not parse: {exc}"
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        return None, f"{MANIFEST_REL} has no `models:` list"
    by_path: dict[str, dict] = {}
    for entry in data["models"]:
        if isinstance(entry, dict) and entry.get("site_path"):
            by_path[entry["site_path"]] = entry
    return by_path, None


def _exporter_produces(root: Path, entry: dict, filename: str) -> bool:
    """The manifest entry's exporter must exist and actually name the .onnx it
    claims to produce (the file itself, or any .py beside it in the chapter dir)."""
    exporter_rel = entry.get("exporter")
    if not exporter_rel:
        return False
    exporter = root / exporter_rel
    if not exporter.is_file():
        return False
    try:
        if filename in exporter.read_text(encoding="utf-8"):
            return True
    except OSError:
        return False
    # Self-exporting chapters split train/export across sibling .py files.
    for py in exporter.parent.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            if filename in py.read_text(encoding="utf-8"):
                return True
        except OSError:
            continue
    return False


def _iter_site_model_refs(root: Path):
    """Yield (where, lineno, model_url, filename) for every /models/<name>.onnx
    reference a site source file makes (the live toys' hard-coded MODEL_URL)."""
    src = root / SITE_SRC_REL
    if not src.is_dir():
        return
    for path in sorted(src.rglob("*")):
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        if not path.is_file() or path.suffix not in SITE_SRC_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for m in MODEL_URL_RE.finditer(line):
                yield path.relative_to(root).as_posix(), lineno, m.group(0), m.group("file")


def check_triangle(root: Path) -> list[str]:
    """Every model a live site toy loads must resolve through the manifest to a
    real exporter (site toy -> manifest entry -> producing exporter)."""
    refs = list(_iter_site_model_refs(root))
    if not refs:
        return []  # no live toy loads a model yet — nothing to hold to the triangle

    by_path, err = _load_manifest(root)
    if by_path is None:
        # Toys reference models but there is no usable manifest — every ref dangles.
        return [
            f"{where}:{lineno}: toy loads '{url}' but {err} — add a "
            f"{MANIFEST_REL} entry (site_path: {url}) so the build can fetch it."
            for where, lineno, url, _ in refs
        ]

    failures: list[str] = []
    for where, lineno, url, filename in refs:
        entry = by_path.get(url)
        if entry is None:
            failures.append(
                f"{where}:{lineno}: toy loads '{url}' but no entry in "
                f"{MANIFEST_REL} serves that path — the fetch cannot provision it "
                "and the demo can never come alive. Add a manifest entry."
            )
            continue
        if not _exporter_produces(root, entry, filename):
            failures.append(
                f"{where}:{lineno}: '{url}' has a {MANIFEST_REL} entry but its "
                f"exporter '{entry.get('exporter', '<missing>')}' does not exist "
                f"or does not produce '{filename}' — the manifest points at a "
                "model nothing builds. Fix the exporter path or the producer."
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail on dangling model-asset references in chapter "
        "demo/embed.yaml files (site/CLAUDE.md silent-broken-demo rule) and on a "
        "broken model-hosting triangle (site toy -> manifest -> exporter)."
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2]),
        help="repo root (default: auto-detected from this script's location)",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    try:
        chapters = discover_chapters(root)
    except ChapterError as exc:
        print(f"FAIL check_demo_assets: {exc}", file=sys.stderr)
        return 1

    if not chapters:
        print("check_demo_assets: no chapters discovered — OK")
        return 0

    present = _working_tree_basenames(root)

    failures: list[str] = []
    checked = 0
    for chapter in chapters:
        embed = chapter.directory / "demo" / "embed.yaml"
        if not embed.is_file():
            continue  # not every chapter ships a demo embed
        checked += 1
        where = embed.relative_to(root).as_posix()
        failures.extend(
            scan_embed(embed, present, _chapter_code_text(chapter), where)
        )

    triangle_failures = check_triangle(root)

    if failures or triangle_failures:
        if failures:
            print(
                f"FAIL check_demo_assets: {len(failures)} dangling model asset(s) "
                "in demo embeds — a reader would hit a broken demo "
                "(site/CLAUDE.md):",
                file=sys.stderr,
            )
            for failure in failures:
                print(f"  {failure}", file=sys.stderr)
        if triangle_failures:
            print(
                f"FAIL check_demo_assets: {len(triangle_failures)} broken "
                "model-hosting triangle edge(s) — a live toy loads a model the "
                f"build cannot provision ({MANIFEST_REL}):",
                file=sys.stderr,
            )
            for failure in triangle_failures:
                print(f"  {failure}", file=sys.stderr)
        return 1

    triangle_n = len(list(_iter_site_model_refs(root)))
    print(
        f"check_demo_assets: {checked} demo embed(s), every model reference "
        f"resolves or is marked PENDING; {triangle_n} live toy model ref(s) "
        f"trace site->manifest->exporter — OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
