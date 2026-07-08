#!/usr/bin/env python3
"""upload_datasets_hf — publish the OPTIONAL dataset mirrors to the Hugging Face Hub.

Sibling of scripts/upload_models_hf.py. The zero2robot demo sets are all REGENERATED
locally from seeded generators (datasets/datasets.yaml, source: regenerate), so a
Hub mirror is a pure convenience: it lets a fresh Colab skip the (deterministic)
regeneration step. This script archives each mirrorable dataset, computes its sha256,
and uploads the archive + a dataset card to the hf_repo named in the manifest. It is
intentionally NOT run by CI or any build — binaries never touch git (root CLAUDE.md
#5), and pushing to the Hub is a deliberate, authenticated, author-gated action.

WHAT IT DOES (for the hf_repo in datasets/datasets.yaml):
  1. reads the manifest; for each `source: regenerate` entry that declares a `mirror:`
     block, locates the regenerated dataset on disk (its `path:`);
  2. tars it to <name>.tar.gz and computes the archive sha256;
  3. if the manifest already records a mirror.sha256, verifies the fresh archive
     matches it (never publish bytes that disagree with the pointer); if the manifest
     sha256 is null, PRINTS the computed digest so the author can paste it back;
  4. uploads each archive to the dataset repo, plus a per-dataset card
     dataset_cards/<name>.md and a top-level README.md index.

External (`source: fetch`, e.g. lerobot/pusht) and `reference-only` entries are
SKIPPED — we never re-host upstream data.

PREREQUISITES (author):
  - pip install huggingface_hub        (NOT a project dependency; author-local only)
  - huggingface-cli login              (or set HF_TOKEN in the environment)
  - the hf_repo in datasets/datasets.yaml must be a dataset repo you can push to.
    Create it once (public) — this script will also create it with --create-repo.
  - every mirrorable dataset must be regenerated locally first, e.g.
      .venv/bin/python curriculum/common/envs/pusht/gen_demos.py \
          --episodes 100 --seed 0 --out outputs/pusht-demos --no-video
    (see each manifest entry's `regenerate:` field).

USAGE:
  python scripts/upload_datasets_hf.py --dry-run           # default: archive + plan, print sha256, no push
  python scripts/upload_datasets_hf.py --create-repo --yes  # create repo + upload for real

Nothing uploads without --yes. --dry-run (the default) archives locally, prints the
plan + each computed sha256, and exits.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "datasets" / "datasets.yaml"
CHUNK = 1 << 16


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _archive(dataset_dir: Path, archive_name: str, dest_dir: Path) -> Path:
    """tar.gz the dataset dir (entries rooted at <name>/) into dest_dir/archive_name."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / archive_name
    with tarfile.open(out, "w:gz") as tar:
        tar.add(dataset_dir, arcname=dataset_dir.name)
    return out


def _dataset_card(entry: dict, sha: str) -> str:
    """A short per-dataset card documenting provenance + how to regenerate."""
    chapters = ", ".join(f"`{c}`" for c in entry.get("chapters", [])) or "?"
    return (
        f"# {entry['name']}\n\n"
        f"zero2robot demo dataset — an OPTIONAL convenience mirror. The primary path is "
        f"to regenerate it locally (deterministic, offline, free); this archive just "
        f"lets a fresh Colab skip that step.\n\n"
        f"| field | value |\n|---|---|\n"
        f"| chapters | {chapters} |\n"
        f"| format | `{entry.get('format', '?')}` |\n"
        f"| generator | `{entry.get('generator', '?')}` |\n"
        f"| seed | `{entry.get('seed', '?')}` |\n"
        f"| tier | `{entry.get('tier', '?')}` |\n"
        f"| archive sha256 | `{sha}` |\n\n"
        f"Provenance: {entry.get('provenance', '?')}\n\n"
        f"Regenerate (produces the same dataset the archive holds):\n\n"
        f"```bash\n{entry.get('regenerate', '?')}\n```\n"
    )


def _readme(hf_repo: str, plan: list[tuple[dict, Path, str]]) -> str:
    rows = "\n".join(
        f"| [`{m['archive']}`]({m['archive']}) | "
        f"{', '.join(f'`{c}`' for c in e.get('chapters', [])) or '?'} | "
        f"`{e.get('format', '?')}` | `{sha[:12]}…` |"
        for e, _, sha in plan
        for m in [e['mirror']]
    )
    return (
        f"# {hf_repo}\n\n"
        "Convenience mirrors of the [zero2robot](https://github.com/kaushikb11/"
        "zero2robot) course's demo datasets. Every archive here is REGENERATED from a "
        "seeded, deterministic generator in the repo — you never need these to take the "
        "course, they only let a fresh Colab skip regeneration. Each dataset card names "
        "the exact command that reproduces the archive's contents.\n\n"
        "| archive | chapters | format | sha256 |\n|---|---|---|---|\n"
        f"{rows}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes", action="store_true",
        help="actually upload (default is a dry-run plan with no network writes)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="archive + print the plan and each sha256, then exit (also the default)",
    )
    parser.add_argument(
        "--create-repo", action="store_true",
        help="create the hf_repo (public dataset repo) if it does not exist",
    )
    args = parser.parse_args(argv)
    do_upload = args.yes and not args.dry_run

    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    hf_repo = data["hf_repo"]
    entries: list[dict] = data["datasets"]

    workdir = Path(tempfile.mkdtemp(prefix="z2r-dataset-archives-"))
    plan: list[tuple[dict, Path, str]] = []   # (entry, archive_path, sha256)
    problems: list[str] = []
    skipped: list[str] = []

    for entry in entries:
        name = entry["name"]
        if entry.get("source") != "regenerate" or not entry.get("mirror"):
            reason = ("external upstream — not re-hosted" if entry.get("source") == "fetch"
                      else "reference-only" if entry.get("source") == "reference-only"
                      else "no mirror declared")
            skipped.append(f"{name}: {reason}")
            continue
        dataset_dir = REPO_ROOT / entry["path"]
        if not dataset_dir.exists():
            problems.append(
                f"{name}: missing {entry['path']} — regenerate first:\n      "
                f"{entry.get('regenerate', '?')}"
            )
            continue
        archive_path = _archive(dataset_dir, entry["mirror"]["archive"], workdir)
        sha = _sha256(archive_path)
        want = entry["mirror"].get("sha256")
        if want and want != sha:
            problems.append(
                f"{name}: archive sha256 {sha[:12]}… != manifest {str(want)[:12]}… — "
                "the local dataset differs from the recorded mirror. Regenerate at the "
                "manifest seed, or refresh mirror.sha256."
            )
            continue
        plan.append((entry, archive_path, sha))

    print(f"upload target: hf_repo = datasets/{hf_repo}")
    for entry, archive_path, sha in plan:
        marker = "sha OK" if entry["mirror"].get("sha256") else "NEW sha -> paste into manifest"
        print(f"  UPLOAD {entry['mirror']['archive']:26s} <- {entry['path']}")
        print(f"         sha256 {sha}  ({marker})")
        print(f"         + dataset_cards/{entry['name']}.md")
    print("  UPLOAD README.md                 <- generated index")
    for s in skipped:
        print(f"  SKIP   {s}")

    if problems:
        print("\nBLOCKED — fix these before uploading:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    if not plan:
        print("\nnothing to upload (no regenerated mirrorable datasets present).")
        return 0

    if not do_upload:
        print("\ndry-run: nothing uploaded. Paste any NEW sha256 above into "
              "datasets/datasets.yaml (mirror.sha256), then re-run with --yes "
              "(add --create-repo the first time) to publish.")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed — `pip install huggingface_hub` "
              "(author-local; not a project dependency).", file=sys.stderr)
        return 1

    api = HfApi()
    if args.create_repo:
        api.create_repo(hf_repo, repo_type="dataset", private=False, exist_ok=True)
        print(f"ensured dataset repo {hf_repo} exists")

    for entry, archive_path, sha in plan:
        api.upload_file(
            path_or_fileobj=str(archive_path),
            path_in_repo=entry["mirror"]["archive"],
            repo_id=hf_repo,
            repo_type="dataset",
            commit_message=f"upload {entry['name']} mirror ({sha[:12]})",
        )
        card = _dataset_card(entry, sha).encode("utf-8")
        api.upload_file(
            path_or_fileobj=card,
            path_in_repo=f"dataset_cards/{entry['name']}.md",
            repo_id=hf_repo,
            repo_type="dataset",
        )
        print(f"uploaded {entry['mirror']['archive']} + dataset card")

    api.upload_file(
        path_or_fileobj=_readme(hf_repo, plan).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=hf_repo,
        repo_type="dataset",
    )
    print(f"uploaded README.md — done. {len(plan)} mirror(s) live at "
          f"https://huggingface.co/datasets/{hf_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
