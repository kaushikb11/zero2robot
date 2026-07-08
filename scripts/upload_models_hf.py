#!/usr/bin/env python3
"""upload_models_hf — push the live-demo policies to the Hugging Face Hub.

THE AUTHOR'S ONE MANUAL STEP. The site's live demos load `/models/<name>.onnx`,
which site/scripts/fetch_models.py provisions at build time from the Hub against
checkpoints/models.yaml. Until the Hub repo exists and holds those files, every
fetch warns and the demos degrade to poster frames. This script populates the Hub
so the demos come alive. It is intentionally NOT run by CI or the build — binaries
never touch git (root CLAUDE.md #5), and pushing to the Hub is a deliberate,
authenticated, author-gated action.

WHAT IT DOES (for the hf_repo named in checkpoints/models.yaml):
  1. reads the manifest and locates each model's provisioned .onnx under
     <public_dir>/models/ (produced by that model's exporter — see the manifest);
  2. verifies each file's sha256 matches the manifest BEFORE upload (never publish
     bytes that disagree with the pointer — that would make every fetch re-download);
  3. uploads each .onnx to the repo root, plus a per-model card
     model_cards/<name>.md (chapter / seed / contract / tier / sha256) and a
     top-level README.md index — so the Hub page documents provenance.

PREREQUISITES (author):
  - pip install huggingface_hub        (NOT a project dependency; author-local only)
  - huggingface-cli login              (or set HF_TOKEN in the environment)
  - the hf_repo in checkpoints/models.yaml must be a repo you can push to. Create
    it once (public) — this script will also create it with --create-repo.
  - every model's .onnx must be present locally: run each exporter first, e.g.
      .venv/bin/python curriculum/phase1_imitation/ch1.1_bc/bc.py --seed 0
      .venv/bin/python curriculum/phase2_reinforcement/ch2.2_sac/export_sac_onnx.py
    (see each manifest entry's `exporter:` field), then refresh the manifest
    sha256 with `shasum -a 256 site/public/models/<name>.onnx`.

USAGE:
  python scripts/upload_models_hf.py --dry-run          # default: plan only, no push
  python scripts/upload_models_hf.py --create-repo --yes  # create repo + upload for real

Nothing uploads without --yes. --dry-run (the default) prints exactly what would
be pushed and exits.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "checkpoints" / "models.yaml"
CHUNK = 1 << 16


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _model_card(entry: dict) -> str:
    """A short per-model card documenting provenance.

    Opens with YAML frontmatter (the Hub's model-card metadata block) so the Hub
    stops warning "empty or missing yaml metadata in repo card" and renders the
    card's license/tags/library chips.
    """
    return (
        "---\n"
        "license: mit\n"
        "library_name: onnxruntime\n"
        "pipeline_tag: robotics\n"
        "tags:\n"
        "  - robotics\n"
        "  - onnx\n"
        "  - zero2robot\n"
        f"  - {entry.get('chapter', 'zero2robot')}\n"
        "---\n\n"
        f"# {entry['name']}\n\n"
        f"zero2robot live-demo policy. Loaded by the site at "
        f"`{entry['site_path']}` and provisioned at build time by "
        f"`site/scripts/fetch_models.py` (sha256-verified).\n\n"
        f"| field | value |\n|---|---|\n"
        f"| chapter | `{entry.get('chapter', '?')}` |\n"
        f"| contract | `{entry.get('contract', '?')}` |\n"
        f"| seed | `{entry.get('seed', '?')}` |\n"
        f"| tier | `{entry.get('tier', '?')}` |\n"
        f"| exporter | `{entry.get('exporter', '?')}` |\n"
        f"| sha256 | `{entry['sha256']}` |\n\n"
        f"Reproduce: run the exporter above at the recorded seed on the recorded "
        f"tier; the export is CPU-deterministic, so the bytes (and this sha256) "
        f"reproduce exactly.\n"
    )


def _readme(hf_repo: str, entries: list[dict]) -> str:
    rows = "\n".join(
        f"| [`{e['filename']}`]({e['filename']}) | `{e.get('chapter', '?')}` | "
        f"`{e.get('contract', '?')}` | `{e['sha256'][:12]}…` |"
        for e in entries
    )
    return (
        "---\n"
        "license: mit\n"
        "library_name: onnxruntime\n"
        "pipeline_tag: robotics\n"
        "tags:\n"
        "  - robotics\n"
        "  - onnx\n"
        "  - zero2robot\n"
        "  - imitation-learning\n"
        "  - reinforcement-learning\n"
        "---\n\n"
        f"# {hf_repo}\n\n"
        "ONNX policies that drive the [zero2robot](https://github.com/kaushikb11/"
        "zero2robot) course's live browser demos. Each file is the ONNX export of "
        "a from-scratch chapter policy; the site fetches these at build time "
        "(sha256-verified against `checkpoints/models.yaml`) and runs them in "
        "`onnxruntime-web`. No training required to read the course — these are the "
        "provisioned demo weights.\n\n"
        "| model | chapter | contract | sha256 |\n|---|---|---|---|\n"
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
        help="print the plan and exit (this is also the default when --yes is absent)",
    )
    parser.add_argument(
        "--create-repo", action="store_true",
        help="create the hf_repo (public model repo) if it does not exist",
    )
    args = parser.parse_args(argv)
    do_upload = args.yes and not args.dry_run

    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    hf_repo = data["hf_repo"]
    public_dir = REPO_ROOT / data.get("public_dir", "site/public")
    entries: list[dict] = data["models"]

    # Validate every file is present + matches the manifest BEFORE touching the Hub.
    plan: list[tuple[dict, Path]] = []
    problems: list[str] = []
    for entry in entries:
        local = public_dir / entry["site_path"].lstrip("/")
        if not local.is_file():
            problems.append(
                f"{entry['name']}: missing {local.relative_to(REPO_ROOT)} — run its "
                f"exporter ({entry.get('exporter', '?')}) first"
            )
            continue
        got = _sha256(local)
        if got != entry["sha256"]:
            problems.append(
                f"{entry['name']}: sha256 mismatch (manifest {entry['sha256'][:12]}…, "
                f"file {got[:12]}…) — re-export or refresh the manifest sha256"
            )
            continue
        plan.append((entry, local))

    print(f"upload target: hf_repo = {hf_repo}")
    for entry, local in plan:
        print(f"  UPLOAD {entry['filename']:24s} <- {local.relative_to(REPO_ROOT)} "
              f"(sha {entry['sha256'][:12]}…)  + model_cards/{entry['name']}.md")
    print("  UPLOAD README.md            <- generated index")

    if problems:
        print("\nBLOCKED — fix these before uploading:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    if not do_upload:
        print("\ndry-run: nothing uploaded. Re-run with --yes (add --create-repo "
              "the first time) to publish.")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed — `pip install huggingface_hub` "
              "(author-local; not a project dependency).", file=sys.stderr)
        return 1

    api = HfApi()
    if args.create_repo:
        api.create_repo(hf_repo, repo_type="model", private=False, exist_ok=True)
        print(f"ensured repo {hf_repo} exists")

    for entry, local in plan:
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=entry["filename"],
            repo_id=hf_repo,
            repo_type="model",
            commit_message=f"upload {entry['name']} ({entry['chapter']}, {entry['sha256'][:12]})",
        )
        card = _model_card(entry).encode("utf-8")
        api.upload_file(
            path_or_fileobj=card,
            path_in_repo=f"model_cards/{entry['name']}.md",
            repo_id=hf_repo,
            repo_type="model",
        )
        print(f"uploaded {entry['filename']} + model card")

    api.upload_file(
        path_or_fileobj=_readme(hf_repo, entries).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=hf_repo,
        repo_type="model",
    )
    print(f"uploaded README.md — done. {len(plan)} model(s) live at "
          f"https://huggingface.co/{hf_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
