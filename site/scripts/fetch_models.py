#!/usr/bin/env python3
"""fetch_models — provision the live-demo .onnx policies at site build time.

The trained policies the site's live toys load (`/models/<name>.onnx`) are
git-ignored binaries (root CLAUDE.md #5: no binaries in git). On a fresh checkout
or a deploy runner they simply are not there, so every live demo degrades to its
poster frame. This script closes that gap WITHOUT git: it reads the in-repo
pointer manifest (checkpoints/models.yaml) and, for each model, provisions
site/public/models/<name>.onnx from the Hugging Face Hub, sha256-verified against
the manifest.

Per entry:
  - present AND sha256 matches the manifest  -> SKIP (already provisioned; idempotent)
  - absent, OR present but sha256 mismatches  -> download from the Hub, verify sha256,
    then atomically move into place
  - any failure (Hub not populated yet, network down, checksum mismatch) -> WARN and
    continue. NEVER hard-fail the build: the demo shows its poster frame (site/CLAUDE.md)
    and the reader still gets prose + code. A broken fetch must not break the docs.

Wired into `npm run prebuild` next to sync:hashes. Run by hand:

    .venv/bin/python site/scripts/fetch_models.py            # provision what's missing/stale
    .venv/bin/python site/scripts/fetch_models.py --check    # report only, download nothing

Env knobs:
    Z2R_MODELS_BASEURL   override the Hub base (default https://huggingface.co)
    Z2R_SKIP_MODEL_FETCH set to 1 to skip entirely (offline builds; demos degrade)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "checkpoints" / "models.yaml"
DEFAULT_BASEURL = "https://huggingface.co"
CHUNK = 1 << 16


def _warn(msg: str) -> None:
    print(f"fetch_models: WARN {msg}", file=sys.stderr)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest() -> dict | None:
    """Parse the manifest. A missing/broken manifest is a WARN, not a build break."""
    if not MANIFEST.is_file():
        _warn(f"no manifest at {MANIFEST.relative_to(REPO_ROOT)} — nothing to provision")
        return None
    try:
        data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        _warn(f"could not parse {MANIFEST.relative_to(REPO_ROOT)}: {exc}")
        return None
    if not isinstance(data, dict) or not data.get("models"):
        _warn(f"{MANIFEST.relative_to(REPO_ROOT)} has no `models:` list — nothing to do")
        return None
    return data


def _hub_url(baseurl: str, hf_repo: str, filename: str) -> str:
    return f"{baseurl.rstrip('/')}/{hf_repo}/resolve/main/{filename}"


def _download(url: str, dest_dir: Path) -> Path:
    """Download `url` to a temp file in dest_dir. Raises on any HTTP/network error."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".onnx.part", dir=dest_dir)
    tmp = Path(tmp_name)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "zero2robot-fetch-models"})
        with urllib.request.urlopen(req, timeout=60) as resp, os.fdopen(fd, "wb") as out:
            while chunk := resp.read(CHUNK):
                out.write(chunk)
        return tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report present/stale/missing per model; download nothing",
    )
    args = parser.parse_args(argv)

    if os.environ.get("Z2R_SKIP_MODEL_FETCH") == "1":
        print("fetch_models: Z2R_SKIP_MODEL_FETCH=1 — skipping (demos degrade to posters)")
        return 0

    data = _load_manifest()
    if data is None:
        return 0  # WARN already printed; never hard-fail the build

    hf_repo = data.get("hf_repo", "")
    public_dir = REPO_ROOT / data.get("public_dir", "site/public")
    baseurl = os.environ.get("Z2R_MODELS_BASEURL", DEFAULT_BASEURL)

    skipped = fetched = failed = 0
    for entry in data["models"]:
        name = entry.get("name", "<unnamed>")
        filename = entry["filename"]
        want_sha = entry["sha256"]
        # site_path is like "/models/bc_policy.onnx"; local path = public_dir + it.
        local = public_dir / entry["site_path"].lstrip("/")

        if local.is_file() and _sha256(local) == want_sha:
            print(f"  SKIP  {name:20s} present + sha256 match")
            skipped += 1
            continue

        if local.is_file():
            _warn(f"{name}: on-disk sha256 mismatch — will re-fetch")

        if args.check:
            state = "STALE" if local.is_file() else "MISSING"
            print(f"  {state:6s}{name:20s} would fetch from Hub")
            continue

        url = _hub_url(baseurl, hf_repo, filename)
        try:
            tmp = _download(url, local.parent)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
            _warn(f"{name}: fetch failed ({exc}) — demo degrades to its poster frame")
            failed += 1
            continue

        got_sha = _sha256(tmp)
        if got_sha != want_sha:
            tmp.unlink(missing_ok=True)
            _warn(
                f"{name}: sha256 mismatch after download "
                f"(manifest {want_sha[:12]}…, got {got_sha[:12]}…) — discarding; "
                "the served bytes do not match the manifest. Re-run the exporter "
                "and refresh the manifest sha256, or re-upload to the Hub."
            )
            failed += 1
            continue

        tmp.replace(local)
        print(f"  FETCH {name:20s} provisioned {local.relative_to(REPO_ROOT)}")
        fetched += 1

    verb = "would provision" if args.check else "provisioned"
    print(
        f"fetch_models: {len(data['models'])} model(s) — "
        f"{skipped} present, {fetched} {verb}, {failed} unavailable (poster fallback)"
    )
    return 0  # graceful degradation: build proceeds regardless


if __name__ == "__main__":
    raise SystemExit(main())
