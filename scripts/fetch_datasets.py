#!/usr/bin/env python3
"""fetch_datasets — provision the chapter demo datasets at build/setup time.

Sibling of site/scripts/fetch_models.py. The datasets the chapters load
(outputs/<name>) are git-ignored (root CLAUDE.md #5: no binaries in git). On a
fresh checkout they are simply not there. This script reads the in-repo pointer
manifest (datasets/datasets.yaml) and, per dataset, either confirms it is present,
fetches its optional Hugging Face mirror (sha256-verified), or — for the seeded
`regenerate` sets, which are the primary path — prints the exact command to
rebuild it locally. It NEVER hard-fails: a chapter regenerates its own data on
first run, so a missing dataset is a note, not a blocker.

Per manifest entry, keyed off `source:`:
  regenerate  present on disk                  -> SKIP
              missing, has a Hub mirror w/ sha -> download the archive, verify
                                                  sha256, unpack into place
              missing, no usable mirror        -> print the `regenerate:` command
                                                  (deterministic, offline, free)
  fetch       external upstream dataset        -> print the `fetch:` command (we do
                                                  not re-host or auto-pull it)
  reference-only                               -> SKIP with its note (never fetched)

Any failure (Hub not populated, network down, checksum mismatch) -> WARN and
continue. A broken fetch must not break a build or a learner's setup.

Run by hand:
    .venv/bin/python scripts/fetch_datasets.py            # provision what's missing/stale
    .venv/bin/python scripts/fetch_datasets.py --check    # report only, download nothing

Env knobs:
    Z2R_DATASETS_BASEURL   override the Hub base (default https://huggingface.co)
    Z2R_SKIP_DATASET_FETCH set to 1 to skip entirely (offline; regenerate on demand)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "datasets" / "datasets.yaml"
DEFAULT_BASEURL = "https://huggingface.co"
CHUNK = 1 << 16


def _warn(msg: str) -> None:
    print(f"fetch_datasets: WARN {msg}", file=sys.stderr)


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
    if not isinstance(data, dict) or not data.get("datasets"):
        _warn(f"{MANIFEST.relative_to(REPO_ROOT)} has no `datasets:` list — nothing to do")
        return None
    return data


def _present(path: Path) -> bool:
    """A LeRobot dataset dir is 'present' if it has meta/info.json; an .npz if the
    file exists; otherwise a non-empty dir/file counts. Keep it forgiving."""
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    if (path / "meta" / "info.json").is_file():
        return True
    return any(path.iterdir())


def _hub_archive_url(baseurl: str, hf_repo: str, archive: str) -> str:
    # datasets resolve under /datasets/<repo>/resolve/main/<file>
    return f"{baseurl.rstrip('/')}/datasets/{hf_repo}/resolve/main/{archive}"


def _download(url: str, dest_dir: Path) -> Path:
    """Download `url` to a temp file in dest_dir. Raises on any HTTP/network error."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".tar.gz.part", dir=dest_dir)
    tmp = Path(tmp_name)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "zero2robot-fetch-datasets"})
        with urllib.request.urlopen(req, timeout=120) as resp, os.fdopen(fd, "wb") as out:
            while chunk := resp.read(CHUNK):
                out.write(chunk)
        return tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract `archive` under `dest`, refusing any member that escapes it."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(f"unsafe path in archive: {member.name}")
        tar.extractall(dest)  # noqa: S202 — members validated above


def _fetch_mirror(entry: dict, baseurl: str, hf_repo: str) -> bool:
    """Download + verify + unpack a regenerate-set's optional Hub mirror. Returns
    True on success. Any failure WARNs and returns False (caller falls back to regen)."""
    mirror = entry.get("mirror")
    if not mirror or not mirror.get("archive"):
        return False
    want_sha = mirror.get("sha256")
    if not want_sha:
        return False  # no published mirror yet — regenerate instead
    archive = mirror["archive"]
    unpack_to = REPO_ROOT / mirror.get("unpack_to", f"outputs/{entry['name']}")
    url = _hub_archive_url(baseurl, hf_repo, archive)
    try:
        tmp = _download(url, unpack_to.parent)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        _warn(f"{entry['name']}: mirror fetch failed ({exc}) — regenerate instead")
        return False
    got = _sha256(tmp)
    if got != want_sha:
        tmp.unlink(missing_ok=True)
        _warn(
            f"{entry['name']}: archive sha256 mismatch "
            f"(manifest {want_sha[:12]}…, got {got[:12]}…) — discarding; re-upload the "
            "mirror or refresh the manifest sha256. Regenerate to proceed."
        )
        return False
    try:
        _safe_extract(tmp, unpack_to)
    except (tarfile.TarError, ValueError, OSError) as exc:
        _warn(f"{entry['name']}: could not unpack mirror ({exc}) — regenerate instead")
        return False
    finally:
        tmp.unlink(missing_ok=True)
    print(f"  FETCH  {entry['name']:18s} mirror unpacked -> {unpack_to.relative_to(REPO_ROOT)}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report present/missing per dataset; download nothing",
    )
    args = parser.parse_args(argv)

    if os.environ.get("Z2R_SKIP_DATASET_FETCH") == "1":
        print("fetch_datasets: Z2R_SKIP_DATASET_FETCH=1 — skipping (regenerate on demand)")
        return 0

    data = _load_manifest()
    if data is None:
        return 0  # WARN already printed; never hard-fail

    hf_repo = data.get("hf_repo", "")
    baseurl = os.environ.get("Z2R_DATASETS_BASEURL", DEFAULT_BASEURL)

    present = fetched = regen = external = refonly = 0
    for entry in data["datasets"]:
        name = entry.get("name", "<unnamed>")
        source = entry.get("source", "regenerate")

        if source == "reference-only":
            print(f"  REF    {name:18s} discussed at scale, never fetched")
            refonly += 1
            continue

        if source == "fetch":
            # External upstream dataset (e.g. lerobot/pusht). We do not auto-pull it.
            print(f"  EXTERN {name:18s} upstream — fetch with:\n"
                  f"           {entry.get('fetch', '(see manifest note)')}")
            external += 1
            continue

        # source == regenerate
        local = REPO_ROOT / entry["path"]
        if _present(local):
            print(f"  SKIP   {name:18s} present at {entry['path']}")
            present += 1
            continue

        if args.check:
            print(f"  MISS   {name:18s} would fetch mirror or regenerate")
            continue

        if _fetch_mirror(entry, baseurl, hf_repo):
            fetched += 1
            continue

        # No usable mirror — the deterministic regenerate command IS the fallback.
        print(f"  REGEN  {name:18s} not on disk; regenerate (deterministic, offline):\n"
              f"           {entry['regenerate']}")
        regen += 1

    print(
        f"fetch_datasets: {len(data['datasets'])} dataset(s) — "
        f"{present} present, {fetched} fetched, {regen} to regenerate, "
        f"{external} external, {refonly} reference-only"
    )
    return 0  # graceful degradation: setup/build proceeds regardless


if __name__ == "__main__":
    raise SystemExit(main())
