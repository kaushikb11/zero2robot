"""check_notebook_hashes: notebook <-> chapter artifact hash sync."""

import hashlib

import check_notebook_hashes


def hash_of(chapter_dir, artifact_name="bc.py") -> str:
    return hashlib.sha256((chapter_dir / artifact_name).read_bytes()).hexdigest()


def put_notebook(tmp_path, chapter_id="ch1.1-bc"):
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir(exist_ok=True)
    (notebooks / f"{chapter_id}.ipynb").write_text("{}", encoding="utf-8")


def set_hash(chapter_dir, value: str):
    meta = chapter_dir / "meta.yaml"
    meta.write_text(
        meta.read_text(encoding="utf-8") + f"notebook_hash: {value}\n",
        encoding="utf-8",
    )


def test_chapter_without_hash_skipped_with_note(
    make_chapter, tmp_path, capsys
):
    make_chapter()
    assert check_notebook_hashes.main(["--root", str(tmp_path)]) == 0
    assert "not yet generated" in capsys.readouterr().out


def test_matching_hash_and_notebook_pass(make_chapter, tmp_path):
    chapter_dir = make_chapter()
    set_hash(chapter_dir, hash_of(chapter_dir))
    put_notebook(tmp_path)
    assert check_notebook_hashes.main(["--root", str(tmp_path)]) == 0


def test_stale_hash_fails(make_chapter, tmp_path, capsys):
    chapter_dir = make_chapter()
    set_hash(chapter_dir, "0" * 64)
    put_notebook(tmp_path)
    assert check_notebook_hashes.main(["--root", str(tmp_path)]) == 1
    assert "STALE" in capsys.readouterr().err


def test_missing_notebook_file_fails(make_chapter, tmp_path, capsys):
    chapter_dir = make_chapter()
    set_hash(chapter_dir, hash_of(chapter_dir))
    assert check_notebook_hashes.main(["--root", str(tmp_path)]) == 1
    assert "ch1.1-bc.ipynb" in capsys.readouterr().err


def test_missing_artifact_fails(make_chapter, tmp_path, capsys):
    chapter_dir = make_chapter(artifact_text=None)
    set_hash(chapter_dir, "0" * 64)
    put_notebook(tmp_path)
    assert check_notebook_hashes.main(["--root", str(tmp_path)]) == 1
    assert "missing" in capsys.readouterr().err


def test_no_chapters_passes(tmp_path):
    assert check_notebook_hashes.main(["--root", str(tmp_path)]) == 0


# --- notebook_file_hash: detect hand-edited notebooks (#6) ------------------


def nb_hash(tmp_path, chapter_id="ch1.1-bc") -> str:
    return hashlib.sha256(
        (tmp_path / "notebooks" / f"{chapter_id}.ipynb").read_bytes()
    ).hexdigest()


def set_meta_line(chapter_dir, key, value):
    meta = chapter_dir / "meta.yaml"
    meta.write_text(
        meta.read_text(encoding="utf-8") + f"{key}: {value}\n", encoding="utf-8"
    )


def test_matching_notebook_file_hash_passes(make_chapter, tmp_path):
    chapter_dir = make_chapter()
    put_notebook(tmp_path)
    set_meta_line(chapter_dir, "notebook_file_hash", nb_hash(tmp_path))
    assert check_notebook_hashes.main(["--root", str(tmp_path)]) == 0


def test_hand_edited_notebook_file_hash_fails(make_chapter, tmp_path, capsys):
    chapter_dir = make_chapter()
    put_notebook(tmp_path)
    set_meta_line(chapter_dir, "notebook_file_hash", nb_hash(tmp_path))
    # Simulate a hand-edit of the notebook bytes after the hash was recorded.
    (tmp_path / "notebooks" / "ch1.1-bc.ipynb").write_text(
        '{"hand": "edited"}', encoding="utf-8"
    )
    assert check_notebook_hashes.main(["--root", str(tmp_path)]) == 1
    assert "HAND-EDITED" in capsys.readouterr().err


def test_notebook_file_hash_missing_notebook_fails(make_chapter, tmp_path, capsys):
    chapter_dir = make_chapter()
    set_meta_line(chapter_dir, "notebook_file_hash", "0" * 64)
    assert check_notebook_hashes.main(["--root", str(tmp_path)]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_chapter_with_neither_hash_skipped(make_chapter, tmp_path, capsys):
    make_chapter()
    assert check_notebook_hashes.main(["--root", str(tmp_path)]) == 0
    assert "not yet generated" in capsys.readouterr().out
