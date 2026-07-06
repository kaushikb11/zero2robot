"""gen_notebook: deterministic generation + comment-preserving meta writes.

Runs against tiny synthetic chapter trees (conftest make_chapter), never the
real curriculum/ tree. Kernel-free: we only build/serialize notebooks, never
execute them (the ci-notebook lane does that).
"""

import hashlib

import gen_notebook
import nbformat


def _run(tmp_path, *chapters):
    return gen_notebook.main(["--root", str(tmp_path), *chapters])


def test_generates_valid_deterministic_notebook(make_chapter, tmp_path):
    make_chapter()
    assert _run(tmp_path, "ch1.1-bc") == 0
    nb_path = tmp_path / "notebooks" / "ch1.1-bc.ipynb"
    assert nb_path.is_file()

    first = nb_path.read_bytes()
    # Valid nbformat v4, no embedded outputs, stable cell ids.
    nb = nbformat.reads(first.decode("utf-8"), as_version=4)
    nbformat.validate(nb)
    assert [c["id"] for c in nb.cells] == [f"cell-{i}" for i in range(len(nb.cells))]
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None

    # Regeneration is byte-identical (deterministic output).
    assert _run(tmp_path, "ch1.1-bc") == 0
    assert nb_path.read_bytes() == first


def test_region_becomes_a_code_cell(make_chapter, tmp_path):
    make_chapter()
    _run(tmp_path, "ch1.1-bc")
    nb = nbformat.read(
        tmp_path / "notebooks" / "ch1.1-bc.ipynb", as_version=4
    )
    sources = "".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"]) for c in nb.cells)
    # The DETERMINISTIC_ARTIFACT's `model` region body is emitted verbatim.
    assert "def loss_for_seed(seed):" in sources
    # Injected scaffolding is present.
    assert "Z2R_PROFILE" in sources
    assert "wallclock.render_line" in sources


def test_meta_records_both_hashes(make_chapter, tmp_path):
    chapter_dir = make_chapter()
    _run(tmp_path, "ch1.1-bc")

    import yaml

    meta = yaml.safe_load((chapter_dir / "meta.yaml").read_text())
    artifact_hash = hashlib.sha256((chapter_dir / "bc.py").read_bytes()).hexdigest()
    nb_bytes = (tmp_path / "notebooks" / "ch1.1-bc.ipynb").read_bytes()
    assert meta["notebook_hash"] == artifact_hash
    assert meta["notebook_file_hash"] == hashlib.sha256(nb_bytes).hexdigest()


def test_meta_write_is_idempotent_and_comment_preserving(make_chapter, tmp_path):
    chapter_dir = make_chapter(
        meta={"note_key": "keep me"},
    )
    meta_path = chapter_dir / "meta.yaml"
    # Add a human comment the generator must not clobber.
    meta_path.write_text(
        meta_path.read_text() + "# human comment stays\n", encoding="utf-8"
    )

    _run(tmp_path, "ch1.1-bc")
    after_first = meta_path.read_text()
    _run(tmp_path, "ch1.1-bc")
    after_second = meta_path.read_text()

    assert after_first == after_second  # idempotent
    assert "# human comment stays" in after_second  # comment preserved
    assert after_second.count("notebook_hash:") == 1  # no duplicate keys
    assert after_second.count("notebook_file_hash:") == 1


def test_unknown_chapter_fails(make_chapter, tmp_path, capsys):
    make_chapter()
    assert _run(tmp_path, "ch9.9-nope") == 1
    assert "unknown chapter" in capsys.readouterr().err


def test_all_flag_regenerates_only_generated_chapters(make_chapter, tmp_path):
    make_chapter()
    # Not generated yet -> --all is a no-op.
    assert gen_notebook.main(["--root", str(tmp_path), "--all"]) == 0
    assert not (tmp_path / "notebooks" / "ch1.1-bc.ipynb").exists()

    # After a first generation the chapter carries notebook_hash, so --all picks
    # it up and reproduces byte-identical output.
    _run(tmp_path, "ch1.1-bc")
    before = (tmp_path / "notebooks" / "ch1.1-bc.ipynb").read_bytes()
    assert gen_notebook.main(["--root", str(tmp_path), "--all"]) == 0
    assert (tmp_path / "notebooks" / "ch1.1-bc.ipynb").read_bytes() == before
