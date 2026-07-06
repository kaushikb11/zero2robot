"""check_loc_caps: 450 hard cap / 400 warning on chapter artifacts."""

import check_loc_caps


def artifact_of_lines(n: int) -> str:
    return "\n".join(f"x{i} = {i}" for i in range(n)) + "\n"


def test_no_chapters_passes(tmp_path, capsys):
    assert check_loc_caps.main(["--root", str(tmp_path)]) == 0
    assert "no chapters" in capsys.readouterr().out


def test_small_artifact_passes(make_chapter, tmp_path):
    make_chapter(artifact_text=artifact_of_lines(50))
    assert check_loc_caps.main(["--root", str(tmp_path)]) == 0


def test_at_hard_cap_passes_with_warning(make_chapter, tmp_path, capsys):
    make_chapter(artifact_text=artifact_of_lines(450))
    assert check_loc_caps.main(["--root", str(tmp_path)]) == 0
    assert "exceeds the 400-line target" in capsys.readouterr().err


def test_at_warn_threshold_no_warning(make_chapter, tmp_path, capsys):
    make_chapter(artifact_text=artifact_of_lines(400))
    assert check_loc_caps.main(["--root", str(tmp_path)]) == 0
    assert "WARNING" not in capsys.readouterr().err


def test_over_hard_cap_fails_listing_file_and_count(
    make_chapter, tmp_path, capsys
):
    make_chapter(artifact_text=artifact_of_lines(451))
    assert check_loc_caps.main(["--root", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "bc.py:451" in err
    assert "450" in err


def test_missing_artifact_fails(make_chapter, tmp_path, capsys):
    make_chapter(artifact_text=None)
    assert check_loc_caps.main(["--root", str(tmp_path)]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_meta_without_artifact_key_fails(tmp_path, capsys):
    chapter_dir = tmp_path / "curriculum" / "phase1_imitation" / "ch1.1_bc"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "meta.yaml").write_text("id: ch1.1-bc\n", encoding="utf-8")
    assert check_loc_caps.main(["--root", str(tmp_path)]) == 1
    assert "no 'artifact' key" in capsys.readouterr().err


def test_malformed_meta_yaml_fails(tmp_path, capsys):
    chapter_dir = tmp_path / "curriculum" / "phase1_imitation" / "ch1.1_bc"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "meta.yaml").write_text("id: [unclosed\n", encoding="utf-8")
    assert check_loc_caps.main(["--root", str(tmp_path)]) == 1
    assert "invalid YAML" in capsys.readouterr().err
