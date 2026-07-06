"""check_prose_code_drift: region hash verification against meta.yaml."""

import check_prose_code_drift
from lib.regions import region_sha256

ARTIFACT = (
    "import torch\n"
    "# --- region: model ---\n"
    "class Net:\n"
    "    pass\n"
    "# --- endregion ---\n"
)
MODEL_TEXT = "class Net:\n    pass\n"


def test_chapter_without_region_hashes_passes(make_chapter, tmp_path):
    make_chapter(artifact_text=ARTIFACT)
    assert check_prose_code_drift.main(["--root", str(tmp_path)]) == 0


def test_matching_hashes_pass(make_chapter, tmp_path):
    make_chapter(
        artifact_text=ARTIFACT,
        meta={"region_hashes": {"model": region_sha256(MODEL_TEXT)}},
    )
    assert check_prose_code_drift.main(["--root", str(tmp_path)]) == 0


def test_drifted_region_fails(make_chapter, tmp_path, capsys):
    make_chapter(
        artifact_text=ARTIFACT,
        meta={"region_hashes": {"model": region_sha256("class Old:\n")}},
    )
    assert check_prose_code_drift.main(["--root", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "drifted" in err
    assert "model" in err


def test_recorded_region_missing_from_artifact_fails(
    make_chapter, tmp_path, capsys
):
    make_chapter(
        artifact_text=ARTIFACT,
        meta={
            "region_hashes": {
                "model": region_sha256(MODEL_TEXT),
                "train_loop": region_sha256("gone\n"),
            }
        },
    )
    assert check_prose_code_drift.main(["--root", str(tmp_path)]) == 1
    assert "train_loop" in capsys.readouterr().err


def test_malformed_markers_fail_even_without_region_hashes(
    make_chapter, tmp_path, capsys
):
    make_chapter(artifact_text="# --- region: model ---\nx = 1\n")
    assert check_prose_code_drift.main(["--root", str(tmp_path)]) == 1
    assert "never closed" in capsys.readouterr().err


def test_duplicate_region_names_fail(make_chapter, tmp_path, capsys):
    text = (
        "# --- region: model ---\n# --- endregion ---\n"
        "# --- region: model ---\n# --- endregion ---\n"
    )
    make_chapter(artifact_text=text)
    assert check_prose_code_drift.main(["--root", str(tmp_path)]) == 1
    assert "duplicate" in capsys.readouterr().err


def test_missing_artifact_fails(make_chapter, tmp_path, capsys):
    make_chapter(artifact_text=None)
    assert check_prose_code_drift.main(["--root", str(tmp_path)]) == 1
    assert "missing" in capsys.readouterr().err


def test_non_mapping_region_hashes_fails(make_chapter, tmp_path, capsys):
    make_chapter(artifact_text=ARTIFACT, meta={"region_hashes": ["model"]})
    assert check_prose_code_drift.main(["--root", str(tmp_path)]) == 1
    assert "must be a mapping" in capsys.readouterr().err


def test_no_chapters_passes(tmp_path):
    assert check_prose_code_drift.main(["--root", str(tmp_path)]) == 0
