"""check_forbidden_imports: pedagogy_gate-aligned import blocking."""

import check_forbidden_imports

CLEAN = "import torch\nimport numpy as np\nfrom pathlib import Path\n"


def test_clean_chapter_passes(make_chapter, tmp_path):
    make_chapter(artifact_text=CLEAN)
    assert check_forbidden_imports.main(["--root", str(tmp_path)]) == 0


def test_no_chapters_passes(tmp_path):
    assert check_forbidden_imports.main(["--root", str(tmp_path)]) == 0


def test_top_level_import_hydra_fails(make_chapter, tmp_path, capsys):
    make_chapter(artifact_text="import torch\nimport hydra\n")
    assert check_forbidden_imports.main(["--root", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "bc.py:2" in err
    assert "hydra" in err


def test_from_import_gymnasium_fails(make_chapter, tmp_path, capsys):
    make_chapter(artifact_text="from gymnasium import spaces\n")
    assert check_forbidden_imports.main(["--root", str(tmp_path)]) == 1
    assert "gymnasium" in capsys.readouterr().err


def test_function_level_import_fails(make_chapter, tmp_path, capsys):
    text = "def make_env():\n    import gym\n    return gym\n"
    make_chapter(artifact_text=text)
    assert check_forbidden_imports.main(["--root", str(tmp_path)]) == 1
    assert "bc.py:2" in capsys.readouterr().err


def test_word_boundary_does_not_flag_prefixes(make_chapter, tmp_path):
    # 'gymnastics' must not trip the 'gym' rule; 'hydralib' not 'hydra'.
    make_chapter(artifact_text="import gymnastics\nimport hydralib\n")
    assert check_forbidden_imports.main(["--root", str(tmp_path)]) == 0


def test_transformers_blocked_without_grant(make_chapter, tmp_path, capsys):
    make_chapter(artifact_text="from transformers import AutoModel\n")
    assert check_forbidden_imports.main(["--root", str(tmp_path)]) == 1
    assert "allow_transformers" in capsys.readouterr().err


def test_transformers_allowed_with_grant(make_chapter, tmp_path):
    make_chapter(
        artifact_text="from transformers import AutoModel\n",
        meta={"allow_transformers": True},
    )
    assert check_forbidden_imports.main(["--root", str(tmp_path)]) == 0


def test_sibling_chapter_code_also_scanned(make_chapter, tmp_path, capsys):
    chapter_dir = make_chapter(artifact_text=CLEAN)
    (chapter_dir / "helpers.py").write_text(
        "import omegaconf\n", encoding="utf-8"
    )
    assert check_forbidden_imports.main(["--root", str(tmp_path)]) == 1
    assert "helpers.py:1" in capsys.readouterr().err


def test_exercise_files_are_scanned(make_chapter, tmp_path, capsys):
    # Exercise code under the chapter dir IS scanned (canonical ex1_*.py names
    # carry digits; forbidden imports there must be caught too).
    chapter_dir = make_chapter(artifact_text=CLEAN)
    exercises = chapter_dir / "exercises"
    exercises.mkdir()
    (exercises / "ex1_bughunt.py").write_text("import gym\n", encoding="utf-8")
    assert check_forbidden_imports.main(["--root", str(tmp_path)]) == 1
    assert "exercises/ex1_bughunt.py:1" in capsys.readouterr().err


def test_digit_named_artifact_is_scanned(make_chapter, tmp_path, capsys):
    # A chapter .py whose name carries a digit (rl_v2.py) must be scanned.
    make_chapter(artifact_name="rl_v2.py", artifact_text="import hydra\n")
    assert check_forbidden_imports.main(["--root", str(tmp_path)]) == 1
    assert "rl_v2.py:1" in capsys.readouterr().err


def test_tests_subdir_not_scanned(make_chapter, tmp_path):
    # tests/ is human-owned and off-limits — never scanned.
    chapter_dir = make_chapter(artifact_text=CLEAN)
    tests_dir = chapter_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_thing.py").write_text("import gym\n", encoding="utf-8")
    assert check_forbidden_imports.main(["--root", str(tmp_path)]) == 0
