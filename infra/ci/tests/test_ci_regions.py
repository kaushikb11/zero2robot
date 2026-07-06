"""Region marker parser edge cases (lib/regions.py, site/CLAUDE.md convention)."""

import hashlib

import pytest
from lib.regions import RegionError, parse_regions, region_sha256


def test_extracts_named_regions_exact_text():
    text = (
        "import torch\n"
        "# --- region: model ---\n"
        "class Net:\n"
        "    pass\n"
        "# --- endregion ---\n"
        "# --- region: train_loop ---\n"
        "for step in range(3):\n"
        "    pass\n"
        "# --- endregion ---\n"
        "print('done')\n"
    )
    regions = parse_regions(text)
    assert regions == {
        "model": "class Net:\n    pass\n",
        "train_loop": "for step in range(3):\n    pass\n",
    }


def test_region_text_excludes_markers_and_preserves_blank_lines():
    text = "# --- region: a ---\n\nx = 1\n\n# --- endregion ---\n"
    assert parse_regions(text)["a"] == "\nx = 1\n\n"


def test_indented_markers_and_hyphenated_names():
    text = "    # --- region: data-loader ---\n    x = 1\n    # --- endregion ---\n"
    assert parse_regions(text) == {"data-loader": "    x = 1\n"}


def test_no_regions_is_empty_dict():
    assert parse_regions("x = 1\n") == {}


def test_unclosed_region_raises():
    with pytest.raises(RegionError, match="never closed"):
        parse_regions("# --- region: model ---\nx = 1\n", source="bc.py")


def test_nested_region_raises():
    text = (
        "# --- region: outer ---\n"
        "# --- region: inner ---\n"
        "# --- endregion ---\n"
        "# --- endregion ---\n"
    )
    with pytest.raises(RegionError, match="inside open region"):
        parse_regions(text)


def test_duplicate_region_name_raises():
    text = (
        "# --- region: model ---\n"
        "# --- endregion ---\n"
        "# --- region: model ---\n"
        "# --- endregion ---\n"
    )
    with pytest.raises(RegionError, match="duplicate region name 'model'"):
        parse_regions(text)


def test_stray_endregion_raises():
    with pytest.raises(RegionError, match="no open region"):
        parse_regions("x = 1\n# --- endregion ---\n")


def test_error_names_source_and_line():
    with pytest.raises(RegionError, match=r"act\.py:2"):
        parse_regions("x = 1\n# --- endregion ---\n", source="act.py")


def test_region_sha256_matches_hashlib():
    text = "x = 1\n"
    assert region_sha256(text) == hashlib.sha256(text.encode()).hexdigest()
