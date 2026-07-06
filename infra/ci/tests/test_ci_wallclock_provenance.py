"""check_wallclock_provenance: wallclock.csv schema + provenance rules."""

import check_wallclock_provenance

HEADER = "chapter,tier,wallclock_min,config_hash,commit,date,status\n"


def run(tmp_path):
    return check_wallclock_provenance.main(["--root", str(tmp_path)])


def test_good_pending_and_measured_rows_pass(
    make_chapter, write_wallclock, tmp_path
):
    make_chapter(chapter_id="ch1.1-bc")
    write_wallclock(
        HEADER
        + "ch1.1-bc,cpu-laptop,,,,,PENDING\n"
        + "ch1.1-bc,t4,12.5,abc123def456,deadbee,2026-07-01,MEASURED\n"
    )
    assert run(tmp_path) == 0


def test_missing_csv_fails(make_chapter, tmp_path, capsys):
    make_chapter()
    assert run(tmp_path) == 1
    assert "does not exist" in capsys.readouterr().err


def test_wrong_header_fails(make_chapter, write_wallclock, tmp_path, capsys):
    make_chapter(chapter_id="ch1.1-bc")
    write_wallclock("chapter,tier,minutes,hash,commit,date,status\n")
    assert run(tmp_path) == 1
    assert "header" in capsys.readouterr().err


def test_wrong_field_count_fails(
    make_chapter, write_wallclock, tmp_path, capsys
):
    make_chapter(chapter_id="ch1.1-bc")
    write_wallclock(HEADER + "ch1.1-bc,cpu-laptop,,,,PENDING\n")
    assert run(tmp_path) == 1
    assert "expected 7 fields" in capsys.readouterr().err


def test_orphan_row_warns_but_passes(
    make_chapter, write_wallclock, tmp_path, capsys
):
    # The CSV may lead the chapter by one PR: unknown id is a WARNING.
    make_chapter(chapter_id="ch1.1-bc")
    write_wallclock(
        HEADER
        + "ch1.1-bc,cpu-laptop,,,,,PENDING\n"
        + "ch9.9-future,cpu-laptop,,,,,PENDING\n"
    )
    assert run(tmp_path) == 0
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "ch9.9-future" in err


def test_chapter_without_any_row_fails(
    make_chapter, write_wallclock, tmp_path, capsys
):
    make_chapter(chapter_id="ch1.1-bc")
    make_chapter(dirname="ch1.2_act", chapter_id="ch1.2-act")
    write_wallclock(HEADER + "ch1.1-bc,cpu-laptop,,,,,PENDING\n")
    assert run(tmp_path) == 1
    assert "ch1.2-act" in capsys.readouterr().err


def test_pending_row_with_data_fails(
    make_chapter, write_wallclock, tmp_path, capsys
):
    make_chapter(chapter_id="ch1.1-bc")
    write_wallclock(HEADER + "ch1.1-bc,cpu-laptop,10.0,,,,PENDING\n")
    assert run(tmp_path) == 1
    assert "PENDING row must have empty" in capsys.readouterr().err


def test_measured_row_missing_fields_fails(
    make_chapter, write_wallclock, tmp_path, capsys
):
    make_chapter(chapter_id="ch1.1-bc")
    write_wallclock(HEADER + "ch1.1-bc,t4,12.5,,deadbee,2026-07-01,MEASURED\n")
    assert run(tmp_path) == 1
    assert "config_hash" in capsys.readouterr().err


def test_measured_row_nonpositive_minutes_fails(
    make_chapter, write_wallclock, tmp_path, capsys
):
    make_chapter(chapter_id="ch1.1-bc")
    write_wallclock(HEADER + "ch1.1-bc,t4,0,abc,deadbee,2026-07-01,MEASURED\n")
    assert run(tmp_path) == 1
    assert "positive float" in capsys.readouterr().err


def test_measured_row_nonfloat_minutes_fails(
    make_chapter, write_wallclock, tmp_path
):
    make_chapter(chapter_id="ch1.1-bc")
    write_wallclock(
        HEADER + "ch1.1-bc,t4,fast,abc,deadbee,2026-07-01,MEASURED\n"
    )
    assert run(tmp_path) == 1


def test_measured_row_bad_date_fails(
    make_chapter, write_wallclock, tmp_path, capsys
):
    make_chapter(chapter_id="ch1.1-bc")
    write_wallclock(
        HEADER + "ch1.1-bc,t4,12.5,abc,deadbee,07/01/2026,MEASURED\n"
    )
    assert run(tmp_path) == 1
    assert "ISO" in capsys.readouterr().err


def test_measured_row_impossible_date_fails(
    make_chapter, write_wallclock, tmp_path
):
    make_chapter(chapter_id="ch1.1-bc")
    write_wallclock(
        HEADER + "ch1.1-bc,t4,12.5,abc,deadbee,2026-13-40,MEASURED\n"
    )
    assert run(tmp_path) == 1


def test_unknown_status_fails(make_chapter, write_wallclock, tmp_path, capsys):
    make_chapter(chapter_id="ch1.1-bc")
    write_wallclock(HEADER + "ch1.1-bc,t4,,,,,ESTIMATED\n")
    assert run(tmp_path) == 1
    assert "not PENDING or MEASURED" in capsys.readouterr().err


def test_blank_trailing_line_tolerated(
    make_chapter, write_wallclock, tmp_path
):
    make_chapter(chapter_id="ch1.1-bc")
    write_wallclock(HEADER + "ch1.1-bc,cpu-laptop,,,,,PENDING\n\n")
    assert run(tmp_path) == 0
