"""Tests for the shared CSV reader behind both bulk imports.

These pin the boring parts (delimiter, BOM, row numbers, blank lines) at the
source, so a regression there fails here instead of in a dozen endpoint tests.
"""

from app.api.schemas.csv_import import CsvImportProblem, SiteImportRow
from app.services.csv_import import drop_problem_rows, read_csv_rows

REQUIRED = ("name", "latitude")
OPTIONAL = ("notes",)


def _read(text: str, encoding: str = "utf-8"):
    return read_csv_rows(text.encode(encoding), REQUIRED, OPTIONAL, "site")


def test_delimiter_defaults_to_comma():
    rows, problems = _read("name,latitude,notes\nCAM01,52.1,hello\n")
    assert problems == []
    assert rows[0].values == {"name": "CAM01", "latitude": "52.1", "notes": "hello"}


def test_delimiter_is_semicolon_when_the_header_has_no_comma():
    rows, problems = _read("name;latitude;notes\nCAM01;52.1;hello\n")
    assert problems == []
    assert rows[0].values["name"] == "CAM01"


def test_delimiter_stays_comma_when_the_header_has_both_characters():
    # A quoted column name holding a semicolon must not flip the delimiter.
    rows, problems = _read('name,latitude,"notes;extra"\n')
    assert [p.column for p in problems] == ["notes;extra"]
    assert rows == []


def test_byte_order_mark_is_stripped():
    rows, problems = _read("﻿name,latitude\nCAM01,52.1\n")
    assert problems == []
    assert rows[0].values["name"] == "CAM01"


def test_crlf_line_endings_are_accepted():
    rows, problems = _read("name,latitude\r\nCAM01,52.1\r\n")
    assert problems == []
    assert len(rows) == 1


def test_carriage_return_only_line_endings_are_accepted():
    """Excel for Mac's "CSV (Macintosh)" format. This used to raise csv.Error
    and reach the user as HTTP 500."""
    rows, problems = _read("name,latitude\rCAM01,52.1\rCAM02,52.2\r")
    assert problems == []
    assert [r.values["name"] for r in rows] == ["CAM01", "CAM02"]


def test_carriage_return_only_row_numbers_match_the_lines_in_the_file():
    rows, _ = _read("name,latitude\rCAM01,52.1\rCAM02,52.2\r")
    assert [r.row for r in rows] == [2, 3]


def test_carriage_return_only_file_still_picks_the_semicolon_delimiter():
    """Proves _pick_delimiter and the reader agree about where lines end."""
    rows, problems = _read("name;latitude\rCAM01;52.1\r")
    assert problems == []
    assert rows[0].values["name"] == "CAM01"


def test_mixed_carriage_return_and_crlf_endings_are_accepted():
    rows, problems = _read("name,latitude\r\nCAM01,52.1\rCAM02,52.2\r\n")
    assert problems == []
    assert len(rows) == 2


def test_a_lone_carriage_return_inside_a_quoted_value_counts_as_a_line():
    """Pins the one behaviour change from reading with newline="". A
    spreadsheet shows that value across two lines too, so the later row
    numbers shifting is the honest answer."""
    rows, problems = _read('name,latitude,notes\n"a\rb",52.1,x\nCAM02,52.2,y\n')
    assert problems == []
    assert [r.row for r in rows] == [3, 4]


# Enough filler to push the runaway field past csv's 131,072 character
# field_size_limit, which is what turns a stray quote into an exception.
_RUNAWAY_FILLER = "x,1\n" * 40000


def test_an_unbalanced_quotation_mark_is_reported_not_raised():
    """One stray quote makes csv read the rest of the file as a single value
    until it hits its field size limit. That used to be HTTP 500."""
    rows, problems = _read('name,latitude\nCAM01,52.1\n"open\n' + _RUNAWAY_FILLER)
    assert rows == []
    assert len(problems) == 1
    assert "quotation mark" in problems[0].message


def test_the_unreadable_problem_carries_a_row_number():
    _, problems = _read('name,latitude\n"open\n' + _RUNAWAY_FILLER)
    assert problems[0].row is not None


def test_a_genuinely_huge_quoted_value_is_reported_not_raised():
    """Balanced quotes, but one cell past csv's field limit. Same message,
    since the advice to re-save the file still applies."""
    rows, problems = _read(f'name,latitude,notes\nCAM01,52.1,"{"x" * 200000}"\n')
    assert rows == []
    assert len(problems) == 1


def test_absent_optional_column_reads_as_empty_string():
    rows, _ = _read("name,latitude\nCAM01,52.1\n")
    assert rows[0].values["notes"] == ""


def test_missing_trailing_cells_read_as_empty_string():
    rows, problems = _read("name,latitude,notes\nCAM01,52.1\n")
    assert problems == []
    assert rows[0].values["notes"] == ""


def test_values_are_trimmed():
    rows, _ = _read("name,latitude\n  CAM01  , 52.1 \n")
    assert rows[0].values["name"] == "CAM01"
    assert rows[0].values["latitude"] == "52.1"


def test_row_numbers_match_the_line_in_the_file():
    rows, _ = _read("name,latitude\nCAM01,52.1\n\nCAM02,52.2\n")
    assert [r.row for r in rows] == [2, 4]


def test_blank_lines_are_ignored():
    rows, problems = _read("name,latitude\n\nCAM01,52.1\n\n")
    assert problems == []
    assert len(rows) == 1


def test_rows_of_only_separators_are_ignored():
    # Excel likes to append these when other columns were once wider.
    rows, problems = _read("name,latitude,notes\nCAM01,52.1,\n,,\n,,\n")
    assert problems == []
    assert len(rows) == 1


def test_leading_blank_line_before_the_header_is_ignored():
    rows, problems = _read("\nname,latitude\nCAM01,52.1\n")
    assert problems == []
    assert rows[0].values["name"] == "CAM01"


def test_row_with_more_values_than_the_header_is_reported_and_skipped():
    rows, problems = _read("name,latitude\nCAM01,52.1,oops\n")
    assert rows == []
    assert len(problems) == 1
    assert problems[0].row == 2


def test_quoted_value_containing_the_delimiter_is_one_value():
    rows, problems = _read('name,latitude,notes\nCAM01,52.1,"near the river, north side"\n')
    assert problems == []
    assert rows[0].values["notes"] == "near the river, north side"


def test_missing_required_column_is_reported():
    rows, problems = _read("name,notes\nCAM01,hello\n")
    assert rows == []
    assert [p.column for p in problems] == ["latitude"]
    assert "must contain: name, latitude" in problems[0].message


def test_unrecognised_column_is_reported_with_the_allowed_ones():
    rows, problems = _read("name,latitude,camera_id\nCAM01,52.1,7\n")
    assert rows == []
    assert [p.column for p in problems] == ["camera_id"]
    assert "Allowed columns are: name, latitude, notes." in problems[0].message


def test_repeated_column_is_reported():
    rows, problems = _read("name,latitude,name\nCAM01,52.1,CAM02\n")
    assert rows == []
    assert [p.column for p in problems] == ["name"]
    assert "more than once" in problems[0].message


def test_a_typo_reports_both_the_unknown_and_the_missing_column():
    _, problems = _read("name,lat\nCAM01,52.1\n")
    assert {p.column for p in problems} == {"lat", "latitude"}


def test_empty_file_is_reported():
    rows, problems = _read("")
    assert rows == []
    assert problems[0].row is None
    assert "empty" in problems[0].message


def test_whitespace_only_file_is_reported():
    _, problems = _read("\n \n")
    assert "empty" in problems[0].message


def test_header_only_file_is_reported():
    rows, problems = _read("name,latitude\n")
    assert rows == []
    assert "no sites" in problems[0].message


def test_invalid_utf8_is_reported():
    rows, problems = read_csv_rows(
        "name,latitude\nZ\xfcrich,52.1\n".encode("latin-1"), REQUIRED, OPTIONAL, "site"
    )
    assert rows == []
    assert "UTF-8" in problems[0].message


def test_unicode_values_survive():
    rows, _ = _read("name,latitude\nZürich-Süd,52.1\n")
    assert rows[0].values["name"] == "Zürich-Süd"


def test_drop_problem_rows_keeps_only_clean_rows():
    rows = [
        SiteImportRow(row=2, name="a", latitude=1.0, longitude=1.0),
        SiteImportRow(row=3, name="b", latitude=1.0, longitude=1.0),
    ]
    problems = [CsvImportProblem(row=3, message="bad")]
    assert [r.row for r in drop_problem_rows(rows, problems)] == [2]


def test_drop_problem_rows_ignores_file_level_problems():
    rows = [SiteImportRow(row=2, name="a", latitude=1.0, longitude=1.0)]
    problems = [CsvImportProblem(message="whole file")]
    assert len(drop_problem_rows(rows, problems)) == 1
