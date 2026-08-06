"""
Shared CSV reading for the bulk imports.

Knows nothing about sites or deployments. It owns exactly the parts that are
identical for every import: decoding, the delimiter rule, the header check,
blank lines, extra cells and row numbering. Everything domain specific
(number parsing, path checks, duplicate rules) lives in the per-import
modules next to this one.

Row numbers are the line numbers a spreadsheet shows, so the header is row 1
and the first record is row 2. That is what the user needs to find the row
again in Excel.
"""

import csv
import io
from dataclasses import dataclass
from typing import Protocol, TypeVar

from app.api.schemas.csv_import import CsvImportProblem

# Generous: roughly 20k rows. The whole file is read into memory, so there
# has to be a ceiling somewhere.
MAX_CSV_BYTES = 2 * 1024 * 1024

_NOT_UTF8 = (
    "The file is not valid UTF-8 text. Open it in your spreadsheet program "
    "and save it again as CSV UTF-8."
)
_TOO_MANY_VALUES = (
    "This row has more values than the header. Check for an extra comma, or "
    "put quotation marks around a value that contains a comma."
)
_UNREADABLE = (
    "This row could not be read. A quotation mark is probably missing its "
    "partner, so everything after it was read as one value. Check the "
    "quotation marks above this row, or open the file in your spreadsheet "
    "program and save it again as CSV."
)


@dataclass(frozen=True)
class RawCsvRow:
    """One record, with every known column present.

    Absent optional columns read as an empty string, so callers never have to
    guard for a missing key.
    """

    row: int
    values: dict[str, str]


def blank_to_none(value: str) -> str | None:
    """An empty cell means "not given", which is None in the database."""
    return value or None


class _HasRow(Protocol):
    row: int


_RowT = TypeVar("_RowT", bound=_HasRow)


def drop_problem_rows(rows: list[_RowT], problems: list[CsvImportProblem]) -> list[_RowT]:
    """Keep only the rows nothing is wrong with.

    The preview lists these as "will be imported", so a row that parsed but
    then failed a database check must not appear among them.
    """
    bad = {p.row for p in problems if p.row is not None}
    return [row for row in rows if row.row not in bad]


def read_csv_rows(
    content: bytes,
    required_columns: tuple[str, ...],
    optional_columns: tuple[str, ...],
    item_label: str,
) -> tuple[list[RawCsvRow], list[CsvImportProblem]]:
    """Decode and split a CSV into rows, without interpreting any value.

    `item_label` is the singular noun used in the file-level messages
    ("site", "deployment").

    A broken header returns no rows at all: per-row problems on a file with
    the wrong columns are noise, and fixing the header changes all of them.
    """
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return [], [CsvImportProblem(message=_NOT_UTF8)]

    if not text.strip():
        return [], [
            CsvImportProblem(
                message=(
                    f"The file is empty. Add a header row and at least one {item_label}."
                )
            )
        ]

    # newline="" is the documented way to feed csv. It leaves line endings to
    # the reader, which treats \r, \r\n and \n alike, so a file saved as
    # "CSV (Macintosh)" reads normally. The default (\n only) left a lone \r
    # sitting inside a field, and csv raised, which reached the user as a 500.
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=_pick_delimiter(text))
    known = set(required_columns) | set(optional_columns)

    header: list[str] | None = None
    rows: list[RawCsvRow] = []
    problems: list[CsvImportProblem] = []

    try:
        for cells in reader:
            if not any(cell.strip() for cell in cells):
                # Blank line, or one of the trailing ",,,," rows Excel likes to
                # append. Skipped whether it comes before or after the header.
                continue

            if header is None:
                header = [cell.strip() for cell in cells]
                header_problems = _check_header(header, required_columns, optional_columns)
                if header_problems:
                    return [], header_problems
                continue

            if len(cells) > len(header):
                problems.append(
                    CsvImportProblem(row=reader.line_num, message=_TOO_MANY_VALUES)
                )
                continue

            values = dict.fromkeys(known, "")
            for index, name in enumerate(header):
                if index < len(cells):
                    values[name] = cells[index].strip()
            rows.append(RawCsvRow(row=reader.line_num, values=values))
    except csv.Error:
        # Almost always an unbalanced quotation mark: everything after it is
        # read as one value until csv gives up on the 131,072 character field
        # limit, thousands of lines later. Report the line the reader stopped
        # at, which is the only position we know, and name the likely cause.
        # Anything else csv can raise lands here too, so this can never 500.
        return [], [CsvImportProblem(row=reader.line_num, message=_UNREADABLE)]

    if not rows and not problems:
        problems.append(
            CsvImportProblem(
                message=(
                    f"The file has a header row but no {item_label}s. "
                    f"Add one row per {item_label}."
                )
            )
        )

    return rows, problems


def _pick_delimiter(text: str) -> str:
    """Comma, unless the header line uses semicolons and no comma at all.

    That is the file Excel writes in a locale where the comma is the decimal
    separator. One explicit rule instead of csv.Sniffer, which guesses from a
    sample and gets single-column files and paths containing commas wrong.
    """
    header_line = next((line for line in text.splitlines() if line.strip()), "")
    return ";" if ";" in header_line and "," not in header_line else ","


def _check_header(
    header: list[str],
    required_columns: tuple[str, ...],
    optional_columns: tuple[str, ...],
) -> list[CsvImportProblem]:
    """Every problem with the column names, reported together.

    These messages name the allowed columns rather than staying constant like
    the row-level ones. There is at most one per column so there is nothing
    to group, and naming them is what makes the fix obvious.
    """
    allowed = ", ".join(required_columns + optional_columns)
    problems: list[CsvImportProblem] = []
    seen: set[str] = set()

    for name in header:
        if name in seen:
            problems.append(
                CsvImportProblem(
                    column=name,
                    message="This column appears more than once. Use each column only once.",
                )
            )
            continue
        seen.add(name)
        if name not in required_columns and name not in optional_columns:
            problems.append(
                CsvImportProblem(
                    column=name,
                    message=f"This column is not recognised. Allowed columns are: {allowed}.",
                )
            )

    required_list = ", ".join(required_columns)
    for name in required_columns:
        if name not in seen:
            problems.append(
                CsvImportProblem(
                    column=name,
                    message=f"This column is missing. The first row must contain: {required_list}.",
                )
            )

    return problems
