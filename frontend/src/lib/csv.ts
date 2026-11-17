/**
 * A table as CSV text (RFC 4180), for the "Download CSV" buttons that save
 * what a card shows. UTF-8 without a BOM and CRLF line ends, like the
 * backend's exports.
 *
 * A cell is quoted only when it has to be: when it holds a comma, a
 * quote or a line break. Null and undefined are empty cells.
 */

export type CsvCell = string | number | null | undefined;

function cell(value: CsvCell): string {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function toCsv(headers: string[], rows: CsvCell[][]): string {
  return [headers, ...rows].map((row) => row.map(cell).join(",")).join("\r\n") + "\r\n";
}
