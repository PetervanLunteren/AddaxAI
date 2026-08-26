/**
 * Shared shell for the CSV bulk-import dialogs.
 *
 * Owns the flow that is the same for every import: pick a file, send it for
 * a dry run, show what will be created and what is wrong, then send it again
 * for real. All or nothing: while the backend reports any problem the import
 * button stays disabled and nothing is written.
 *
 * The caller supplies the wording, the column reference, the example file
 * and the two requests. This component does not know which import it runs.
 */

import { useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { Download, Upload } from "lucide-react";
import { toast } from "sonner";

import type {
  CsvImportPreview,
  CsvImportProblem,
  CsvImportResult,
} from "@/api/types";
import { downloadTextFile } from "@/lib/download";
import { Button } from "./button";
import { Callout } from "./callout";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./dialog";

/** How many row numbers to name before falling back to "and N more". */
const MAX_LISTED_ROWS = 10;
/** How many offending values to name alongside those row numbers. */
const MAX_LISTED_VALUES = 3;
/** Longest offending value shown before it is cut back to its tail. */
const MAX_VALUE_CHARS = 32;

export interface CsvColumnHelp {
  /** Exact column name, as it must appear in the header row. */
  name: string;
  /** Optional columns may be left blank. Required is the default. */
  optional?: boolean;
  help: string;
}

interface CsvImportDialogProps<TRow> {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: string;
  /** Singular and plural of the thing being imported, for counts and labels. */
  noun: { one: string; many: string };
  columns: CsvColumnHelp[];
  exampleFilename: string;
  exampleCsv: string;
  /** Dry run: check the file, write nothing. */
  onPreview: (file: File) => Promise<CsvImportPreview<TRow>>;
  /**
   * Write for real. Resolves with the whole result, not just a count: the
   * confirm step re-checks the file and can refuse it, and those problems are
   * the only signal the user gets that nothing was written.
   */
  onImport: (file: File) => Promise<CsvImportResult>;
  /** Body of one row in the "will be imported" list. */
  renderRow: (row: TRow) => React.ReactNode;
  /**
   * Turns a phrase inside one kind of problem message into a link, so the
   * way out sits in the sentence that describes the problem rather than as a
   * loose button that could belong to any of them.
   */
  problemLink?: ProblemMessageLink;
}

export interface ProblemMessageLink {
  /** Only problems reported against this column are linked. */
  column: string;
  /**
   * The exact phrase to turn into a link. It has to match the backend
   * message, which is pinned by a test on that constant so a reword cannot
   * silently drop the link. If it stops matching, the message still renders,
   * just without a link.
   */
  phrase: string;
  to: string;
}

export function CsvImportDialog<TRow extends { row: number }>({
  open,
  onOpenChange,
  title,
  description,
  noun,
  columns,
  exampleFilename,
  exampleCsv,
  onPreview,
  onImport,
  renderRow,
  problemLink,
}: CsvImportDialogProps<TRow>) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);

  const previewMutation = useMutation({ mutationFn: onPreview });
  const importMutation = useMutation({
    mutationFn: onImport,
    // Gated on `imported`, not on `problems`. The confirm step re-checks the
    // file and answers 200 with `{imported: 0, problems: [...]}` when it no
    // longer passes, so celebrating anything that is not an actual write is
    // how this reported "Imported 0 sites" over a failed import.
    onSuccess: (result) => {
      if (result.imported === 0) return;
      toast.success(
        `Imported ${result.imported} ${result.imported === 1 ? noun.one : noun.many}`
      );
      close();
    },
  });

  const preview = previewMutation.data ?? null;
  const problemGroups = useMemo(
    () => groupProblems(preview?.problems ?? []),
    [preview]
  );

  // Problems the confirm step found, which the preview knew nothing about.
  const importProblems = importMutation.data?.problems ?? [];
  const importProblemGroups = useMemo(
    () => groupProblems(importMutation.data?.problems ?? []),
    [importMutation.data]
  );

  const badRows = new Set(
    (preview?.problems ?? []).map((p) => p.row).filter((r) => r !== null)
  ).size;
  const totalRows = (preview?.rows.length ?? 0) + badRows;

  const canImport =
    !!preview && preview.problems.length === 0 && preview.rows.length > 0;
  const importDisabled =
    !file ||
    previewMutation.isPending ||
    !canImport ||
    importProblems.length > 0 ||
    importMutation.isPending;

  function close() {
    setFile(null);
    previewMutation.reset();
    importMutation.reset();
    onOpenChange(false);
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const picked = e.target.files?.[0];
    // Clear immediately so picking the same file again still fires a change.
    // The whole fix-and-retry loop depends on this.
    e.target.value = "";
    if (!picked) return;
    // Also the only place a failed confirm's problems are cleared. Reordering
    // these four lines would leave them showing over a fresh file.
    importMutation.reset();
    setFile(picked);
    previewMutation.mutate(picked);
  }

  return (
    <Dialog open={open} onOpenChange={(next) => (next ? onOpenChange(true) : close())}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div>
            <ul className="rounded-lg border divide-y text-sm">
              {columns.map((column) => (
                <li key={column.name} className="flex gap-3 px-3 py-2">
                  {/* Wide enough for the longest name plus "optional"
                      (paired_cameras) without wrapping into the help. */}
                  <span className="w-40 shrink-0 font-mono text-xs leading-5">
                    {column.name}
                    {column.optional && (
                      <span className="ml-1 font-sans text-muted-foreground">
                        optional
                      </span>
                    )}
                  </span>
                  <span className="text-muted-foreground">{column.help}</span>
                </li>
              ))}
            </ul>
            <p className="mt-2 text-xs text-muted-foreground">
              The first row must hold these column names. Column order does not
              matter. If a value contains a comma, put it in double quotes.
              Spreadsheet apps do this for you.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => downloadTextFile(exampleFilename, exampleCsv)}
            >
              <Download className="mr-2 h-4 w-4" />
              Download example
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => inputRef.current?.click()}
            >
              <Upload className="mr-2 h-4 w-4" />
              {file ? "Choose another file" : "Choose file"}
            </Button>
            <input
              ref={inputRef}
              type="file"
              accept=".csv,text/csv"
              className="hidden"
              onChange={handleFileChange}
            />
            {file && (
              <span className="truncate text-sm text-muted-foreground">
                {file.name}
              </span>
            )}
          </div>

          {previewMutation.isPending && (
            <p className="text-sm text-muted-foreground">Checking file...</p>
          )}

          {previewMutation.isError && (
            <Callout variant="error" title="Could not read that file">
              {errorMessage(previewMutation.error)}
            </Callout>
          )}

          {problemGroups.length > 0 && (
            <Callout
              variant="error"
              // A file-level problem (empty file, wrong columns) has no row
              // numbers, and "0 of 0 rows" would be a nonsense headline.
              title={
                badRows > 0
                  ? `${badRows} of ${totalRows} rows have a problem`
                  : "This file cannot be imported"
              }
            >
              <ProblemList groups={problemGroups} link={problemLink} />
            </Callout>
          )}

          {/* Hidden while the confirm step is reporting problems. The preview
              rows are stale by then, and "5 rows look fine" sitting above
              "Nothing was imported" contradicts itself. */}
          {preview && preview.rows.length > 0 && importProblems.length === 0 && (
            <div>
              <p className="mb-2 text-sm">
                {canImport
                  ? `${preview.rows.length} ${
                      preview.rows.length === 1 ? noun.one : noun.many
                    } will be imported:`
                  : `${preview.rows.length} row${
                      preview.rows.length === 1 ? "" : "s"
                    } look fine:`}
              </p>
              <ul className="max-h-[40vh] overflow-y-auto rounded-lg border divide-y text-sm">
                {preview.rows.map((row) => (
                  <li
                    key={row.row}
                    className="flex items-baseline gap-4 px-3 py-2"
                  >
                    {renderRow(row)}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* One callout for both ways the confirm step can fail: the request
              itself failed, or it came back 200 saying it refused the file.
              The second is the common one, because the confirm re-checks
              against the database and the disk. */}
          {(importMutation.isError || importProblemGroups.length > 0) && (
            <Callout
              variant="error"
              title="Nothing was imported"
            >
              {importMutation.isError ? (
                errorMessage(importMutation.error)
              ) : (
                <ProblemList groups={importProblemGroups} link={problemLink} />
              )}
            </Callout>
          )}
        </div>

        <DialogFooter>
          <Button type="button" variant="outline" onClick={close}>
            Cancel
          </Button>
          <Button
            type="button"
            disabled={importDisabled}
            onClick={() => file && importMutation.mutate(file)}
          >
            {importMutation.isPending
              ? "Importing..."
              : previewMutation.isPending
                ? "Checking file..."
                : canImport
                  ? `Import ${preview.rows.length} ${
                      preview.rows.length === 1 ? noun.one : noun.many
                    }`
                  : `Import ${noun.many}`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/**
 * The grouped problems plus the instruction that goes with them.
 *
 * Shared by the preview callout and the confirm-failed callout so the two
 * cannot drift. Module level on purpose: a component declared inside the
 * dialog body would be a new type on every render, which remounts it and
 * throws away the list's scroll position.
 */
function ProblemList({
  groups,
  link,
}: {
  groups: ProblemGroup[];
  link?: ProblemMessageLink;
}) {
  return (
    <>
      <ul className="mt-1 max-h-40 space-y-2 overflow-y-auto">
        {groups.map((group) => (
          <li key={`${group.column}|${group.message}`}>
            <span className="font-medium">
              {group.column && (
                <span className="font-mono text-xs">{group.column}: </span>
              )}
              {linkedMessage(group, link)}
            </span>
            {group.rows.length > 0 && (
              <span className="block text-xs opacity-80">
                {describeRows(group)}
              </span>
            )}
          </li>
        ))}
      </ul>
      <p className="mt-2">
        Nothing is imported while there is a problem. Fix the file and choose
        it again.
      </p>
    </>
  );
}

/**
 * The problem message, with the configured phrase turned into a link.
 *
 * Plain text when this problem is about another column, or when the phrase
 * is not in the message. That second case is the safe failure: a reworded
 * backend message loses the link but still reads correctly.
 */
function linkedMessage(group: ProblemGroup, link?: ProblemMessageLink) {
  if (!link || group.column !== link.column) return group.message;

  const [before, ...rest] = group.message.split(link.phrase);
  if (rest.length === 0) return group.message;

  return (
    <>
      {before}
      <Link
        to={link.to}
        className="font-semibold underline underline-offset-2"
      >
        {link.phrase}
      </Link>
      {rest.join(link.phrase)}
    </>
  );
}

interface ProblemGroup {
  column: string | null;
  message: string;
  rows: number[];
  values: string[];
}

/**
 * One line per distinct problem instead of one per row.
 *
 * CSV mistakes are systematic, not random: a sheet exported with comma
 * decimals has the same problem on every row. Listing all forty makes the
 * user read all forty to learn there is one thing to fix.
 */
function groupProblems(problems: CsvImportProblem[]): ProblemGroup[] {
  const groups = new Map<string, ProblemGroup>();

  for (const problem of problems) {
    const key = `${problem.column ?? ""}|${problem.message}`;
    const group = groups.get(key) ?? {
      column: problem.column,
      message: problem.message,
      rows: [],
      values: [],
    };
    if (problem.row !== null) group.rows.push(problem.row);
    if (problem.value && !group.values.includes(problem.value)) {
      group.values.push(problem.value);
    }
    groups.set(key, group);
  }

  return [...groups.values()].sort((a, b) => b.rows.length - a.rows.length);
}

/** "3 rows: 2, 7, 12 (forest ridge, Cam 2)" */
function describeRows(group: ProblemGroup): string {
  const sorted = [...group.rows].sort((a, b) => a - b);
  const shown = sorted.slice(0, MAX_LISTED_ROWS).join(", ");
  const rest = sorted.length - MAX_LISTED_ROWS;
  const rows = rest > 0 ? `${shown} and ${rest} more` : shown;

  const values = group.values.slice(0, MAX_LISTED_VALUES).map(shorten).join(", ");
  const more = group.values.length > MAX_LISTED_VALUES ? ", ..." : "";
  const suffix = values ? ` (${values}${more})` : "";

  return `${sorted.length} row${sorted.length === 1 ? "" : "s"}: ${rows}${suffix}`;
}

/**
 * Keep the end of a long value.
 *
 * The offending value is usually a name (short) or a folder path (long, and
 * only its tail identifies which folder). Printing a full path here wrapped
 * every problem over three lines and buried the row numbers.
 */
function shorten(value: string): string {
  return value.length <= MAX_VALUE_CHARS
    ? value
    : `...${value.slice(-MAX_VALUE_CHARS)}`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
