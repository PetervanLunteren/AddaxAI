/**
 * Folder Selector Component
 *
 * Simplified version matching Create Project modal style.
 * - Clean input field with info tooltip
 * - File count shown below
 * - Electron native picker or manual input for dev
 */

import { Fragment, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Folder,
  FolderInput,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Callout } from "@/components/ui/callout";
import { Checkbox } from "@/components/ui/checkbox";
import { FieldHeader } from "@/components/ui/field-header";
import { formatDate, formatDateSpan } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "@/components/ui/dropdown-menu";
import { useFolderScan } from "@/hooks/useFolderScan";
import { isElectron } from "@/lib/platform";
import { formatOffset } from "@/lib/utils";

// Dev-only: Test deployment folders for quick selection
const TEST_DEPLOYMENTS: { scope: string; path: string }[] = [
  {
    scope: "Deployment",
    path: "/Users/peter/Downloads/example-data/project_Kenya/Chui River/deployment_001",
  },
  {
    scope: "Site",
    path: "/Users/peter/Downloads/example-data/project_Kenya/Chui River",
  },
  {
    scope: "Project",
    path: "/Users/peter/Downloads/example-data/project_Kenya",
  },
  {
    scope: "Videos",
    path: "/Users/peter/Downloads/example-data/project_Ukraine/loc_SIMON03/dep001",
  },
  {
    scope: "Edge cases",
    path: "/Users/peter/Downloads/example-data/test_package",
  },
  {
    scope: "ENA24",
    path: "/Users/peter/Downloads/example-data/ena24",
  },
];

interface FolderSelectorProps {
  value: string | null;
  onChange: (path: string) => void;
  error?: string;
  /** Current datetime offset in seconds (0 = no offset). */
  datetimeOffsetSeconds?: number;
  /** Called when the user clicks "Adjust dates". Parent opens the modal. */
  onAdjustDates?: () => void;
  /** Hide the built-in "Folder" label (when the parent provides its own). */
  hideLabel?: boolean;
  /** Optional caption under the built-in "Folder" label. Ignored when
   *  hideLabel is set (the parent supplies its own). Use it to clarify what
   *  the folder is for, since the selector is reused for source and
   *  destination folders. */
  caption?: string;
  /** Hide the scan result caption (file counts, dates, adjust-dates link). */
  hideScanResult?: boolean;
  /** Render the "scanning..." state as a single muted line instead of a
   *  bordered box, for places where vertical space is at a premium. */
  compactScanResult?: boolean;
  /** Skip the folder scan entirely. For picking a *destination* folder,
   *  which may not exist yet and whose contents are irrelevant. Pair
   *  with ``hideScanResult`` so no scan panel renders. */
  noScan?: boolean;
  /** What the user loses when no file carries a capture date, phrased
   *  for this mode: projects lose charts and trap nights, a folder run
   *  gets an empty date column in its tables. Supplying this is what
   *  renders the missing-date warning at all, so a picker that is not
   *  about to analyse the folder (bulk relink) stays quiet. */
  missingDateNote?: string;
  /** Whether the file-date fallback is on for this folder. */
  useFileMtimeFallback?: boolean;
  /** Called when the user ticks the file-date fallback. Supplying this is
   *  what renders the opt-in at all, so call sites that do not pass it
   *  (bulk relink, destination pickers) are unaffected. */
  onUseFileMtimeFallbackChange?: (value: boolean) => void;
}

export function FolderSelector({
  value,
  onChange,
  error,
  datetimeOffsetSeconds = 0,
  onAdjustDates,
  hideLabel = false,
  caption,
  hideScanResult = false,
  compactScanResult = false,
  noScan = false,
  missingDateNote,
  useFileMtimeFallback = false,
  onUseFileMtimeFallbackChange,
}: FolderSelectorProps) {
  // `scanError` is not decoration. Without it a failed scan leaves
  // `scanResult` undefined, which falls through to the "No images found in
  // this folder" branch below and blames the user's data for a drive that
  // did not answer.
  const {
    data: scanResult,
    isLoading: isScanning,
    error: scanError,
  } = useFolderScan(noScan ? null : value);
  const [isDragOver, setIsDragOver] = useState(false);
  const inElectron = isElectron();

  // The scan only fills the mtime range when it found no capture dates, so
  // this is both "the user opted in" and "there is something to show".
  const showingFileDates = Boolean(
    useFileMtimeFallback && scanResult?.mtime_start_date,
  );
  const captionStart = showingFileDates
    ? scanResult!.mtime_start_date
    : (scanResult?.start_date ?? null);
  const captionEnd = showingFileDates
    ? scanResult!.mtime_end_date
    : (scanResult?.end_date ?? null);
  // Reads "which run from 7 Apr 2024 to 28 Apr 2024", or "which are all
  // 7 Apr 2024" when the whole folder lands on one day (common for a
  // single clip). Null when the files could not be stat'ed, which is
  // what hides the opt-in entirely.
  const fileDatesPhrase = (() => {
    const from = scanResult?.mtime_start_date;
    const to = scanResult?.mtime_end_date;
    if (!from || !to) return null;
    const first = formatDate(from);
    const last = formatDate(to);
    return first === last
      ? `which are all ${first}`
      : `which run from ${first} to ${last}`;
  })();

  // Handle Electron folder selection
  const handleElectronSelect = async () => {
    if (!window.electronAPI) return;

    const folderPath = await window.electronAPI.selectFolder();
    if (folderPath) {
      onChange(folderPath);
    }
  };

  // Resolve a dropped file's absolute path through the Electron preload.
  // Single-folder drops only — additional items are ignored.
  const handleDrop = (e: React.DragEvent<HTMLButtonElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (!file) return;
    if (!window.electronAPI?.getDroppedFolderPath) return;
    const path = window.electronAPI.getDroppedFolderPath(file);
    if (path) onChange(path);
  };

  // File count summary
  const hasFiles = scanResult && scanResult.total_count > 0;

  return (
    <div className="space-y-2">
        {/* Label + caption (suppressed when the parent provides its own label) */}
        {!hideLabel && (
          <FieldHeader
            label={<label className="text-sm font-medium">Folder</label>}
            caption={caption}
          />
        )}

        {/* Field + scan caption grouped tight (space-y-1) so the caption sits
            4px under the field, matching the model / label captions. */}
        <div className="space-y-1">
        {/* Folder affordance:
            - Selected: breadcrumb pill of the trailing path segments
              (count adapts to segment length, see BreadcrumbsRow) + Change
              button (Change clears the value, returning to empty state).
            - Empty (Electron): drag-and-drop card. Click also opens the
              native picker so the affordance is discoverable for users who
              don't think to drag.
            - Empty (browser/dev): manual text input + test-deployment
              dropdown. Drag-and-drop only resolves to an absolute path
              inside Electron, so the dev fallback stays as it was. */}
        {value ? (
          <BreadcrumbsRow
            path={value}
            error={!!error}
            onClear={() => onChange("")}
          />
        ) : inElectron ? (
          <button
            type="button"
            onClick={handleElectronSelect}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragEnter={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            className={`w-full flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-4 py-10 transition-all ${
              isDragOver
                ? "border-primary bg-primary/5 text-primary"
                : error
                  ? "border-red-500 bg-background text-muted-foreground"
                  : "border-input bg-background text-muted-foreground hover:bg-accent hover:text-foreground"
            }`}
          >
            <FolderInput
              className={`h-7 w-7 transition-transform ${isDragOver ? "scale-110" : ""}`}
            />
            <span className="text-sm">
              {isDragOver
                ? "Drop to select folder"
                : "Drop folder here or click to browse"}
            </span>
          </button>
        ) : (
          <div className="flex gap-2">
            <Input
              type="text"
              value={value || ""}
              onChange={(e) => onChange(e.target.value)}
              placeholder="/Users/peter/Downloads/example-data/project_Kenya/..."
              className={`flex-1 font-mono text-sm ${error ? "border-red-500" : ""}`}
            />
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  className="shrink-0"
                  title="Quick select test deployment"
                >
                  <ChevronDown className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-80">
                <DropdownMenuLabel>Tests</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {TEST_DEPLOYMENTS.map(({ scope, path }) => (
                  <DropdownMenuItem
                    key={path}
                    onClick={() => onChange(path)}
                    className="text-xs"
                  >
                    <span className="font-semibold">{scope}</span>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        )}

        {/* Scan results or error */}
        {error ? (
          <Callout variant="error" size="compact">{error}</Callout>
        ) : hideScanResult ? null : value ? (
          isScanning ? (
            compactScanResult ? (
              <div className="flex items-center gap-2 pl-3 text-xs text-muted-foreground">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                <span>Scanning folder...</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-3 py-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin shrink-0" />
                <span>Scanning folder...</span>
              </div>
            )
          ) : scanError ? (
            <Callout variant="error" size="compact">
              {scanError instanceof Error
                ? scanError.message
                : "Could not read this folder. Check the drive and try again."}
            </Callout>
          ) : hasFiles ? (
            <>
              <CompactScanLine
                imageCount={scanResult.image_count}
                videoCount={scanResult.video_count}
                startDate={captionStart}
                endDate={captionEnd}
                offsetSeconds={datetimeOffsetSeconds}
                onAdjustDates={onAdjustDates}
                datesFromFileMtime={showingFileDates}
              />

              {/* No file carried a capture date. Non-blocking: the
                  backend still detects and classifies them, so the note
                  explains what the run loses and offers the fallback. */}
              {scanResult.missing_datetime && missingDateNote && (
                <Callout variant="warning">
                  <div className="space-y-2">
                    <p className="font-medium">
                      No capture dates found.
                    </p>

                    <p>{missingDateNote}</p>

                    {/* Opt-in fallback. Only rendered where the parent
                        can persist the choice, and only when the files
                        could be read, so there is no dead state. The
                        range is shown up front because it is the only
                        safeguard: a copied folder reads as today. */}
                    {onUseFileMtimeFallbackChange && fileDatesPhrase && (
                      <label className="flex cursor-pointer items-start gap-3 rounded border border-amber-300 bg-amber-100 p-3">
                        <Checkbox
                          checked={useFileMtimeFallback}
                          onCheckedChange={onUseFileMtimeFallbackChange}
                          className="mt-0.5"
                        />
                        <span className="text-sm font-medium">
                          Use the file dates from your computer,{" "}
                          {fileDatesPhrase}
                        </span>
                      </label>
                    )}

                    {/* Validation log */}
                    {scanResult.datetime_validation_log && scanResult.datetime_validation_log.length > 0 && (
                      <details className="mt-2 rounded border border-amber-300 bg-amber-100 p-3">
                        <summary className="cursor-pointer text-sm font-medium">
                          Technical details
                        </summary>
                        <div className="mt-2 space-y-1 font-mono text-xs text-amber-900">
                          {scanResult.datetime_validation_log.map((log, idx) => (
                            <div key={idx} className="whitespace-pre-wrap break-words">
                              {log}
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                  </div>
                </Callout>
              )}
            </>
          ) : /* Only claim the folder is empty when a scan actually came
                back saying so. This used to be the final `else`, so it
                also fired for "no data and no error" states — a query
                paused because the window is hidden, or one disabled via
                `noScan` — telling the user a folder they never scanned
                holds no images. */
          scanResult ? (
            <Callout variant="error" size="compact">
              No images found in this folder
            </Callout>
          ) : null
        ) : null}
        </div>
    </div>
  );
}

/**
 * Single muted caption summarising a folder scan: file counts and rough
 * date span, dot-separated. Skips counts that are zero and falls back to a
 * "no dates found" note when timestamps are absent. When `onAdjustDates` is
 * supplied, appends an "Adjust dates" link (and shows the active offset) in
 * the same style as the label picker's "Refine" link.
 */
function CompactScanLine({
  imageCount,
  videoCount,
  startDate,
  endDate,
  offsetSeconds,
  onAdjustDates,
  datesFromFileMtime = false,
}: {
  imageCount: number;
  videoCount: number;
  startDate: string | null;
  endDate: string | null;
  offsetSeconds: number;
  onAdjustDates?: () => void;
  /** The dates came from file modification times, not from the camera.
   *  Names the source in the caption so the number is not mistaken for a
   *  capture time. */
  datesFromFileMtime?: boolean;
}) {
  // Date-only. For capture dates this is an estimate: the scan reads a
  // sample of files, not every one. File dates are exact, since a stat()
  // is cheap enough to run over the whole folder. Per-file times are in
  // the Adjust-dates modal either way.
  const range = formatDateSpan(startDate, endDate, offsetSeconds);
  let dateRange: string;
  if (range) {
    dateRange = datesFromFileMtime
      ? `Dates span roughly ${range} (from file dates)`
      : `Dates span roughly ${range}`;
  } else {
    dateRange = "No dates found";
  }

  const parts: string[] = [];
  if (imageCount > 0) {
    parts.push(`${imageCount} ${imageCount === 1 ? "image" : "images"}`);
  }
  if (videoCount > 0) {
    parts.push(`${videoCount} ${videoCount === 1 ? "video" : "videos"}`);
  }
  parts.push(dateRange);
  if (offsetSeconds !== 0) {
    parts.push(`offset ${formatOffset(offsetSeconds)}`);
  }

  return (
    <div className="pl-3 text-xs text-muted-foreground">
      {parts.join(" · ")}
      {onAdjustDates && startDate && (
        <>
          {" · "}
          <button
            type="button"
            onClick={onAdjustDates}
            className="text-primary font-medium hover:underline"
          >
            Adjust dates
          </button>
        </>
      )}
    </div>
  );
}

/**
 * Breadcrumb pill for a selected folder. Shows the folder icon + the
 * trailing path segments separated by chevrons; longer paths get a
 * leading ellipsis. The number of segments adapts to their length:
 * short names (e.g. `dep002 / loc_SIMON03 / project_kenya / data`) get
 * up to 4 levels, long names collapse to fewer so the row does not
 * overflow on typical container widths. No bold weight: every segment
 * reads at the same emphasis. The "Change" button clears the
 * selection so the parent re-renders the empty state.
 */
const MAX_BREADCRUMB_CHARS = 40;
const MAX_BREADCRUMB_SEGMENTS = 4;

/** Pick the trailing segments that fit in a rough character budget,
 *  always keeping at least the leaf. Tuned by visual judgement on
 *  typical camera-trap paths; see header doc for examples. */
function pickTailSegments(parts: string[]): string[] {
  let charBudget = 0;
  let kept = 0;
  for (let i = parts.length - 1; i >= 0; i--) {
    const segChars = parts[i].length + (kept > 0 ? 1 : 0); // +1 for the chevron
    // Stop only after we have at least one segment, so an unusually long
    // leaf still gets shown (it'll truncate via CSS).
    if (kept >= 1 && (charBudget + segChars > MAX_BREADCRUMB_CHARS || kept >= MAX_BREADCRUMB_SEGMENTS)) {
      break;
    }
    charBudget += segChars;
    kept++;
  }
  return parts.slice(-Math.max(kept, 1));
}

function BreadcrumbsRow({
  path,
  error,
  onClear,
}: {
  path: string;
  error: boolean;
  onClear: () => void;
}) {
  // Split on either / or \ so Windows paths render the same way.
  const parts = path.split(/[\\/]/).filter(Boolean);
  const tail = pickTailSegments(parts);
  const truncated = parts.length > tail.length;

  return (
    <div className="flex gap-2">
      <div
        className={`flex-1 flex items-center gap-1.5 rounded-md border bg-background px-3 py-2 text-sm overflow-hidden min-w-0 ${
          error ? "border-red-500" : "border-input"
        }`}
        title={path}
      >
        <Folder className="h-4 w-4 text-muted-foreground shrink-0" />
        {truncated && (
          <>
            <span className="text-muted-foreground shrink-0">…</span>
            <ChevronRight className="h-3 w-3 text-muted-foreground shrink-0" />
          </>
        )}
        {tail.map((seg, i) => (
          <Fragment key={i}>
            {i > 0 && (
              <ChevronRight className="h-3 w-3 text-muted-foreground shrink-0" />
            )}
            <span
              className={
                i === tail.length - 1
                  ? "truncate"
                  : "text-muted-foreground truncate"
              }
            >
              {seg}
            </span>
          </Fragment>
        ))}
      </div>
      <Button
        type="button"
        variant="outline"
        onClick={onClear}
        className="shrink-0"
      >
        Change
      </Button>
    </div>
  );
}
