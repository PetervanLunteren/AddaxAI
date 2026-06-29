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
  /** Render scan results as a single muted dot-separated line instead
   *  of the tall teal card, for places where vertical space is at a
   *  premium. */
  compactScanResult?: boolean;
  /** Skip the folder scan entirely. For picking a *destination* folder,
   *  which may not exist yet and whose contents are irrelevant. Pair
   *  with ``hideScanResult`` so no scan panel renders. */
  noScan?: boolean;
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
}: FolderSelectorProps) {
  const { data: scanResult, isLoading: isScanning } = useFolderScan(
    noScan ? null : value,
  );
  const [isDragOver, setIsDragOver] = useState(false);
  const inElectron = isElectron();

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
        {/* Label (suppressed when the parent provides its own label) */}
        {!hideLabel && (
          <label className="text-sm font-medium">Folder</label>
        )}
        {!hideLabel && caption && (
          <p className="text-xs text-muted-foreground">{caption}</p>
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
          ) : hasFiles ? (
            compactScanResult ? (
              <CompactScanLine
                imageCount={scanResult.image_count}
                videoCount={scanResult.video_count}
                startDate={scanResult.start_date}
                endDate={scanResult.end_date}
                offsetSeconds={datetimeOffsetSeconds}
              />
            ) : (
            <>
              <CompactScanLine
                imageCount={scanResult.image_count}
                videoCount={scanResult.video_count}
                startDate={scanResult.start_date}
                endDate={scanResult.end_date}
                offsetSeconds={datetimeOffsetSeconds}
                onAdjustDates={onAdjustDates}
              />

              {/* Some files lack a capture date. Non-blocking: the
                  backend still detects and classifies them; they just
                  drop out of time-based stats. */}
              {scanResult.missing_datetime && (
                <Callout variant="warning">
                  <div className="space-y-2">
                    <p className="font-medium">
                      Some files have no capture date.
                    </p>

                    <p>
                      AddaxAI will still detect and classify them. Files
                      without a date are left out of time-based stats,
                      charts, and trap-night effort. This usually means
                      the files were copied or had their metadata
                      stripped; raw files from the camera SD card keep
                      it.
                    </p>

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
            )
          ) : (
            <Callout variant="error" size="compact">
              No images found in this folder
            </Callout>
          )
        ) : null}
        </div>
    </div>
  );
}

/**
 * Single muted caption summarising a folder scan: file counts and rough
 * date span, dot-separated. Skips counts that are zero and falls back to a
 * "no datetime metadata" note when EXIF timestamps are absent. When
 * `onAdjustDates` is supplied, appends an "Adjust dates" link (and shows the
 * active offset) in the same style as the label picker's "Refine" link.
 */
function CompactScanLine({
  imageCount,
  videoCount,
  startDate,
  endDate,
  offsetSeconds,
  onAdjustDates,
}: {
  imageCount: number;
  videoCount: number;
  startDate: string | null;
  endDate: string | null;
  offsetSeconds: number;
  onAdjustDates?: () => void;
}) {
  // Date-only, rough estimate: the scan reads a sample of files, not every
  // one, so this is an approximate span. Exact per-file times are in the
  // Adjust-dates modal.
  const fmt = (d: Date) =>
    d.toLocaleDateString([], {
      day: "numeric",
      month: "short",
      year: "numeric",
    });
  let dateRange: string;
  if (startDate && endDate) {
    const offsetMs = offsetSeconds * 1000;
    const s = new Date(new Date(startDate).getTime() + offsetMs);
    const e = new Date(new Date(endDate).getTime() + offsetMs);
    const range = s.toDateString() === e.toDateString()
      ? fmt(s)
      : `${fmt(s)} – ${fmt(e)}`;
    dateRange = `Dates span roughly ${range}`;
  } else {
    dateRange = "No datetime metadata";
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
