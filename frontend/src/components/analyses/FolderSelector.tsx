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
  AlertCircle,
  Calendar,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Folder,
  FolderInput,
  Image,
  Loader2,
  MapPin,
  MapPinOff,
  Video,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription } from "@/components/ui/alert";
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
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

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
  /** Hide the scan result panel (file counts, GPS, dates, adjust-dates link). */
  hideScanResult?: boolean;
  /** Hide the GPS pin in the scan result. Used in Timelapse integration where
   *  there is no Site / Map context that would consume it. */
  hideGps?: boolean;
  /** Hide the "DateTime metadata not found" red alert. Used in
   *  Timelapse integration where the runner does not require EXIF
   *  DateTimeOriginal — files without timestamps are simply absent
   *  from the sequence-level smoother, but detection and
   *  classification still run on them. The main app needs the warning
   *  because its DB load step (`json_pipeline.load_json_to_database`)
   *  raises MissingTimestampError on missing timestamps. */
  hideDatetimeWarning?: boolean;
  /** Render scan results as a single muted dot-separated line instead
   *  of the tall teal card. Used in the Timelapse window where vertical
   *  space is at a premium and GPS / adjust-dates / missing-datetime are
   *  all hidden anyway. */
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
  hideScanResult = false,
  hideGps = false,
  hideDatetimeWarning = false,
  compactScanResult = false,
  noScan = false,
}: FolderSelectorProps) {
  const { data: scanResult, isLoading: isScanning } = useFolderScan(
    noScan ? null : value,
  );
  const [showManualInput, setShowManualInput] = useState(!isElectron());
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
    <TooltipProvider>
      <div className="space-y-2">
        {/* Label (suppressed when the parent provides its own label) */}
        {!hideLabel && (
          <label className="text-sm font-medium">Folder</label>
        )}

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
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : hideScanResult ? null : value ? (
          isScanning ? (
            compactScanResult ? (
              <div className="flex items-center gap-2 px-1 text-xs text-muted-foreground">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                <span>Scanning folder...</span>
              </div>
            ) : (
              <Alert>
                <Loader2 className="h-4 w-4 animate-spin" />
                <AlertDescription>Scanning folder...</AlertDescription>
              </Alert>
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
              <div className="border border-[#0f6064] bg-[#ebf0f2] rounded-lg p-4 space-y-2">
                {/* File counts — only show types that exist */}
                {scanResult.image_count > 0 && (
                  <div className="flex items-center gap-1.5 text-sm text-[#0f6064]">
                    <Image className="h-4 w-4" />
                    <span>{scanResult.image_count} {scanResult.image_count === 1 ? "image" : "images"}</span>
                  </div>
                )}
                {scanResult.video_count > 0 && (
                  <div className="flex items-center gap-1.5 text-sm text-[#0f6064]">
                    <Video className="h-4 w-4" />
                    <span>{scanResult.video_count} {scanResult.video_count === 1 ? "video" : "videos"}</span>
                  </div>
                )}

                {/* GPS — show "found" with coordinates in tooltip, or "not found".
                    Suppressed in Timelapse integration where the app has no Site
                    or Map context that would consume the coordinates. */}
                {!hideGps && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="flex items-center gap-1.5 text-sm text-[#0f6064] cursor-default">
                        {scanResult.gps_location ? (
                          <>
                            <MapPin className="h-4 w-4" />
                            <span>GPS found</span>
                          </>
                        ) : (
                          <>
                            <MapPinOff className="h-4 w-4" />
                            <span>No GPS metadata</span>
                          </>
                        )}
                      </div>
                    </TooltipTrigger>
                    {scanResult.gps_location && (
                      <TooltipContent>
                        {scanResult.gps_location.latitude.toFixed(6)}, {scanResult.gps_location.longitude.toFixed(6)}
                      </TooltipContent>
                    )}
                  </Tooltip>
                )}

                {/* Date range — unambiguous format (e.g. "7 Feb 2016") */}
                <div className="flex items-center gap-1.5 text-sm text-[#0f6064]">
                  <Calendar className="h-4 w-4" />
                  <span>
                    {scanResult.start_date && scanResult.end_date ? (
                      (() => {
                        const fmt = (d: Date) => d.toLocaleString([], { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
                        const offsetMs = datetimeOffsetSeconds * 1000;
                        const start = new Date(new Date(scanResult.start_date).getTime() + offsetMs);
                        const end = new Date(new Date(scanResult.end_date).getTime() + offsetMs);
                        return `${fmt(start)} – ${fmt(end)}`;
                      })()
                    ) : (
                      "No datetime metadata"
                    )}
                  </span>
                </div>

                {/* Datetime offset link */}
                {onAdjustDates && scanResult.start_date && (
                  <div className="text-sm text-[#0f6064]">
                    {datetimeOffsetSeconds !== 0 ? (
                      <span>
                        Offset:{" "}
                        <button
                          type="button"
                          onClick={onAdjustDates}
                          className="underline underline-offset-2 hover:text-[#0a4a4d]"
                        >
                          {formatOffset(datetimeOffsetSeconds)}
                        </button>
                      </span>
                    ) : (
                      <span>
                        Dates look wrong?{" "}
                        <button
                          type="button"
                          onClick={onAdjustDates}
                          className="underline underline-offset-2 hover:text-[#0a4a4d]"
                        >
                          Adjust dates
                        </button>
                      </span>
                    )}
                  </div>
                )}
              </div>

              {/* DateTime missing error. Suppressed in Timelapse integration
                  where missing EXIF is not a hard failure (the runner
                  still detects and classifies; only sequence smoothing
                  skips those files). */}
              {scanResult.missing_datetime && !hideDatetimeWarning && (
                <Alert variant="destructive" className="bg-red-50 border-red-300">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription className="text-sm">
                    <div className="space-y-2">
                      <p className="font-semibold">DateTime metadata not found.</p>

                      <p>DateTime information is essential for accurate statistics, graphs, and exports in AddaxAI. This usually means the images have been processed, uploaded/downloaded, copied, or stripped of metadata. Please use the raw data directly from the camera SD card with DateTime metadata intact.</p>

                      {/* Validation log */}
                      {scanResult.datetime_validation_log && scanResult.datetime_validation_log.length > 0 && (
                        <details className="mt-2 p-3 bg-red-100 rounded border border-red-300">
                          <summary className="cursor-pointer font-semibold text-sm">
                            Technical Details
                          </summary>
                          <div className="mt-2 space-y-1 font-mono text-xs text-red-900">
                            {scanResult.datetime_validation_log.map((log, idx) => (
                              <div key={idx} className="whitespace-pre-wrap break-words">
                                {log}
                              </div>
                            ))}
                          </div>
                        </details>
                      )}

                      <p className="text-sm">
                        Using raw data and AddaxAI still can't find the timestamps? Please contact{' '}
                        <a href="mailto:peter@addaxdatascience.com" className="underline font-semibold">
                          peter@addaxdatascience.com
                        </a>
                      </p>
                    </div>
                  </AlertDescription>
                </Alert>
              )}
            </>
            )
          ) : (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>No images found in this folder</AlertDescription>
            </Alert>
          )
        ) : null}
      </div>
    </TooltipProvider>
  );
}

/**
 * Single-row teal scan summary for the Timelapse window: same palette as
 * the full main-app panel, but compressed into one horizontal pill so it
 * costs much less vertical space. Skips counts that are zero. Falls back
 * to a "no datetime metadata" cell when EXIF timestamps are absent
 * (Timelapse mode allows this).
 */
function CompactScanLine({
  imageCount,
  videoCount,
  startDate,
  endDate,
  offsetSeconds,
}: {
  imageCount: number;
  videoCount: number;
  startDate: string | null;
  endDate: string | null;
  offsetSeconds: number;
}) {
  const fmt = (d: Date) =>
    d.toLocaleString([], {
      day: "numeric",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  let dateRange: string;
  if (startDate && endDate) {
    const offsetMs = offsetSeconds * 1000;
    const s = new Date(new Date(startDate).getTime() + offsetMs);
    const e = new Date(new Date(endDate).getTime() + offsetMs);
    dateRange = `${fmt(s)} – ${fmt(e)}`;
  } else {
    dateRange = "No datetime metadata";
  }
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-1 rounded-md border border-[#0f6064] bg-[#ebf0f2] px-3 py-2 text-sm text-[#0f6064]">
      {imageCount > 0 && (
        <div className="flex items-center gap-1.5">
          <Image className="h-4 w-4" />
          <span>
            {imageCount} {imageCount === 1 ? "image" : "images"}
          </span>
        </div>
      )}
      {videoCount > 0 && (
        <div className="flex items-center gap-1.5">
          <Video className="h-4 w-4" />
          <span>
            {videoCount} {videoCount === 1 ? "video" : "videos"}
          </span>
        </div>
      )}
      <div className="flex items-center gap-1.5">
        <Calendar className="h-4 w-4" />
        <span>{dateRange}</span>
      </div>
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
