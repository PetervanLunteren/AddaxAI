/**
 * Folder Selector Component
 *
 * Simplified version matching Create Project modal style.
 * - Clean input field with info tooltip
 * - File count shown below
 * - Electron native picker or manual input for dev
 */

import { useState } from "react";
import { Folder, CheckCircle2, AlertCircle, Loader2, ChevronDown, Image, Video, MapPin, MapPinOff, Calendar } from "lucide-react";
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
const TEST_DEPLOYMENTS = [
  // Ukraine
  "/Users/peter/Downloads/example-data/project_Ukraine/loc_SIMON03/dep001",
  "/Users/peter/Downloads/example-data/project_Ukraine/loc_SIMON03/dep002",
  // Kenya
  "/Users/peter/Downloads/example-data/project_Kenya/Chui River/deployment_001",
  "/Users/peter/Downloads/example-data/project_Kenya/Chui River/deployment_002",
  "/Users/peter/Downloads/example-data/project_Kenya/Kifaru Plains/deployment_001",
  "/Users/peter/Downloads/example-data/project_Kenya/Kifaru Plains/deployment_002",
  "/Users/peter/Downloads/example-data/project_Kenya/Kifaru Plains/deployment_003",
  "/Users/peter/Downloads/example-data/project_Kenya/Kifaru Plains/deployment_004",
  "/Users/peter/Downloads/example-data/project_Kenya/Kifaru Plains/deployment_005",
  "/Users/peter/Downloads/example-data/project_Kenya/Kifaru Plains/deployment_006",
  "/Users/peter/Downloads/example-data/project_Kenya/Kifaru Plains/deployment_007",
  "/Users/peter/Downloads/example-data/project_Kenya/Kifaru Plains/deployment_008",
  "/Users/peter/Downloads/example-data/project_Kenya/Loita Hills/deployment_001",
  "/Users/peter/Downloads/example-data/project_Kenya/Loita Hills/deployment_002",
  "/Users/peter/Downloads/example-data/project_Kenya/Loita Hills/deployment_003",
  // New Zealand
  "/Users/peter/Downloads/example-data/project_NewZealand/NI-TAR03/deployment_001",
  "/Users/peter/Downloads/example-data/project_NewZealand/NI-TAR03/deployment_002",
  "/Users/peter/Downloads/example-data/project_NewZealand/NI-TAR03/deployment_003",
  "/Users/peter/Downloads/example-data/project_NewZealand/NI-TAR03/deployment_004",
  "/Users/peter/Downloads/example-data/project_NewZealand/NI-TAR03/deployment_005",
  "/Users/peter/Downloads/example-data/project_NewZealand/OT-FJI02/deployment_001",
  "/Users/peter/Downloads/example-data/project_NewZealand/OT-FJI02/deployment_002",
  "/Users/peter/Downloads/example-data/project_NewZealand/OT-FJI02/deployment_003",
  "/Users/peter/Downloads/example-data/project_NewZealand/OT-FJI02/deployment_004",
  "/Users/peter/Downloads/example-data/project_NewZealand/OT-FJI02/deployment_005",
  "/Users/peter/Downloads/example-data/project_NewZealand/SI-MTK04/deployment_001",
  "/Users/peter/Downloads/example-data/project_NewZealand/SI-MTK04/deployment_002",
  "/Users/peter/Downloads/example-data/project_NewZealand/SI-MTK04/deployment_003",
  "/Users/peter/Downloads/example-data/project_NewZealand/SI-MTK04/deployment_004",
  "/Users/peter/Downloads/example-data/project_NewZealand/SI-MTK04/deployment_005",
  "/Users/peter/Downloads/example-data/project_NewZealand/WK-WAI01/deployment_001",
  "/Users/peter/Downloads/example-data/project_NewZealand/WK-WAI01/deployment_002",
  "/Users/peter/Downloads/example-data/project_NewZealand/WK-WAI01/deployment_003",
  // Seattle
  "/Users/peter/Downloads/example-data/project_Seattle/dans_backyard",
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
  /** Hide the GPS pin in the scan result. Used in Timelapse mode where
   *  there is no Site / Map context that would consume it. */
  hideGps?: boolean;
  /** Hide the "DateTime metadata not found" red alert. Used in
   *  Timelapse mode where the runner does not require EXIF
   *  DateTimeOriginal — files without timestamps are simply absent
   *  from the sequence-level smoother, but detection and
   *  classification still run on them. The main app needs the warning
   *  because its DB load step (`json_pipeline.load_json_to_database`)
   *  raises MissingTimestampError on missing timestamps. */
  hideDatetimeWarning?: boolean;
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
}: FolderSelectorProps) {
  const { data: scanResult, isLoading: isScanning } = useFolderScan(value);
  const [showManualInput, setShowManualInput] = useState(!isElectron());
  const inElectron = isElectron();

  // Handle Electron folder selection
  const handleElectronSelect = async () => {
    if (!window.electronAPI) return;

    const folderPath = await window.electronAPI.selectFolder();
    if (folderPath) {
      onChange(folderPath);
    }
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

        {/* Input or button */}
        {inElectron ? (
          <div className="flex gap-2">
            <Input
              type="text"
              value={value || ""}
              readOnly
              placeholder="No folder selected"
              className={`flex-1 font-mono text-sm ${error ? "border-red-500" : ""}`}
            />
            <Button
              type="button"
              variant="outline"
              onClick={handleElectronSelect}
              className="shrink-0"
            >
              <Folder className="h-4 w-4 mr-2" />
              Select
            </Button>
          </div>
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
                <DropdownMenuLabel>Test deployments</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {TEST_DEPLOYMENTS.map((path) => {
                  const parts = path.split("/");
                  const project = parts[parts.length - 3];
                  const site = parts[parts.length - 2];
                  const deployment = parts[parts.length - 1];
                  return (
                    <DropdownMenuItem
                      key={path}
                      onClick={() => onChange(path)}
                      className="font-mono text-xs"
                    >
                      <div className="flex flex-col">
                        <span className="font-semibold">{deployment}</span>
                        <span className="text-muted-foreground">
                          {project} / {site}
                        </span>
                      </div>
                    </DropdownMenuItem>
                  );
                })}
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
            <Alert>
              <Loader2 className="h-4 w-4 animate-spin" />
              <AlertDescription>Scanning folder...</AlertDescription>
            </Alert>
          ) : hasFiles ? (
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
                    Suppressed in Timelapse mode where the app has no Site
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

              {/* DateTime missing error. Suppressed in Timelapse mode
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
