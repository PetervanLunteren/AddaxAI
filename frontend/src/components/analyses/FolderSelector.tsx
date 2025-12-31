/**
 * Folder Selector Component
 *
 * Simplified version matching Create Project modal style.
 * - Clean input field with info tooltip
 * - File count shown below
 * - Electron native picker or manual input for dev
 */

import { useState } from "react";
import { Folder, Info, CheckCircle2, AlertCircle, Loader2, ChevronDown, Image, Video, MapPin, Calendar } from "lucide-react";
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
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

// Dev-only: Test deployment folders for quick selection
const TEST_DEPLOYMENTS = [
  // Ukraine
  "/Users/peter/Downloads/example-projects-small/project_Ukraine/loc_SIMON03/dep001",
  "/Users/peter/Downloads/example-projects-small/project_Ukraine/loc_SIMON03/dep002",
  // Kenya
  "/Users/peter/Downloads/example-projects-small/project_Kenya/Chui River/deployment_001",
  "/Users/peter/Downloads/example-projects-small/project_Kenya/Chui River/deployment_002",
  "/Users/peter/Downloads/example-projects-small/project_Kenya/Kifaru Plains/deployment_001",
  "/Users/peter/Downloads/example-projects-small/project_Kenya/Kifaru Plains/deployment_002",
  "/Users/peter/Downloads/example-projects-small/project_Kenya/Kifaru Plains/deployment_003",
  "/Users/peter/Downloads/example-projects-small/project_Kenya/Kifaru Plains/deployment_004",
  "/Users/peter/Downloads/example-projects-small/project_Kenya/Kifaru Plains/deployment_005",
  "/Users/peter/Downloads/example-projects-small/project_Kenya/Kifaru Plains/deployment_006",
  "/Users/peter/Downloads/example-projects-small/project_Kenya/Kifaru Plains/deployment_007",
  "/Users/peter/Downloads/example-projects-small/project_Kenya/Kifaru Plains/deployment_008",
  "/Users/peter/Downloads/example-projects-small/project_Kenya/Loita Hills/deployment_001",
  "/Users/peter/Downloads/example-projects-small/project_Kenya/Loita Hills/deployment_002",
  "/Users/peter/Downloads/example-projects-small/project_Kenya/Loita Hills/deployment_003",
  // New Zealand
  "/Users/peter/Downloads/example-projects-small/project_NewZealand/NI-TAR03/deployment_001",
  "/Users/peter/Downloads/example-projects-small/project_NewZealand/NI-TAR03/deployment_002",
  "/Users/peter/Downloads/example-projects-small/project_NewZealand/NI-TAR03/deployment_003",
  "/Users/peter/Downloads/example-projects-small/project_NewZealand/NI-TAR03/deployment_004",
  "/Users/peter/Downloads/example-projects-small/project_NewZealand/NI-TAR03/deployment_005",
  "/Users/peter/Downloads/example-projects-small/project_NewZealand/OT-FJI02/deployment_001",
  "/Users/peter/Downloads/example-projects-small/project_NewZealand/OT-FJI02/deployment_002",
  "/Users/peter/Downloads/example-projects-small/project_NewZealand/OT-FJI02/deployment_003",
  "/Users/peter/Downloads/example-projects-small/project_NewZealand/OT-FJI02/deployment_004",
  "/Users/peter/Downloads/example-projects-small/project_NewZealand/OT-FJI02/deployment_005",
  "/Users/peter/Downloads/example-projects-small/project_NewZealand/SI-MTK04/deployment_001",
  "/Users/peter/Downloads/example-projects-small/project_NewZealand/SI-MTK04/deployment_002",
  "/Users/peter/Downloads/example-projects-small/project_NewZealand/SI-MTK04/deployment_003",
  "/Users/peter/Downloads/example-projects-small/project_NewZealand/SI-MTK04/deployment_004",
  "/Users/peter/Downloads/example-projects-small/project_NewZealand/SI-MTK04/deployment_005",
  "/Users/peter/Downloads/example-projects-small/project_NewZealand/WK-WAI01/deployment_001",
  "/Users/peter/Downloads/example-projects-small/project_NewZealand/WK-WAI01/deployment_002",
  "/Users/peter/Downloads/example-projects-small/project_NewZealand/WK-WAI01/deployment_003",
];

interface FolderSelectorProps {
  value: string | null;
  onChange: (path: string) => void;
  error?: string;
}

export function FolderSelector({ value, onChange, error }: FolderSelectorProps) {
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
        {/* Label with info tooltip */}
        <label className="flex items-center gap-1.5 text-sm font-medium">
          Folder
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
            </TooltipTrigger>
            <TooltipContent>
              <p className="max-w-sm">
                Select a folder containing all images and videos from a single deployment.
                A deployment is one camera SD card from start to end at a single site. It is essential
                to add one complete deployment at a time (not partial, not multiple) to ensure accurate
                statistics, exports, maps, and graphs. The system will recursively search all subfolders
                for images and videos. Add multiple deployments by queuing each one separately.
              </p>
            </TooltipContent>
          </Tooltip>
        </label>

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
              placeholder="/Users/peter/Downloads/example-projects-small/project_Kenya/..."
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
        ) : value ? (
          isScanning ? (
            <Alert>
              <Loader2 className="h-4 w-4 animate-spin" />
              <AlertDescription>Scanning folder...</AlertDescription>
            </Alert>
          ) : hasFiles ? (
            <>
              <div className="border border-[#0f6064] bg-[#ebf0f2] rounded-lg p-4 space-y-2">
                {/* All scan results in vertical list */}
                <div className="flex items-center gap-1.5 text-sm text-[#0f6064]">
                  <Image className="h-4 w-4" />
                  <span>{scanResult.image_count} images</span>
                </div>

                <div className="flex items-center gap-1.5 text-sm text-[#0f6064]">
                  <Video className="h-4 w-4" />
                  <span>{scanResult.video_count} videos</span>
                </div>

                <div className="flex items-center gap-1.5 text-sm text-[#0f6064]">
                  <MapPin className="h-4 w-4" />
                  <span>
                    {scanResult.gps_location
                      ? `${scanResult.gps_location.latitude.toFixed(6)}, ${scanResult.gps_location.longitude.toFixed(6)}`
                      : 'Not found'}
                  </span>
                </div>

                <div className="flex items-center gap-1.5 text-sm text-[#0f6064]">
                  <Calendar className="h-4 w-4" />
                  <span>
                    {scanResult.start_date && scanResult.end_date ? (
                      <>
                        {new Date(scanResult.start_date).toLocaleDateString()} - {new Date(scanResult.end_date).toLocaleDateString()}
                        {(() => {
                          const start = new Date(scanResult.start_date);
                          const end = new Date(scanResult.end_date);
                          const days = Math.ceil((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24));
                          return days > 0 ? ` (${days} days)` : '';
                        })()}
                      </>
                    ) : (
                      'Not found'
                    )}
                  </span>
                </div>
              </div>

              {/* DateTime missing error */}
              {scanResult.missing_datetime && (
                <Alert variant="destructive" className="bg-red-50 border-red-300">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription className="text-sm">
                    <div className="space-y-2">
                      <p className="font-semibold">DateTime metadata not found in images.</p>

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
                        AddaxAI still can't find the timestamps? Please contact{' '}
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
