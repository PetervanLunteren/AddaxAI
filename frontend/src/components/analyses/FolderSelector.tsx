/**
 * Folder Selector Component
 *
 * Allows users to select a folder containing camera trap images.
 * - Uses native Electron dialog in production
 * - Uses manual text input in browser (development)
 * - Auto-scans folder for file count and validation
 */

import { useState } from "react";
import { Folder, Loader2, AlertCircle, CheckCircle2, Edit3 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useFolderScan } from "@/hooks/useFolderScan";
import { isElectron } from "@/lib/platform";

interface FolderSelectorProps {
  value: string | null;
  onChange: (path: string) => void;
  error?: string;
}

export function FolderSelector({ value, onChange, error }: FolderSelectorProps) {
  const { data: scanResult, isLoading: isScanning } = useFolderScan(value);
  const [showManualInput, setShowManualInput] = useState(false);
  const inElectron = isElectron();

  // Handle Electron folder selection
  const handleElectronSelect = async () => {
    if (!window.electronAPI) return;

    const folderPath = await window.electronAPI.selectFolder();
    if (folderPath) {
      onChange(folderPath);
    }
  };

  // Determine validation state
  const hasFiles = scanResult && scanResult.total_count > 0;
  const hasError = error;
  const showWarning = !isScanning && value && !hasFiles && !hasError;

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">
        Folder
        <span className="text-red-600 ml-1">*</span>
      </label>

      {/* Folder display area */}
      <div
        className={`
          relative rounded-lg border-2 border-dashed p-4 transition-colors
          ${hasError ? "border-red-500 bg-red-50" : "border-gray-300"}
          ${showWarning ? "border-yellow-500 bg-yellow-50" : ""}
          ${hasFiles && !hasError ? "border-green-500 bg-green-50" : ""}
        `}
      >

        <div className="flex items-start gap-3">
          <Folder className="h-5 w-5 text-gray-400 mt-0.5 shrink-0" />

          <div className="flex-1 min-w-0">
            {value ? (
              <>
                <p className="text-sm font-medium text-gray-900 truncate" title={value}>
                  {value.split("/").pop() || value}
                </p>
                <p className="text-xs text-gray-500 truncate mt-0.5" title={value}>
                  {value}
                </p>

                {/* Status indicator */}
                {isScanning && (
                  <div className="flex items-center gap-2 mt-2 text-sm text-gray-600">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>Scanning folder...</span>
                  </div>
                )}

                {!isScanning && hasFiles && !hasError && (
                  <div className="flex items-center gap-2 mt-2 text-sm text-green-700">
                    <CheckCircle2 className="h-4 w-4" />
                    <span>
                      {scanResult.total_count} files found ({scanResult.image_count} images
                      {scanResult.video_count > 0 && `, ${scanResult.video_count} videos`})
                    </span>
                  </div>
                )}

                {showWarning && (
                  <div className="flex items-center gap-2 mt-2 text-sm text-yellow-700">
                    <AlertCircle className="h-4 w-4" />
                    <span>No images found in this folder</span>
                  </div>
                )}

                {hasError && (
                  <div className="flex items-center gap-2 mt-2 text-sm text-red-700">
                    <AlertCircle className="h-4 w-4" />
                    <span>{error || "Failed to scan folder"}</span>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center py-2">
                <p className="text-sm text-gray-600">No folder selected</p>
                <p className="text-xs text-gray-500 mt-1">
                  {inElectron
                    ? "Click the button below to select a folder"
                    : "Click the button below to enter a folder path"}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Select button / Manual input toggle */}
      {inElectron ? (
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={handleElectronSelect}
          className="w-full"
        >
          <Folder className="h-4 w-4 mr-2" />
          {value ? "Change folder" : "Select folder"}
        </Button>
      ) : (
        <>
          {!showManualInput ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setShowManualInput(true)}
              className="w-full"
            >
              <Edit3 className="h-4 w-4 mr-2" />
              Enter folder path manually
            </Button>
          ) : (
            <div className="space-y-2">
              <Input
                type="text"
                placeholder="/Users/peter/Downloads/example-projects-small/project_Kenya/..."
                value={value || ""}
                onChange={(e) => onChange(e.target.value)}
                className="text-sm font-mono"
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => setShowManualInput(false)}
                className="w-full"
              >
                Use drag and drop instead
              </Button>
            </div>
          )}
        </>
      )}

      {/* Help text */}
      <p className="text-xs text-gray-500">
        {inElectron
          ? "Select a folder containing camera trap images (JPG, PNG, or other image formats)"
          : showManualInput
          ? "Paste the full folder path (dev mode only - Electron will use native picker)"
          : "Enter folder path manually for testing (dev mode - Electron will use native picker)"}
      </p>
    </div>
  );
}
