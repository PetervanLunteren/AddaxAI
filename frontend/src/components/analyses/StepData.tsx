/**
 * Step 1: Data - Folder selection with modern styling.
 */

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Callout } from "@/components/ui/callout";
import { useWizard } from "@/components/wizard/WizardContext";
import { FolderOpen, HardDrive, Check } from "lucide-react";

interface StepDataProps {
  folderPath: string;
  onFolderChange: (path: string) => void;
}

export function StepData({ folderPath, onFolderChange }: StepDataProps) {
  const { setCanGoNext } = useWizard();
  const [isSelecting, setIsSelecting] = useState(false);

  const handleSelectFolder = async () => {
    setIsSelecting(true);
    try {
      // TODO: Implement folder selection via backend API
      // For now, use a placeholder
      const mockPath = "/Users/peter/Documents/CameraTrap/Deployment01";
      onFolderChange(mockPath);
      setCanGoNext(true);
    } catch (error) {
      console.error("Failed to select folder:", error);
    } finally {
      setIsSelecting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <Label htmlFor="folderPath" className="text-lg font-semibold text-gray-900 flex items-center gap-2">
          <FolderOpen className="w-5 h-5 text-blue-600" />
          Data Storage
        </Label>
        <p className="text-sm text-gray-600 mt-2">
          Select the folder containing your camera trap images. All images in this folder and subfolders will be analyzed.
        </p>
      </div>

      <div className="space-y-4">
        <Button
          type="button"
          variant="outline"
          onClick={handleSelectFolder}
          disabled={isSelecting}
          className="w-full h-24 border-2 border-dashed hover:border-blue-500 hover:bg-blue-50 transition-colors group"
        >
          <div className="flex flex-col items-center gap-2">
            <HardDrive className="w-8 h-8 text-gray-400 group-hover:text-blue-600 transition-colors" />
            <span className="font-medium">
              {isSelecting ? "Selecting..." : "Browse for Folder"}
            </span>
            <span className="text-xs text-gray-500">Click to select deployment folder</span>
          </div>
        </Button>

        {folderPath && (
          <Callout variant="success" title="Folder selected" className="animate-in fade-in slide-in-from-top-2 duration-300">
            <p className="break-all font-mono text-xs">{folderPath}</p>
          </Callout>
        )}
      </div>

      <Callout variant="info">
        <p className="text-sm">
          <strong>Tip:</strong> Make sure your images are organized in a clear folder structure.
          Subfolders will be scanned recursively.
        </p>
      </Callout>
    </div>
  );
}
