/**
 * Queue Item Component
 *
 * Displays a single deployment queue entry in list format.
 * Shows: deployment name (from folder), site, file count, status
 * Actions: view details, delete
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Folder, MapPin, Image, MoreVertical, Trash2, Eye } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { sitesApi } from "@/api/sites";
import { projectsApi } from "@/api/projects";
import { modelsApi } from "@/api/models";
import { useFolderScan } from "@/hooks/useFolderScan";
import type { DeploymentQueueEntry } from "@/api/deployment-queue";

interface QueueItemProps {
  entry: DeploymentQueueEntry;
  onDelete: (id: string) => void;
}

export function QueueItem({ entry, onDelete }: QueueItemProps) {
  const [showDetails, setShowDetails] = useState(false);

  // Fetch site info
  const { data: site } = useQuery({
    queryKey: ["sites", entry.site_id],
    queryFn: () => (entry.site_id ? sitesApi.get(entry.site_id) : null),
    enabled: !!entry.site_id,
  });

  // Fetch project info
  const { data: project } = useQuery({
    queryKey: ["projects", entry.project_id],
    queryFn: () => projectsApi.get(entry.project_id),
    enabled: showDetails, // Only fetch when details are shown
  });

  // Fetch model info for friendly names
  const { data: detectionModels } = useQuery({
    queryKey: ["models", "detection"],
    queryFn: () => modelsApi.listDetectionModels(),
    enabled: showDetails,
  });

  const { data: classificationModels } = useQuery({
    queryKey: ["models", "classification"],
    queryFn: () => modelsApi.listClassificationModels(),
    enabled: showDetails,
  });

  // Fetch SpeciesNet locations for friendly country/state names
  const { data: locations } = useQuery({
    queryKey: ["speciesnet-locations"],
    queryFn: () => modelsApi.getSpeciesNetLocations(),
    enabled: showDetails && !!project?.classification_model_id?.toLowerCase().includes("speciesnet"),
  });

  // Get file count from folder scan
  const { data: scanResult, isLoading: isScanning } = useFolderScan(entry.folder_path);

  // Derive deployment name from folder path
  const deploymentName = entry.folder_path.split("/").pop() || "Unknown";

  // Get friendly model names
  const getDetectionModelName = (modelId: string) => {
    const model = detectionModels?.find((m) => m.model_id === modelId);
    return model?.friendly_name || modelId;
  };

  const getClassificationModelName = (modelId: string | null) => {
    if (!modelId) return "None";
    const model = classificationModels?.find((m) => m.model_id === modelId);
    return model?.friendly_name || modelId;
  };

  // Get friendly location names
  const getCountryName = (countryCode: string | null) => {
    if (!countryCode || !locations) return countryCode || "Not set";
    // locations.countries is a Record<string, string> where key is display name and value is code
    // We need to find the key by value
    const entry = Object.entries(locations.countries).find(([_, code]) => code === countryCode);
    return entry ? entry[0] : countryCode;
  };

  const getStateName = (stateCode: string | null) => {
    if (!stateCode || !locations) return stateCode || "Not set";
    // locations.us_states is a Record<string, string> where key is display name and value is code
    const entry = Object.entries(locations.us_states).find(([_, code]) => code === stateCode);
    return entry ? entry[0] : stateCode;
  };

  // Show last 50 characters of path
  const truncatedPath = entry.folder_path.length > 50
    ? "..." + entry.folder_path.slice(-50)
    : entry.folder_path;

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-3 hover:shadow-sm transition-shadow">
      <div className="flex items-center justify-between gap-3">
        {/* Main info */}
        <div className="flex-1 min-w-0">
          {/* Deployment name */}
          <div className="flex items-center gap-2 mb-1">
            <Folder className="h-4 w-4 text-gray-400 shrink-0" />
            <h3 className="font-medium text-sm truncate" title={deploymentName}>
              {deploymentName}
            </h3>
          </div>

          {/* Path */}
          <p className="text-xs text-gray-500 font-mono truncate" title={entry.folder_path}>
            {truncatedPath}
          </p>
        </div>

        {/* Actions dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="h-8 w-8 shrink-0">
              <MoreVertical className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => setShowDetails(true)}>
              <Eye className="h-4 w-4 mr-2" />
              View details
            </DropdownMenuItem>
            <DropdownMenuItem
              onClick={() => onDelete(entry.id)}
              className="text-red-600 focus:text-red-600"
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Details section */}
      {showDetails && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-2 text-xs">
            {/* Folder */}
            <dt className="text-gray-500 font-medium">Folder:</dt>
            <dd className="text-gray-900 font-mono break-all">{entry.folder_path}</dd>

            {/* Site */}
            <dt className="text-gray-500 font-medium">Site:</dt>
            <dd className="text-gray-900">{site ? site.name : "Loading..."}</dd>

            {/* Files */}
            <dt className="text-gray-500 font-medium">Files:</dt>
            <dd className="text-gray-900">
              {isScanning ? (
                "Scanning..."
              ) : scanResult?.total_count ? (
                `${scanResult.total_count} (${scanResult.image_count} images, ${scanResult.video_count} videos)`
              ) : (
                "No files"
              )}
            </dd>

            {/* Created */}
            <dt className="text-gray-500 font-medium">Created:</dt>
            <dd className="text-gray-900">{new Date(entry.created_at).toLocaleString()}</dd>

            {/* Project settings section */}
            {project && (
              <>
                <dt className="text-gray-500 font-medium mt-2 col-span-2 text-[11px] uppercase tracking-wide">
                  Project Settings
                </dt>

                {/* Detection model */}
                <dt className="text-gray-500 font-medium">Detection model:</dt>
                <dd className="text-gray-900">
                  {getDetectionModelName(project.detection_model_id)}
                </dd>

                {/* Classification model */}
                <dt className="text-gray-500 font-medium">Classification model:</dt>
                <dd className="text-gray-900">
                  {getClassificationModelName(project.classification_model_id)}
                </dd>

                {/* Species selection - only show if not SpeciesNet */}
                {!project.classification_model_id?.toLowerCase().includes("speciesnet") && project.excluded_classes && (
                  <>
                    <dt className="text-gray-500 font-medium">Species selection:</dt>
                    <dd className="text-gray-900">
                      {project.excluded_classes.length === 0
                        ? "All species"
                        : `${project.excluded_classes.length} species excluded`}
                    </dd>
                  </>
                )}

                {/* Geographic location - only show if SpeciesNet */}
                {project.classification_model_id?.toLowerCase().includes("speciesnet") && (
                  <>
                    <dt className="text-gray-500 font-medium">Country:</dt>
                    <dd className="text-gray-900">{getCountryName(project.country_code)}</dd>

                    {(project.country_code === "US" || project.country_code === "USA") && (
                      <>
                        <dt className="text-gray-500 font-medium">State:</dt>
                        <dd className="text-gray-900">{getStateName(project.state_code)}</dd>
                      </>
                    )}
                  </>
                )}
              </>
            )}

            {/* Error */}
            {entry.error && (
              <>
                <dt className="text-red-600 font-medium">Error:</dt>
                <dd className="text-red-600">{entry.error}</dd>
              </>
            )}
          </div>

          <Button
            variant="link"
            size="sm"
            onClick={() => setShowDetails(false)}
            className="px-0 mt-3"
          >
            Hide details
          </Button>
        </div>
      )}
    </div>
  );
}
