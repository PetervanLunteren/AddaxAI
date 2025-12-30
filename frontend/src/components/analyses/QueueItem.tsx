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
import { Badge } from "@/components/ui/badge";
import { sitesApi } from "@/api/sites";
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

  // Get file count from folder scan
  const { data: scanResult, isLoading: isScanning } = useFolderScan(entry.folder_path);

  // Derive deployment name from folder path
  const deploymentName = entry.folder_path.split("/").pop() || "Unknown";

  // Status badge styling
  const statusColors = {
    pending: "bg-gray-100 text-gray-700",
    processing: "bg-blue-100 text-blue-700",
    completed: "bg-green-100 text-green-700",
    failed: "bg-red-100 text-red-700",
  };

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 hover:shadow-sm transition-shadow">
      <div className="flex items-start justify-between gap-3">
        {/* Main info */}
        <div className="flex-1 min-w-0 space-y-2">
          {/* Deployment name */}
          <div className="flex items-center gap-2">
            <Folder className="h-4 w-4 text-gray-400 shrink-0" />
            <h3 className="font-medium text-sm truncate" title={deploymentName}>
              {deploymentName}
            </h3>
          </div>

          {/* Site */}
          {site && (
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <MapPin className="h-3.5 w-3.5 shrink-0" />
              <span className="truncate">{site.name}</span>
            </div>
          )}

          {/* File count */}
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <Image className="h-3.5 w-3.5 shrink-0" />
            <span>
              {isScanning ? (
                "Scanning..."
              ) : scanResult?.total_count ? (
                `${scanResult.total_count} files`
              ) : (
                "No files"
              )}
            </span>
          </div>

          {/* Status */}
          <Badge
            className={`text-xs ${statusColors[entry.status]}`}
            variant="secondary"
          >
            {entry.status}
          </Badge>
        </div>

        {/* Actions dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="h-8 w-8">
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

      {/* Details dialog (simple for now) */}
      {showDetails && (
        <div className="mt-3 pt-3 border-t border-gray-100 space-y-1 text-xs text-gray-600">
          <p>
            <strong>Folder:</strong> {entry.folder_path}
          </p>
          <p>
            <strong>Created:</strong> {new Date(entry.created_at).toLocaleString()}
          </p>
          {entry.error && (
            <p className="text-red-600">
              <strong>Error:</strong> {entry.error}
            </p>
          )}
          <Button variant="link" size="sm" onClick={() => setShowDetails(false)} className="px-0">
            Hide details
          </Button>
        </div>
      )}
    </div>
  );
}
