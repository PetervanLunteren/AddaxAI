/**
 * Queue Item Component
 *
 * Displays a single deployment queue entry in list format.
 * Shows: deployment name (from folder), site, file count, status
 * Actions: view details, delete
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Folder, MoreVertical, Trash2, Eye, EyeOff } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { TagPills } from "@/components/ui/tag-pills";
import { sitesApi } from "@/api/sites";
import { useFolderScan } from "@/hooks/useFolderScan";
import type { DeploymentQueueEntry } from "@/api/deployment-queue";

function formatDatetimeOffset(seconds: number): string {
  const sign = seconds >= 0 ? "+" : "-";
  const abs = Math.abs(seconds);
  const h = Math.floor(abs / 3600);
  const m = Math.floor((abs % 3600) / 60);
  const s = abs % 60;
  const parts: string[] = [];
  if (h > 0) parts.push(`${h}h`);
  if (m > 0) parts.push(`${m}m`);
  if (s > 0 || parts.length === 0) parts.push(`${s}s`);
  return `${sign}${parts.join(" ")}`;
}

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

  const siteLabel = entry.site_id
    ? (site?.name ?? "Loading...")
    : "(no site)";

  const hasTags = entry.tags && Object.keys(entry.tags).length > 0;
  const hasOffset =
    entry.datetime_offset_seconds != null && entry.datetime_offset_seconds !== 0;

  // Status badge styling
  const getStatusBadge = () => {
    const baseClasses = "inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium";

    switch (entry.status) {
      case "pending":
        return {
          classes: `${baseClasses} bg-gray-100 text-gray-700`,
          label: "Pending"
        };
      case "processing":
        return {
          classes: `${baseClasses} bg-teal-50 text-teal-700`,
          label: "Processing"
        };
      case "completed":
        return {
          classes: `${baseClasses} bg-green-50 text-green-700`,
          label: "Completed"
        };
      case "failed":
        return {
          classes: `${baseClasses} bg-red-100 text-red-700`,
          label: "Failed"
        };
      default:
        return {
          classes: `${baseClasses} bg-gray-100 text-gray-700`,
          label: entry.status
        };
    }
  };

  const statusBadge = getStatusBadge();

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
            <span className={statusBadge.classes}>
              {statusBadge.label}
            </span>
          </div>

          {/* Path: width-based truncation from the start so the
              trailing deployment folder stays visible. */}
          <p
            dir="rtl"
            className="text-xs text-gray-500 font-mono truncate text-left"
            title={entry.folder_path}
          >
            <bdi>{entry.folder_path}</bdi>
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
            {/* Site */}
            <dt className="text-gray-500 font-medium">Site:</dt>
            <dd className="text-gray-900">{siteLabel}</dd>

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
            <dd className="text-gray-900">{new Date(entry.created_at_utc).toLocaleString()}</dd>

            {/* Datetime offset (only when non-zero) */}
            {hasOffset && (
              <>
                <dt className="text-gray-500 font-medium">Time offset:</dt>
                <dd className="text-gray-900">
                  {formatDatetimeOffset(entry.datetime_offset_seconds!)}
                </dd>
              </>
            )}

            {/* Notes */}
            {entry.notes && (
              <>
                <dt className="text-gray-500 font-medium">Notes:</dt>
                <dd className="text-gray-900 whitespace-pre-wrap break-words">
                  {entry.notes}
                </dd>
              </>
            )}

            {/* Tags */}
            {hasTags && (
              <>
                <dt className="text-gray-500 font-medium">Tags:</dt>
                <dd className="text-gray-900">
                  <TagPills tags={entry.tags} maxVisible={8} />
                </dd>
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
            variant="outline"
            size="sm"
            onClick={() => setShowDetails(false)}
            className="mt-3"
          >
            <EyeOff className="h-4 w-4 mr-2" />
            Hide details
          </Button>
        </div>
      )}
    </div>
  );
}
