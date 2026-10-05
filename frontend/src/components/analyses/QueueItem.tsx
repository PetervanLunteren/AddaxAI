/**
 * Queue Item Component
 *
 * Displays a single deployment queue entry in list format.
 * Shows: deployment name (from folder), site, file count, status
 * Actions: view details, delete
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Folder, Trash2, Eye, EyeOff, LifeBuoy } from "lucide-react";
import { basename } from "@/lib/path-utils";
import { formatDateSpan, formatOffsetSummary } from "@/lib/utils";
import { exportDiagnosticReport } from "@/lib/diagnostic-export";
import { Button } from "@/components/ui/button";
import { TagPills } from "@/components/ui/tag-pills";
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
  const deploymentName = basename(entry.folder_path) || "Unknown";

  const siteLabel = entry.site_id
    ? (site?.name ?? "Loading...")
    : "(no site)";

  const hasTags = entry.tags && Object.keys(entry.tags).length > 0;
  const hasOffset =
    (entry.datetime_offset_seconds != null && entry.datetime_offset_seconds !== 0) ||
    Object.values(entry.camera_offsets ?? {}).some((s) => s !== 0);

  // Status badge styling
  const getStatusBadge = () => {
    const baseClasses = "inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium";

    switch (entry.status) {
      case "pending":
        return {
          classes: `${baseClasses} bg-muted text-muted-foreground`,
          label: "Pending"
        };
      case "processing":
        return {
          classes: `${baseClasses} bg-primary/10 text-primary-ink`,
          label: "Processing"
        };
      case "completed":
        return {
          classes: `${baseClasses} bg-success-subtle text-success-ink`,
          label: "Completed"
        };
      case "failed":
        return {
          classes: `${baseClasses} bg-destructive-subtle text-destructive-ink`,
          label: "Failed"
        };
      default:
        return {
          classes: `${baseClasses} bg-muted text-muted-foreground`,
          label: entry.status
        };
    }
  };

  const statusBadge = getStatusBadge();

  return (
    <div className="rounded-lg border border-border bg-card p-3 hover:shadow-sm transition-shadow">
      <div className="flex items-center justify-between gap-3">
        {/* Main info */}
        <div className="flex-1 min-w-0">
          {/* Deployment name */}
          <div className="flex items-center gap-2 mb-1">
            <Folder className="h-4 w-4 text-muted-foreground shrink-0" />
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
            className="text-xs text-muted-foreground font-mono truncate text-left"
            title={entry.folder_path}
          >
            <bdi>{entry.folder_path}</bdi>
          </p>
        </div>

        {/* Inline actions — fast clicking, no dropdown hop. */}
        <div className="flex items-center gap-1 shrink-0">
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() => setShowDetails((v) => !v)}
            title={showDetails ? "Hide details" : "View details"}
          >
            {showDetails ? (
              <EyeOff className="h-4 w-4" />
            ) : (
              <Eye className="h-4 w-4" />
            )}
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-destructive-ink hover:text-destructive-ink hover:bg-destructive-subtle"
            onClick={() => onDelete(entry.id)}
            title="Delete"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Details section */}
      {showDetails && (
        <div className="mt-3 pt-3 border-t border-border">
          <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-2 text-xs">
            {/* Site */}
            <dt className="text-muted-foreground font-medium">Site:</dt>
            <dd className="text-foreground">{siteLabel}</dd>

            {/* Files */}
            <dt className="text-muted-foreground font-medium">Files:</dt>
            <dd className="text-foreground">
              {isScanning ? (
                "Scanning..."
              ) : scanResult?.total_count ? (
                `${scanResult.total_count} (${scanResult.image_count} images, ${scanResult.video_count} videos)`
              ) : (
                "No files"
              )}
            </dd>

            {/* GPS (from folder scan) */}
            {!isScanning && scanResult && (
              <>
                <dt className="text-muted-foreground font-medium">GPS:</dt>
                <dd className="text-foreground">
                  {scanResult.gps_location ? "Found in EXIF" : "Not found"}
                </dd>
              </>
            )}

            {/* Date range from the folder scan (includes any datetime
                offset). Date-only, rough: the scan reads a sample of files,
                not every one, so this is an approximate span. Entries that
                opted into file dates read those instead, so the range the
                user confirmed when adding the folder is still shown here. */}
            {!isScanning && (() => {
              const fromFileMtime = entry.use_file_mtime_fallback;
              const range = formatDateSpan(
                fromFileMtime ? scanResult?.mtime_start_date ?? null : scanResult?.start_date ?? null,
                fromFileMtime ? scanResult?.mtime_end_date ?? null : scanResult?.end_date ?? null,
                entry.datetime_offset_seconds ?? 0,
              );
              if (!range) return null;
              return (
                <>
                  <dt className="text-muted-foreground font-medium">Date range:</dt>
                  <dd className="text-foreground">
                    {`roughly ${range}`}
                    {fromFileMtime && " (from file dates)"}
                  </dd>
                </>
              );
            })()}

            {/* Created */}
            <dt className="text-muted-foreground font-medium">Created:</dt>
            <dd className="text-foreground">{new Date(entry.created_at_utc).toLocaleString()}</dd>

            {/* Datetime offset (only when non-zero) */}
            {hasOffset && (
              <>
                <dt className="text-muted-foreground font-medium">Time offset:</dt>
                <dd className="text-foreground">
                  {formatOffsetSummary(
                    entry.datetime_offset_seconds ?? 0,
                    entry.camera_offsets ?? {},
                  )}
                </dd>
              </>
            )}

            {/* Paired cameras (only when on) */}
            {entry.paired_cameras && (
              <>
                <dt className="text-muted-foreground font-medium">Paired cameras:</dt>
                <dd className="text-foreground">Subfolders count as one camera</dd>
              </>
            )}

            {/* Notes */}
            {entry.notes && (
              <>
                <dt className="text-muted-foreground font-medium">Notes:</dt>
                <dd className="text-foreground whitespace-pre-wrap break-words">
                  {entry.notes}
                </dd>
              </>
            )}

            {/* Tags */}
            {hasTags && (
              <>
                <dt className="text-muted-foreground font-medium">Tags:</dt>
                <dd className="text-foreground">
                  <TagPills tags={entry.tags} maxVisible={8} />
                </dd>
              </>
            )}

            {/* Error. An analysis failure is the one error in the app the
                user cannot act on from what is on screen: the story is in
                the logs. The button is the same action Help > Export
                diagnostic report runs, put where the user already is, so
                reporting it does not start with hunting through a menu. */}
            {entry.error && (
              <>
                <dt className="text-destructive-ink font-medium">Error:</dt>
                <dd className="text-destructive-ink">
                  {entry.error}
                  <Button
                    variant="outline"
                    size="sm"
                    className="mt-2 flex h-7 items-center gap-1.5 px-2 text-xs"
                    onClick={() => void exportDiagnosticReport()}
                  >
                    <LifeBuoy className="h-3.5 w-3.5" />
                    Export diagnostic report
                  </Button>
                </dd>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
