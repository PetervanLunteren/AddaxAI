/**
 * Deployment info sheet.
 *
 * Read-only side drawer triggered from the Deployments table's row
 * dropdown. Shows investigation-level metadata that isn't in the row:
 * folder path, file-type split, size on disk, verification progress,
 * event / observation counts, detection-category breakdown, top
 * species, trap nights + rate, mean detection / classification
 * confidence, first and last capture timestamps. Plus jump-to links
 * for Verify (filtered by this deployment's site) and Dashboard.
 */

import { useQuery } from "@tanstack/react-query";
import { FolderOpen, Pencil, Scissors, Trash2 } from "lucide-react";

import { deploymentsApi, type DeploymentInfo } from "../../api/deployments";
import { formatCameraDateTime } from "../../lib/datetime";
import { isElectron } from "../../lib/platform";
import { resolveSpeciesName } from "../../lib/species-name-mode";
import { Button } from "../ui/button";
import {
  IdWithCopy,
  NotSet,
  Row,
  Section,
  formatBytes,
} from "../ui/info-sheet-parts";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "../ui/sheet";
import { Separator } from "../ui/separator";

interface DeploymentInfoSheetProps {
  deploymentId: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Opens the Edit dialog for this deployment. Parent closes the sheet. */
  onEdit?: () => void;
  /** Opens the Split dialog for this deployment. Parent closes the sheet. */
  onSplit?: () => void;
  /** Opens the Delete confirmation for this deployment. Parent closes the sheet. */
  onDelete?: () => void;
}

export function DeploymentInfoSheet({
  deploymentId,
  open,
  onOpenChange,
  onEdit,
  onSplit,
  onDelete,
}: DeploymentInfoSheetProps) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["deployments", deploymentId, "info"],
    queryFn: () => deploymentsApi.getInfo(deploymentId!),
    enabled: open && !!deploymentId,
  });

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full overflow-y-auto sm:max-w-xl">
        <SheetHeader>
          <SheetTitle>Deployment info</SheetTitle>
          <SheetDescription>
            Investigation snapshot for this deployment.
          </SheetDescription>
        </SheetHeader>

        <div className="mt-4 grid grid-cols-2 gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={onEdit}
            disabled={!onEdit}
          >
            <Pencil className="mr-2 h-4 w-4" />
            Edit
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={() =>
              data?.folder_path && window.electronAPI?.openPath(data.folder_path)
            }
            disabled={!data?.folder_path}
            title={
              !data?.folder_path
                ? "No folder path set"
                : isElectron()
                  ? "Open in file explorer"
                  : "Only works in the desktop app"
            }
          >
            <FolderOpen className="mr-2 h-4 w-4" />
            Open folder
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={onSplit}
            disabled={!onSplit || !data?.folder_path}
          >
            <Scissors className="mr-2 h-4 w-4" />
            Split
          </Button>
          <Button
            type="button"
            variant="outline"
            className="text-destructive hover:text-destructive"
            onClick={onDelete}
            disabled={!onDelete}
          >
            <Trash2 className="mr-2 h-4 w-4" />
            Delete
          </Button>
        </div>

        {isLoading && (
          <p className="mt-6 text-sm text-muted-foreground">Loading...</p>
        )}
        {isError && (
          <p className="mt-6 text-sm text-destructive">
            Could not load deployment info.
          </p>
        )}
        {data && <InfoBody info={data} />}
      </SheetContent>
    </Sheet>
  );
}

function InfoBody({ info }: { info: DeploymentInfo }) {
  const firstLast =
    info.first_captured_at_local && info.last_captured_at_local
      ? `${formatCameraDateTime(info.first_captured_at_local)} to ${formatCameraDateTime(info.last_captured_at_local)}`
      : null;

  return (
    <div className="mt-6 space-y-6 text-sm">
      <Section title="Reference">
        <Row
          label="Deployment ID"
          value={<IdWithCopy value={info.deployment_id} />}
        />
      </Section>

      <Separator />

      <Section title="Location">
        <Row label="Path" value={info.folder_path ?? <NotSet />} />
        <Row label="Site" value={info.site_name} />
      </Section>

      <Separator />

      <Section title="Dates">
        <Row label="First / last file" value={firstLast ?? <NotSet />} />
        <Row
          label="Trap nights"
          value={
            info.trap_nights === null ? (
              <NotSet />
            ) : (
              info.trap_nights.toLocaleString()
            )
          }
        />
      </Section>

      <Separator />

      <Section title="Files">
        <Row
          label="Count"
          value={
            <>
              {info.files.total.toLocaleString()} total
              {" ("}
              {info.files.images.toLocaleString()} images,{" "}
              {info.files.videos.toLocaleString()} videos)
            </>
          }
        />
        <Row label="Size on disk" value={formatBytes(info.total_size_bytes)} />
        <Row
          label="Verified"
          value={
            <span>
              {info.verification.verified.toLocaleString()} /{" "}
              {info.verification.total.toLocaleString()}
              {info.verification.total > 0 && (
                <span className="ml-2 text-muted-foreground">
                  (
                  {Math.round(
                    (info.verification.verified / info.verification.total) *
                      100,
                  )}
                  %)
                </span>
              )}
            </span>
          }
        />
      </Section>

      <Separator />

      <Section title="Observations">
        <Row label="Events" value={info.event_count.toLocaleString()} />
        <Row
          label="Observations"
          value={info.observation_count.toLocaleString()}
        />
        <Row
          label="Rate per 100 trap nights"
          value={
            info.observation_rate_per_100_trap_nights === null ? (
              <NotSet />
            ) : (
              info.observation_rate_per_100_trap_nights.toFixed(2)
            )
          }
        />
      </Section>

      <Separator />

      <Section title="Observation categories">
        <Row
          label="Animal"
          value={info.detection_categories.animal.toLocaleString()}
        />
        <Row
          label="Person"
          value={info.detection_categories.person.toLocaleString()}
        />
        <Row
          label="Vehicle"
          value={info.detection_categories.vehicle.toLocaleString()}
        />
        <Row
          label="Empty"
          value={info.detection_categories.empty.toLocaleString()}
        />
      </Section>

      <Separator />

      <Section title="Top species">
        {info.top_species.length === 0 ? (
          <p className="text-muted-foreground">No classified observations.</p>
        ) : (
          <ol className="space-y-1">
            {info.top_species.map((s, i) => (
              <li
                key={s.label}
                className="flex items-baseline justify-between gap-4"
              >
                <span className="truncate">
                  <span className="tabular-nums text-muted-foreground">
                    {i + 1}.
                  </span>{" "}
                  {resolveSpeciesName(s)}
                </span>
                <span className="tabular-nums">
                  {s.count.toLocaleString()}
                </span>
              </li>
            ))}
          </ol>
        )}
      </Section>

      {info.warnings && info.warnings.length > 0 && (
        <>
          <Separator />
          <DeploymentWarningsSection warnings={info.warnings} />
        </>
      )}
    </div>
  );
}

function describeWarningType(type: string): string {
  switch (type) {
    case "video_processing_failure":
      return "Could not be read";
    default:
      return type;
  }
}

function DeploymentWarningsSection({
  warnings,
}: {
  warnings: NonNullable<DeploymentInfo["warnings"]>;
}) {
  // Files with no capture date were still detected and classified and live
  // in the database; they are NOT skipped. Keep them separate from the
  // genuinely-skipped files (corrupt / unreadable) so the user isn't told a
  // processed file was dropped.
  const dateless = warnings.filter((w) => w.type === "missing_timestamp");
  const skipped = warnings.filter((w) => w.type !== "missing_timestamp");

  // Group skipped files by type so the user reads "3 files: could not be
  // read" rather than a flat list. Inside each group the paths are listed.
  const groups = new Map<string, typeof warnings>();
  for (const w of skipped) {
    const existing = groups.get(w.type) ?? [];
    existing.push(w);
    groups.set(w.type, existing);
  }

  return (
    <>
      {skipped.length > 0 && (
        <Section title="Skipped files">
          <p className="mb-3 text-xs text-muted-foreground">
            {skipped.length === 1
              ? "1 file was skipped during analysis."
              : `${skipped.length} files were skipped during analysis.`}{" "}
            They are not in the database and won't appear on the dashboard.
          </p>
          {/* Cap visual height. Real-world deployments are 10k+ files, so a
              skip list can be hundreds of rows long; we cap the vertical
              footprint and let the user scroll. Long paths get a per-row
              horizontal scrollbar so they're not truncated. */}
          <div className="max-h-72 space-y-3 overflow-auto rounded-md border bg-muted/30 p-3">
            {Array.from(groups.entries()).map(([type, items]) => (
              <div key={type}>
                <div className="text-xs font-medium" style={{ color: "#882000" }}>
                  {describeWarningType(type)} ({items.length})
                </div>
                <ul className="mt-1 space-y-0.5 text-xs text-muted-foreground">
                  {items.map((w, idx) => (
                    <li
                      key={`${w.path}-${idx}`}
                      className="whitespace-nowrap font-mono"
                    >
                      {w.path}
                      {w.reason && (
                        <span className="ml-1 text-muted-foreground/60">
                          — {w.reason}
                        </span>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </Section>
      )}

      {skipped.length > 0 && dateless.length > 0 && <Separator />}

      {dateless.length > 0 && (
        <Section title="Files without a date">
          <p className="text-xs text-muted-foreground">
            {dateless.length === 1
              ? "1 file had no capture date."
              : `${dateless.length} files had no capture date.`}{" "}
            They were detected and classified and are in the database, just left
            out of time-based stats, charts, and trap-night effort.
          </p>
        </Section>
      )}
    </>
  );
}

