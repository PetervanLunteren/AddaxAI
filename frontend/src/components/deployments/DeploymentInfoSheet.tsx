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
import { FolderOpen, FolderTree, Pencil, Trash2 } from "lucide-react";

import { deploymentsApi, type DeploymentInfo } from "../../api/deployments";
import { formatCameraDateTime } from "../../lib/datetime";
import { isElectron } from "../../lib/platform";
import { normalizeLabel } from "../../utils/labels";
import { Button } from "../ui/button";
import {
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
            <FolderTree className="mr-2 h-4 w-4" />
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
      <Section title="Location">
        <Row label="Path" value={info.folder_path ?? <NotSet />} />
        <Row label="Site" value={info.site_name} />
      </Section>

      <Separator />

      <Section title="Dates">
        <Row label="First / last capture" value={firstLast ?? <NotSet />} />
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
          label="Observations (MaxN)"
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

      <Section title="Detection categories">
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
                  {normalizeLabel(s.display_name ?? s.label)}
                </span>
                <span className="tabular-nums">
                  {s.count.toLocaleString()}
                </span>
              </li>
            ))}
          </ol>
        )}
      </Section>

    </div>
  );
}

