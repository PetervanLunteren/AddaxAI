/**
 * Confirmation dialog shown before a spatial export (GeoJSON, Shapefile,
 * GeoPackage) when any deployment in the project has no camera site
 * assigned. Those deployments will be dropped from the output because
 * the format requires coordinates. The card body links to the
 * Deployments page (pre-filtered to no-site) for the user to fix the
 * underlying data before retrying.
 */

import { Link } from "react-router-dom";

import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { NO_SITE_SENTINEL } from "../../lib/filter-url";

interface SpatialExportConfirmDialogProps {
  projectId: string;
  count: number;
  formatLabel: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onProceed: () => void;
}

export function SpatialExportConfirmDialog({
  projectId,
  count,
  formatLabel,
  open,
  onOpenChange,
  onProceed,
}: SpatialExportConfirmDialogProps) {
  const depWord = count === 1 ? "deployment" : "deployments";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Some deployments have no location</DialogTitle>
          <DialogDescription>
            Spatial formats need lat/lon for every row.
          </DialogDescription>
        </DialogHeader>

        <div
          className="rounded-md border px-3 py-2 text-sm"
          style={{ backgroundColor: "#71b7ba22", borderColor: "#71b7ba" }}
        >
          {count} {depWord} in this project have no camera site, so they
          have no coordinates. They will be excluded from the{" "}
          {formatLabel} export. You can add sites to the existing
          deployments on the{" "}
          <Link
            to={`/projects/${projectId}/deployments?site_ids=${NO_SITE_SENTINEL}`}
            className="underline underline-offset-2"
            onClick={() => onOpenChange(false)}
          >
            Deployments page
          </Link>
          .
        </div>

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            type="button"
            onClick={() => {
              onOpenChange(false);
              onProceed();
            }}
          >
            Export anyway
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
