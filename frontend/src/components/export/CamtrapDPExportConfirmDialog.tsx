/**
 * Pre-flight dialog for CamtrapDP export.
 *
 * Always shown before the download so the user confirms the data fits
 * the CamtrapDP schema's one-camera-one-location-one-period rule. If
 * any deployment in the project has no camera site assigned, a second
 * warning block flags that those will be excluded from the output
 * (CamtrapDP requires coordinates per deployment row). Both blocks
 * link inline to the Deployments page so the user can fix the data.
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

interface CamtrapDPExportConfirmDialogProps {
  projectId: string;
  /** Number of deployments in the project with no camera site. */
  noSiteCount: number;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onProceed: () => void;
}

export function CamtrapDPExportConfirmDialog({
  projectId,
  noSiteCount,
  open,
  onOpenChange,
  onProceed,
}: CamtrapDPExportConfirmDialogProps) {
  const depWord = noSiteCount === 1 ? "deployment" : "deployments";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Before you export Camtrap DP</DialogTitle>
          <DialogDescription>
            Confirm your data meets the Camtrap DP schema before
            exporting.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3 text-sm">
          <div
            className="rounded-md border px-3 py-2"
            style={{ backgroundColor: "#71b7ba22", borderColor: "#71b7ba" }}
          >
            Camtrap DP expects every deployment in the database to
            represent one camera, one location, and one continuous
            period. Basically, exactly what you would get on a single
            SD card. If this criterion is not met, the export publishes
            schema-valid rows that describe mixed data as a single
            camera-period, which would produce erroneous data. You can split existing
            deployments into smaller chunks on the{" "}
            <Link
              to={`/projects/${projectId}/deployments`}
              className="underline underline-offset-2"
              onClick={() => onOpenChange(false)}
            >
              Deployments page
            </Link>
            .
          </div>

          {noSiteCount > 0 && (
            <div
              className="rounded-md border px-3 py-2"
              style={{ backgroundColor: "#71b7ba22", borderColor: "#71b7ba" }}
            >
              {noSiteCount} {depWord} in this project have no camera
              site. They will be excluded from this export because
              Camtrap DP requires lat/lon for every deployment row. You
              can add sites to the existing deployments on the{" "}
              <Link
                to={`/projects/${projectId}/deployments?site_ids=${NO_SITE_SENTINEL}`}
                className="underline underline-offset-2"
                onClick={() => onOpenChange(false)}
              >
                Deployments page
              </Link>
              .
            </div>
          )}
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
            Export
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
