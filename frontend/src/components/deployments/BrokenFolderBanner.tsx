/**
 * Banner for deployments whose files can no longer be found, shown on the
 * pages where the damage is actually visible: Labels and Counts.
 *
 * A missing folder does not break the data, only the pictures. Every count,
 * label and export stays correct, so the app looks healthy while the grid
 * fills with grey tiles. The startup toast and the sidebar dot both report
 * this, but the toast fires on entering the project (minutes before anyone
 * reaches a grid, and it is dismissible) and the dot sits on a nav item in
 * the config group that nobody looks at mid-task. Neither is on screen at
 * the moment the tiles go grey, which is what sent users to support instead
 * of to the Deployments page.
 *
 * Not dismissible: it describes a condition that persists until the folder
 * is reconnected, and it clears itself the moment that happens because
 * `useBrokenDeployments` reads the same query every relink invalidates.
 *
 * Project mode only. Mount it from `LabelsPage` / `CountsPage`, never from
 * the shared `LabelsView` / `VerifyView`, which folder runs also render:
 * a folder run has no Deployments page for the button to reach.
 */

import { useNavigate } from "react-router-dom";
import { useBrokenDeployments } from "../../hooks/useBrokenDeployments";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";

interface BrokenFolderBannerProps {
  projectId: string;
}

export function BrokenFolderBanner({ projectId }: BrokenFolderBannerProps) {
  const navigate = useNavigate();
  const broken = useBrokenDeployments(projectId);

  if (broken.length === 0) return null;

  const count = broken.length;

  return (
    <Callout
      variant="error"
      title="Some photos can't be shown"
      className="mb-6"
      action={
        <Button
          size="sm"
          onClick={() => navigate(`/projects/${projectId}/deployments`)}
        >
          Reconnect
        </Button>
      }
    >
      {count === 1
        ? "1 deployment folder has moved, been renamed, or is disconnected, so its images show up empty here. Your labels and counts are safe."
        : `${count} deployment folders have moved, been renamed, or are disconnected, so their images show up empty here. Your labels and counts are safe.`}
    </Callout>
  );
}
