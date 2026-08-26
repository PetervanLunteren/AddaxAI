/**
 * Import many deployment folders into the queue at once from a CSV file.
 *
 * Everything about the flow lives in CsvImportDialog. This file is only the
 * wording, the columns, the example file and the two requests.
 *
 * Note what this creates: queue entries, not deployments. The deployment row
 * is built by the worker when the queue runs, which is also where the
 * capture dates come from, so the CSV carries no dates.
 */

import { useQueryClient } from "@tanstack/react-query";

import { deploymentQueueApi } from "@/api/deployment-queue";
import type { DeploymentImportRow } from "@/api/types";
import {
  CsvImportDialog,
  type CsvColumnHelp,
} from "@/components/ui/csv-import-dialog";
import { StartTruncatedPath } from "@/components/ui/start-truncated-path";

const COLUMNS: CsvColumnHelp[] = [
  {
    name: "folder",
    // "one camera period" plus a deployment_001 example, so nobody reads this
    // as "point at everything that camera ever recorded". Both path styles
    // named, matching the wording of the not-a-full-path error, so a Windows
    // user is not left guessing from a macOS example.
    help: "Full path to the folder with one camera period's images or videos, for example /Volumes/Data/CAM01/deployment_001 or D:\\Data\\CAM01\\deployment_001. Subfolders are included.",
  },
  {
    name: "site",
    optional: true,
    help: "The name of a site that already exists in this project. Leave it empty to set it later.",
  },
  { name: "notes", optional: true, help: "Free text for your own records." },
  {
    name: "paired_cameras",
    optional: true,
    help: "true when the folder holds one subfolder per camera and the cameras are dependent, triggering on the same animals. Their files form one event and the trap nights count once. Leave it empty for false.",
  },
];

// Every row here is deliberate. The file can only teach through its data,
// because CSV has no comment syntax and an extra explaining column would be
// rejected as unrecognised. In order the rows show: a path containing a space
// with no quotes around it (the artifact that sends people to quote paths and
// then fail), a filled notes, the smallest row, the same site used again for a
// second period (so it is clear one site can hold many deployments), and an
// empty site, which is allowed and means the site is set later, and a paired
// deployment of dependent cameras, one subfolder per camera.
//
// Every path ends in a deployment_NNN folder on purpose. A path like
// .../2026/river-crossing reads as a whole site's footage, which nudges people
// into pointing at one folder per site, and from there into listing a folder
// and its subfolders in the same file. One row is one camera period.
//
// The site names match the site example, so a user who runs both imports in
// order sees the link between the two files work.
const EXAMPLE_CSV = `folder,site,notes,paired_cameras
/Volumes/Field data/Kifaru Plains north/deployment_001,Kifaru Plains north,,
/Volumes/Field data/River crossing/deployment_001,River crossing,SD card was nearly full,
/Volumes/Field data/Acacia thicket/deployment_001,Acacia thicket,,
/Volumes/Field data/River crossing/deployment_002,River crossing,"Second period, same camera",
/Volumes/Field data/unsorted/deployment_001,,Site not decided yet,
/Volumes/Field data/Waterhole/deployment_001,Waterhole,Two cameras facing each other,true
`;

interface ImportDeploymentsDialogProps {
  projectId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ImportDeploymentsDialog({
  projectId,
  open,
  onOpenChange,
}: ImportDeploymentsDialogProps) {
  const queryClient = useQueryClient();

  return (
    <CsvImportDialog<DeploymentImportRow>
      open={open}
      onOpenChange={onOpenChange}
      title="Import deployments from CSV"
      description="Add many folders to the queue at once from a spreadsheet. Save your sheet as a CSV file and choose it here. You will see what will be added before anything is saved, and nothing is analysed until you run the queue."
      noun={{ one: "deployment", many: "deployments" }}
      columns={COLUMNS}
      exampleFilename="example-deployments.csv"
      exampleCsv={EXAMPLE_CSV}
      onPreview={(file) => deploymentQueueApi.importPreview(projectId, file)}
      onImport={async (file) => {
        const result = await deploymentQueueApi.importCsv(projectId, file);
        // Only when something was actually written, so a refused import does
        // not look like it did anything.
        if (result.imported > 0) {
          // Queue entries are not deployments yet, so only the queue needs to
          // refresh. Same as AddDeploymentCard.
          queryClient.invalidateQueries({
            queryKey: ["deployment-queue", projectId],
          });
        }
        return result;
      }}
      // A deployment CSV can only name sites that already exist, so an
      // unknown name is a dead end unless the user is shown the way out.
      // The link sits on the words that describe the way out, inside the
      // message it belongs to, rather than as a separate button that could
      // just as easily have belonged to a missing folder.
      // SITE_NOT_FOUND in backend/app/services/csv_import_deployments.py owns
      // this phrase, and a test there pins it so a reword cannot silently
      // drop the link.
      problemLink={{
        column: "site",
        phrase: "Import your sites first",
        to: `/projects/${projectId}/sites`,
      }}
      // Every column the CSV can carry, plus the media counts the server
      // worked out. The preview is the user's only chance to check that a
      // value landed where they meant it to, so leaving one out makes the
      // preview quietly lie about what was read.
      renderRow={(row) => (
        <>
          <StartTruncatedPath className="flex-1 font-mono text-xs" path={row.folder} />
          <span
            className="w-28 shrink-0 truncate text-muted-foreground"
            title={row.site ?? ""}
          >
            {row.site ?? "No site"}
          </span>
          <span className="w-24 shrink-0 text-right tabular-nums text-muted-foreground">
            {row.image_count} img, {row.video_count} vid
          </span>
          <span className="w-16 shrink-0 text-muted-foreground">
            {row.paired_cameras ? "Paired" : ""}
          </span>
          <span
            className="w-28 shrink-0 truncate text-muted-foreground"
            title={row.notes ?? ""}
          >
            {row.notes ?? ""}
          </span>
        </>
      )}
    />
  );
}
