/**
 * Import many sites at once from a CSV file.
 *
 * Everything about the flow lives in CsvImportDialog. This file is only the
 * wording, the columns, the example file and the two requests.
 */

import { useQueryClient } from "@tanstack/react-query";

import { sitesApi } from "@/api/sites";
import type { SiteImportRow } from "@/api/types";
import { invalidateProjectData } from "@/lib/invalidate-project";
import {
  CsvImportDialog,
  type CsvColumnHelp,
} from "@/components/ui/csv-import-dialog";

const COLUMNS: CsvColumnHelp[] = [
  { name: "name", help: "The site name. It must be unique within this project." },
  // The numeric columns each repeat the dot rule. It is a lookup table, not
  // prose, so a user scanning for the one column they are unsure about finds
  // the rule there rather than having to read the whole list.
  {
    name: "latitude",
    help: "Decimal degrees between -90 and 90, for example 52.0907. Use a dot, not a comma.",
  },
  {
    name: "longitude",
    help: "Decimal degrees between -180 and 180, for example 5.1214. Use a dot, not a comma.",
  },
  {
    name: "elevation_m",
    optional: true,
    help: "Height above sea level in meters, for example 1620 or 1702.5. No thousands separator.",
  },
  {
    name: "habitat_type",
    optional: true,
    help: "Free text, for example forest or grassland.",
  },
  { name: "notes", optional: true, help: "Free text for your own records." },
];

// Every row here is deliberate. The file can only teach through its data,
// because CSV has no comment syntax and an extra explaining column would be
// rejected as unrecognised. In order the rows show: every column filled, an
// empty notes at the end of the line, an empty elevation between two commas,
// all three optional columns empty at once (the smallest a row can be, with
// short coordinates so nobody thinks four decimals are required), and a value
// containing a comma wrapped in double quotes.
//
// Coordinates are negative and use a dot as the decimal separator. A comma
// decimal is the mistake people actually make, so every example contradicts it.
//
// The site names match the deployment example, so a user who runs both imports
// in order sees the link between the two files work.
const EXAMPLE_CSV = `name,latitude,longitude,elevation_m,habitat_type,notes
Kifaru Plains north,-1.4061,35.0117,1620,Savannah,Camera on an acacia at the trail junction
River crossing,-1.4133,35.0208,1585,Riparian,
Acacia thicket,-1.4210,35.0290,,Woodland,Dense cover in the wet season
Salt lick,-1.44,35.04,,,
Ridge viewpoint,-1.4188,35.0412,1702.5,Rocky outcrop,"Steep approach, use the north track"
`;

interface ImportSitesDialogProps {
  projectId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ImportSitesDialog({
  projectId,
  open,
  onOpenChange,
}: ImportSitesDialogProps) {
  const queryClient = useQueryClient();

  return (
    <CsvImportDialog<SiteImportRow>
      open={open}
      onOpenChange={onOpenChange}
      title="Import sites from CSV"
      description="Add many sites at once from a spreadsheet. Save your sheet as a CSV file and choose it here. You will see what will be added before anything is saved."
      noun={{ one: "site", many: "sites" }}
      columns={COLUMNS}
      exampleFilename="example-sites.csv"
      exampleCsv={EXAMPLE_CSV}
      onPreview={(file) => sitesApi.importPreview(projectId, file)}
      onImport={async (file) => {
        const result = await sitesApi.importCsv(projectId, file);
        // Only when something was actually written. Refetching after a
        // refused import made the table visibly reload, which told the user
        // their import had worked when it had not.
        if (result.imported > 0) {
          // Site coordinates feed the map, the sun bands and the exports, so
          // refresh everything, the same as AddSiteModal does.
          invalidateProjectData(queryClient, projectId);
        }
        return result;
      }}
      // Every column the CSV can carry, in the order the columns are
      // documented above. The preview is the user's only chance to check
      // that a value landed where they meant it to, so leaving one out
      // makes the preview quietly lie about what was read.
      renderRow={(row) => (
        <>
          <span className="w-40 shrink-0 truncate font-medium" title={row.name}>
            {row.name}
          </span>
          <span className="shrink-0 tabular-nums text-muted-foreground">
            {row.latitude}, {row.longitude}
          </span>
          <span className="w-16 shrink-0 text-right tabular-nums text-muted-foreground">
            {row.elevation_m === null ? "" : `${row.elevation_m} m`}
          </span>
          <span
            className="w-24 shrink-0 truncate text-muted-foreground"
            title={row.habitat_type ?? ""}
          >
            {row.habitat_type ?? ""}
          </span>
          <span
            className="flex-1 truncate text-muted-foreground"
            title={row.notes ?? ""}
          >
            {row.notes ?? ""}
          </span>
        </>
      )}
    />
  );
}
