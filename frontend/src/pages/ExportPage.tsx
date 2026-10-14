/**
 * Export page (project mode): one card, one row per export, every row the
 * same: what it is on the left, one Export button on the right. Every
 * button opens the same dialog shell (`components/export/ExportDialog`):
 * which deployments, the export's own choices, where to save it, Start.
 *
 * A dialog is mounted only while it is open, so each opening starts from
 * the defaults.
 */

import { Fragment, useState, type ComponentType } from "react";
import { useParams } from "react-router-dom";
import { Download } from "lucide-react";

import { Card, CardContent } from "../components/ui/card";
import { Button } from "../components/ui/button";
import {
  CamtrapDPExportDialog,
  SpatialExportDialog,
  SpreadsheetExportDialog,
  type ExportDialogComponentProps,
} from "../components/export/FileExportDialogs";
import { RecognitionJsonExportDialog } from "../components/export/RecognitionJsonExportDialog";
import { ImageCopiesDialog } from "../components/export/ImageCopiesDialog";
import { ProjectPackageDialog } from "../components/export/ProjectPackageDialog";

interface ExportKind {
  id: string;
  title: string;
  description: string;
  Dialog: ComponentType<ExportDialogComponentProps>;
}

const EXPORTS: ExportKind[] = [
  {
    id: "spreadsheet",
    title: "Spreadsheet",
    description:
      "Summary, counts, detections, files and deployments, including species labels and your corrections.",
    Dialog: SpreadsheetExportDialog,
  },
  {
    id: "spatial",
    title: "Spatial",
    description: "Geographic point layers for GIS tools (QGIS, ArcGIS).",
    Dialog: SpatialExportDialog,
  },
  {
    id: "camtrap-dp",
    title: "Camtrap DP",
    description:
      "A standardised, community-developed exchange format for sharing and archiving camera trap data.",
    Dialog: CamtrapDPExportDialog,
  },
  {
    id: "recognition",
    title: "Recognition file",
    description:
      "The file Timelapse reads, with your labels and corrections, written into each deployment folder.",
    Dialog: RecognitionJsonExportDialog,
  },
  {
    id: "image-copies",
    title: "Image copies",
    description:
      "Copies of your images in species folders, with your labels in the image metadata. Optional boxes and blur.",
    Dialog: ImageCopiesDialog,
  },
  {
    id: "package",
    title: "AddaxAI bundle",
    description:
      "Your labels and corrections for the selected deployments or the whole project, to open in another AddaxAI: to continue on another computer, or to let a team member check them and send them back.",
    Dialog: ProjectPackageDialog,
  },
];

export default function ExportPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [openId, setOpenId] = useState<string | null>(null);
  const open = EXPORTS.find((e) => e.id === openId);

  return (
    <div className="min-h-screen">
      <header className="border-b bg-card/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <h1 className="text-2xl font-bold tracking-tight">Export</h1>
          <p className="text-sm text-muted-foreground">
            Export project data in standardised formats
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <Card>
          <CardContent className="pt-6">
            {EXPORTS.map((kind, i) => (
              <Fragment key={kind.id}>
                {i > 0 && <div className="my-6 border-t" />}
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-[minmax(0,1fr)_12rem] sm:items-center sm:gap-4">
                  <div>
                    <h3 className="text-sm font-medium">{kind.title}</h3>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {kind.description}
                    </p>
                  </div>
                  <Button
                    type="button"
                    className="w-full gap-2"
                    onClick={() => setOpenId(kind.id)}
                  >
                    <Download className="h-4 w-4" />
                    Export…
                  </Button>
                </div>
              </Fragment>
            ))}
          </CardContent>
        </Card>
      </main>

      {projectId && open && (
        <open.Dialog projectId={projectId} onClose={() => setOpenId(null)} />
      )}
    </div>
  );
}
