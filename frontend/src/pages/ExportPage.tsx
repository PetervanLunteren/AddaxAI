/**
 * Export page (project mode) — one Card with a row per export.
 *
 * Each row reads left to right: title + caption (half width), an optional
 * filters slot (quarter width, the site/deployment scope), and one control
 * (quarter width): a "Download" dropdown of formats, or a plain button for
 * a row with a single action (the recognition file, which writes into the
 * deployment folders). Layout follows AddaxAI Connect's ExportsPage; the
 * filters column is WebUI-specific and reserved for more filters later.
 */

import React, { useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  AlertCircle,
  ChevronDown,
  Download,
  FileJson,
  Images,
  Loader2,
  type LucideIcon,
} from "lucide-react";

import { Card, CardContent } from "../components/ui/card";
import { Button } from "../components/ui/button";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "../components/ui/dropdown-menu";
import {
  exportApi,
  type ExportScope,
  type ObservationFormat,
  type SpatialFormat,
} from "../api/export";
import { projectsApi } from "../api/projects";
import { useNoSiteDeployments } from "../hooks/useNoSiteDeployments";
import { ExportScopeSelect } from "../components/export/ExportScopeSelect";
import { SpatialExportConfirmDialog } from "../components/export/SpatialExportConfirmDialog";
import { CamtrapDPExportConfirmDialog } from "../components/export/CamtrapDPExportConfirmDialog";
import { CamtrapDPProgressModal } from "../components/export/CamtrapDPProgressModal";
import { RecognitionJsonExportDialog } from "../components/export/RecognitionJsonExportDialog";
import { ImageCopiesDialog } from "../components/export/ImageCopiesDialog";
import { downloadBlob } from "../lib/download";

interface DownloadOption {
  value: string;
  label: string;
}

const OBSERVATION_OPTIONS: DownloadOption[] = [
  { value: "csv", label: "CSV" },
  { value: "tsv", label: "TSV" },
  { value: "xlsx", label: "XLSX" },
];

const SPATIAL_OPTIONS: DownloadOption[] = [
  { value: "geojson", label: "GeoJSON" },
  { value: "shapefile", label: "Shapefile" },
  { value: "gpkg", label: "GeoPackage" },
];

const CAMTRAP_MEDIA_OPTIONS: DownloadOption[] = [
  { value: "metadata", label: "Metadata only" },
  { value: "thumbnails", label: "Include thumbnails" },
];


function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return "Export failed";
}

function DownloadDropdown({
  options,
  onSelect,
  isLoading,
  disabled = false,
}: {
  options: DownloadOption[];
  onSelect: (value: string) => void;
  isLoading: boolean;
  /** Nothing to export. Separate from isLoading so the button keeps its
   *  normal label instead of claiming an export is in flight. */
  disabled?: boolean;
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          type="button"
          disabled={isLoading || disabled}
          className="w-full gap-2"
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Preparing export...
            </>
          ) : (
            <>
              <Download className="h-4 w-4" />
              Download
              <ChevronDown className="h-4 w-4" />
            </>
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {options.map((opt) => (
          <DropdownMenuItem key={opt.value} onClick={() => onSelect(opt.value)}>
            {opt.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function ExportRow({
  title,
  description,
  filters,
  options,
  isLoading,
  onSelect,
  error,
  disabled = false,
  action,
}: {
  title: string;
  description: React.ReactNode;
  filters?: React.ReactNode;
  options: DownloadOption[];
  isLoading: boolean;
  onSelect: (value: string) => void;
  error: string | null;
  disabled?: boolean;
  /** A single action instead of the format dropdown. */
  action?: { label: string; onClick: () => void; icon?: LucideIcon };
}) {
  const ActionIcon = action?.icon ?? FileJson;
  return (
    <div>
      {/* Text takes the room; the picker and the button each get a fixed
          slot so every row's controls sit at the same place and every
          button has the same width. A row without a picker leaves its
          slot empty rather than shifting the button. */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-[minmax(0,1fr)_12rem_12rem] sm:items-center sm:gap-4">
        <div>
          <h3 className="text-sm font-medium">{title}</h3>
          <p className="mt-1 text-sm text-muted-foreground">{description}</p>
        </div>
        <div>{filters}</div>
        <div>
          {action ? (
            <Button
              type="button"
              className="w-full gap-2"
              disabled={disabled}
              onClick={action.onClick}
            >
              <ActionIcon className="h-4 w-4" />
              {action.label}
            </Button>
          ) : (
            <DownloadDropdown
              options={options}
              onSelect={onSelect}
              isLoading={isLoading}
              disabled={disabled}
            />
          )}
        </div>
      </div>
      {error && (
        <div className="mt-3 flex items-center gap-2 rounded-md bg-destructive/10 p-3 text-sm text-destructive">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}

export default function ExportPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const { data: noSite } = useNoSiteDeployments(projectId);
  // Files with no capture date cannot be represented in CamtrapDP (the
  // schema requires a timestamp per record); the confirm dialog warns
  // that they are left out.
  const { data: noDate } = useQuery({
    queryKey: ["files-without-date", projectId],
    queryFn: () => projectsApi.getFilesWithoutDate(projectId!),
    enabled: !!projectId,
    staleTime: 30_000,
  });

  // Export scope reported by the picker. Undefined = whole project.
  // Applies only to the Spreadsheet export, not spatial or CamTrap DP.
  const [tableScope, setTableScope] = useState<ExportScope | undefined>(undefined);
  // The recognition-file row has its own picker: it writes into the
  // chosen deployments' folders, and a project's folders sit on several
  // drives, so which ones matters more here than for a download.
  const [jsonScope, setJsonScope] = useState<ExportScope | undefined>(undefined);
  const [jsonDialogOpen, setJsonDialogOpen] = useState(false);
  const [mediaDialogOpen, setMediaDialogOpen] = useState(false);

  // Spatial / CamTrap DP carry a selection into their confirm dialogs.
  const [spatialFormat, setSpatialFormat] = useState<SpatialFormat>("geojson");
  const [dpIncludesThumbnails, setDpIncludesThumbnails] = useState(false);

  const [tableLoading, setTableLoading] = useState(false);
  const [spatialLoading, setSpatialLoading] = useState(false);
  const [dpLoading, setDpLoading] = useState(false);

  const [tableError, setTableError] = useState<string | null>(null);
  const [spatialError, setSpatialError] = useState<string | null>(null);
  const [dpError, setDpError] = useState<string | null>(null);

  const [spatialConfirmOpen, setSpatialConfirmOpen] = useState(false);
  const [dpConfirmOpen, setDpConfirmOpen] = useState(false);
  const [dpJobId, setDpJobId] = useState<string | null>(null);

  const noSiteCount = noSite?.count ?? 0;

  // The picker reports an empty deployment list when the user unchecks
  // everything. That cannot be sent as a filter: `withScope` omits an
  // empty array, and the backend reads a missing (or blank) scope as
  // "whole project", so asking for nothing would hand back everything.
  // Rather than teach both layers to carry an empty scope for a result
  // nobody wants, don't offer the download. The picker sitting next to
  // the button already reads "Nothing selected", so the disabled state
  // needs no wording of its own.
  const nothingSelected = tableScope?.deploymentIds?.length === 0;
  const jsonNothingSelected = jsonScope?.deploymentIds?.length === 0;

  // One "Spreadsheet" download covering five tables. XLSX is a single
  // workbook; CSV / TSV save one file per table (browsers may show a
  // one-time "allow multiple downloads" prompt outside Electron).
  const handleDownloadSpreadsheet = async (value: string) => {
    if (!projectId) return;
    const format = value as ObservationFormat;
    setTableLoading(true);
    setTableError(null);
    try {
      if (format === "xlsx") {
        const blob = await exportApi.downloadSpreadsheetXlsx(projectId, tableScope);
        downloadBlob(blob, "addaxai-spreadsheet.xlsx");
      } else {
        // Same order as the XLSX sheets (build_spreadsheet_sheets) and as
        // the docs: broadest question first. Keep the three in step.
        const summary = await exportApi.downloadSummary(projectId, format, tableScope);
        downloadBlob(summary, `addaxai-summary.${format}`);
        const counts = await exportApi.downloadObservations(projectId, format, tableScope);
        downloadBlob(counts, `addaxai-counts.${format}`);
        const detections = await exportApi.downloadDetections(projectId, format, tableScope);
        downloadBlob(detections, `addaxai-detections.${format}`);
        const files = await exportApi.downloadFiles(projectId, format, tableScope);
        downloadBlob(files, `addaxai-files.${format}`);
        const deployments = await exportApi.downloadDeployments(projectId, format, tableScope);
        downloadBlob(deployments, `addaxai-deployments.${format}`);
      }
    } catch (err) {
      setTableError(errorMessage(err));
    } finally {
      setTableLoading(false);
    }
  };

  // Spatial: pick a format from the dropdown, then confirm when any
  // deployment has no site (those rows are dropped from the export).
  const handleSpatialSelect = (value: string) => {
    if (!projectId) return;
    const fmt = value as SpatialFormat;
    setSpatialFormat(fmt);
    if (noSiteCount > 0) {
      setSpatialConfirmOpen(true);
    } else {
      void runSpatialExport(fmt);
    }
  };

  const runSpatialExport = async (fmt: SpatialFormat) => {
    if (!projectId) return;
    setSpatialLoading(true);
    setSpatialError(null);
    const ext: Record<SpatialFormat, string> = {
      geojson: "geojson",
      shapefile: "zip",
      gpkg: "gpkg",
    };
    try {
      const blob = await exportApi.downloadSpatial(projectId, fmt);
      downloadBlob(blob, `addaxai-spatial.${ext[fmt]}`);
    } catch (err) {
      setSpatialError(errorMessage(err));
    } finally {
      setSpatialLoading(false);
    }
  };

  // CamTrap DP: always open the pre-flight dialog — the format has hard
  // schema requirements (one camera / location / period per deployment)
  // the user should explicitly confirm.
  const handleCamtrapSelect = (value: string) => {
    if (!projectId) return;
    setDpIncludesThumbnails(value === "thumbnails");
    setDpConfirmOpen(true);
  };

  const runCamtrapDPExport = async (includeThumbnails: boolean) => {
    if (!projectId) return;
    setDpLoading(true);
    setDpError(null);
    try {
      const { job_id } = await exportApi.prepareCamtrapDP(projectId, includeThumbnails);
      // The progress modal subscribes to the job over WebSocket; when it
      // completes it calls downloadCamtrapDPZip.
      setDpJobId(job_id);
    } catch (err) {
      setDpError(errorMessage(err));
      setDpLoading(false);
    }
  };

  const finalizeCamtrapDPExport = async (jobId: string) => {
    if (!projectId) {
      setDpLoading(false);
      setDpJobId(null);
      return;
    }
    try {
      const blob = await exportApi.downloadCamtrapDPZip(projectId, jobId);
      downloadBlob(blob, "addaxai-camtrap-dp.zip");
    } catch (err) {
      setDpError(errorMessage(err));
    } finally {
      setDpLoading(false);
      setDpJobId(null);
    }
  };

  const abortCamtrapDPExport = (msg: string | null) => {
    if (msg) setDpError(msg);
    setDpLoading(false);
    setDpJobId(null);
  };

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
            <ExportRow
              title="Spreadsheet"
              description="Summary, counts, detections, files and deployments, including species labels and your corrections."
              filters={
                projectId ? (
                  <ExportScopeSelect
                    projectId={projectId}
                    onChange={setTableScope}
                    className="w-full min-w-0"
                  />
                ) : undefined
              }
              options={OBSERVATION_OPTIONS}
              isLoading={tableLoading}
              onSelect={handleDownloadSpreadsheet}
              error={tableError}
              disabled={nothingSelected}
            />

            <div className="my-6 border-t" />

            <ExportRow
              title="Spatial"
              description="Geographic point layers for GIS tools (QGIS, ArcGIS)."
              options={SPATIAL_OPTIONS}
              isLoading={spatialLoading}
              onSelect={handleSpatialSelect}
              error={spatialError}
            />

            <div className="my-6 border-t" />

            <ExportRow
              title="Camtrap DP"
              description="A standardised, community-developed exchange format for sharing and archiving camera trap data."
              options={CAMTRAP_MEDIA_OPTIONS}
              isLoading={dpLoading}
              onSelect={handleCamtrapSelect}
              error={dpError}
            />

            <div className="my-6 border-t" />

            <ExportRow
              title="Recognition file"
              description="The file Timelapse reads, in the MegaDetector output format, with your labels and corrections. Written as addaxai-recognitions.json into each deployment folder."
              filters={
                projectId ? (
                  <ExportScopeSelect
                    projectId={projectId}
                    onChange={setJsonScope}
                    className="w-full min-w-0"
                  />
                ) : undefined
              }
              options={[]}
              isLoading={false}
              onSelect={() => undefined}
              error={null}
              disabled={jsonNothingSelected}
              action={{ label: "Write JSON", onClick: () => setJsonDialogOpen(true) }}
            />

            <div className="my-6 border-t" />

            {/* No scope picker in the row: which deployments is one of
                the "which files" choices and sits with the others in the
                dialog, where it only shows when there is more than one. */}
            <ExportRow
              title="Image copies"
              description="Copies of your images in species folders, with your labels in the image metadata. Optional boxes and blur."
              options={[]}
              isLoading={false}
              onSelect={() => undefined}
              error={null}
              action={{
                label: "Choose options",
                icon: Images,
                onClick: () => setMediaDialogOpen(true),
              }}
            />
          </CardContent>
        </Card>
      </main>

      {projectId && (
        <>
          <SpatialExportConfirmDialog
            projectId={projectId}
            count={noSiteCount}
            formatLabel={
              SPATIAL_OPTIONS.find((o) => o.value === spatialFormat)?.label ??
              spatialFormat.toUpperCase()
            }
            open={spatialConfirmOpen}
            onOpenChange={setSpatialConfirmOpen}
            onProceed={() => void runSpatialExport(spatialFormat)}
          />
          <CamtrapDPExportConfirmDialog
            projectId={projectId}
            noSiteCount={noSiteCount}
            noDateCount={noDate?.count ?? 0}
            open={dpConfirmOpen}
            onOpenChange={setDpConfirmOpen}
            onProceed={() => void runCamtrapDPExport(dpIncludesThumbnails)}
          />

          <CamtrapDPProgressModal
            jobId={dpJobId}
            includesThumbnails={dpIncludesThumbnails}
            onComplete={finalizeCamtrapDPExport}
            onError={abortCamtrapDPExport}
          />

          <RecognitionJsonExportDialog
            projectId={projectId}
            scope={jsonScope}
            open={jsonDialogOpen}
            onOpenChange={setJsonDialogOpen}
          />

          <ImageCopiesDialog
            projectId={projectId}
            open={mediaDialogOpen}
            onOpenChange={setMediaDialogOpen}
          />
        </>
      )}
    </div>
  );
}
