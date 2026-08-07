/**
 * Export page (project mode) — one Card with a row per export.
 *
 * Each row reads left to right: title + caption (half width), an optional
 * filters slot (quarter width, currently the site/deployment scope on the
 * Spreadsheet row), and a single "Download" dropdown of formats (quarter
 * width). The dropdown pattern is used for every export, including
 * single-option ones, so the page reads as one tidy column of controls.
 * Layout follows AddaxAI Connect's ExportsPage; the filters column is
 * WebUI-specific and reserved for more filters later.
 */

import React, { useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { AlertCircle, ChevronDown, Download, Loader2 } from "lucide-react";

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
}: {
  options: DownloadOption[];
  onSelect: (value: string) => void;
  isLoading: boolean;
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button type="button" disabled={isLoading} className="gap-2">
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
}: {
  title: string;
  description: React.ReactNode;
  filters?: React.ReactNode;
  options: DownloadOption[];
  isLoading: boolean;
  onSelect: (value: string) => void;
  error: string | null;
}) {
  return (
    <div>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:gap-6">
        <div className="w-full sm:w-1/2 sm:shrink-0">
          <h3 className="text-sm font-medium">{title}</h3>
          <p className="mt-1 text-sm text-muted-foreground">{description}</p>
        </div>
        <div className="w-full sm:w-1/4">{filters}</div>
        <div className="w-full sm:w-1/4 sm:flex sm:justify-end">
          <DownloadDropdown
            options={options}
            onSelect={onSelect}
            isLoading={isLoading}
          />
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

  // One "Spreadsheet" download covering four tables. XLSX is a single
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
        const deployments = await exportApi.downloadDeployments(projectId, format, tableScope);
        downloadBlob(deployments, `addaxai-deployments.${format}`);
        const files = await exportApi.downloadFiles(projectId, format, tableScope);
        downloadBlob(files, `addaxai-files.${format}`);
        const detections = await exportApi.downloadDetections(projectId, format, tableScope);
        downloadBlob(detections, `addaxai-detections.${format}`);
        const counts = await exportApi.downloadObservations(projectId, format, tableScope);
        downloadBlob(counts, `addaxai-counts.${format}`);
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
      <header className="border-b bg-white/80 backdrop-blur-sm">
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
              description="Tables for deployments, files, detections, and counts."
              filters={
                projectId ? (
                  <ExportScopeSelect
                    projectId={projectId}
                    onChange={setTableScope}
                  />
                ) : undefined
              }
              options={OBSERVATION_OPTIONS}
              isLoading={tableLoading}
              onSelect={handleDownloadSpreadsheet}
              error={tableError}
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
        </>
      )}
    </div>
  );
}
