/**
 * Export page — three cards: Observations, Spatial, CamTrap DP.
 *
 * Ports the layout and formats from AddaxAI Connect's ExportsPage so users
 * can download project data in community-standard shapes: analyst-friendly
 * CSV/TSV/XLSX, GIS-ready GeoJSON/Shapefile/GeoPackage, and a GBIF-compatible
 * CamTrap DP package.
 */

import { useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { AlertCircle, Download, Loader2 } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { Button } from "../components/ui/button";
import { projectsApi } from "../api/projects";
import {
  exportApi,
  type ObservationFormat,
  type SpatialFormat,
} from "../api/export";
import { useNoSiteDeployments } from "../hooks/useNoSiteDeployments";
import { SpatialExportConfirmDialog } from "../components/export/SpatialExportConfirmDialog";
import { DiagnosticReportButton } from "../components/diagnostics/DiagnosticReportButton";
import { CamtrapDPExportConfirmDialog } from "../components/export/CamtrapDPExportConfirmDialog";
import { CamtrapDPProgressModal } from "../components/export/CamtrapDPProgressModal";
import { downloadBlob } from "../lib/download";

function slugify(name: string): string {
  return (
    name
      .toLowerCase()
      .trim()
      .replace(/[^\w\s-]/g, "")
      .replace(/[\s_]+/g, "-") || "project"
  );
}

const OBSERVATION_OPTIONS: { value: ObservationFormat; label: string }[] = [
  { value: "csv", label: "CSV" },
  { value: "tsv", label: "TSV" },
  { value: "xlsx", label: "XLSX" },
];

const SPATIAL_OPTIONS: { value: SpatialFormat; label: string }[] = [
  { value: "geojson", label: "GeoJSON" },
  { value: "shapefile", label: "Shapefile" },
  { value: "gpkg", label: "GeoPackage" },
];

type CamtrapMedia = "metadata" | "thumbnails";

const CAMTRAP_MEDIA_OPTIONS: { value: CamtrapMedia; label: string }[] = [
  { value: "metadata", label: "Metadata only" },
  { value: "thumbnails", label: "Include thumbnails" },
];

interface TextToggleProps<T extends string> {
  options: { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
}

function TextToggle<T extends string>({ options, value, onChange }: TextToggleProps<T>) {
  return (
    <div className="inline-flex rounded-md border border-input overflow-hidden">
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            className={
              "px-4 py-1.5 text-sm font-medium transition-colors " +
              (active
                ? "bg-primary text-primary-foreground"
                : "bg-background text-foreground hover:bg-accent")
            }
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

interface ErrorBannerProps {
  message: string;
}

function ErrorBanner({ message }: ErrorBannerProps) {
  return (
    <div className="flex items-center gap-2 p-3 bg-destructive/10 text-destructive rounded-md text-sm">
      <AlertCircle className="h-4 w-4 flex-shrink-0" />
      <span>{message}</span>
    </div>
  );
}

function todayIso(): string {
  return new Date().toISOString().split("T")[0];
}

function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return "Export failed";
}

export default function ExportPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const { data: project } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  });
  const { data: noSite } = useNoSiteDeployments(projectId);

  const [obsFormat, setObsFormat] = useState<ObservationFormat>("csv");
  const [spatialFormat, setSpatialFormat] = useState<SpatialFormat>("geojson");
  const [dpMedia, setDpMedia] = useState<CamtrapMedia>("metadata");

  const [obsLoading, setObsLoading] = useState(false);
  const [spatialLoading, setSpatialLoading] = useState(false);
  const [dpLoading, setDpLoading] = useState(false);

  const [obsError, setObsError] = useState<string | null>(null);
  const [spatialError, setSpatialError] = useState<string | null>(null);
  const [dpError, setDpError] = useState<string | null>(null);

  const [spatialConfirmOpen, setSpatialConfirmOpen] = useState(false);
  const [dpConfirmOpen, setDpConfirmOpen] = useState(false);
  const [dpJobId, setDpJobId] = useState<string | null>(null);
  const [dpJobIncludesThumbnails, setDpJobIncludesThumbnails] = useState(false);

  const projectSlug = slugify(project?.name ?? "project");
  const today = todayIso();
  const noSiteCount = noSite?.count ?? 0;

  const handleDownloadObservations = async () => {
    if (!projectId) return;
    setObsLoading(true);
    setObsError(null);
    try {
      const blob = await exportApi.downloadObservations(projectId, obsFormat);
      downloadBlob(blob, `observations-${projectSlug}-${today}.${obsFormat}`);
    } catch (err) {
      setObsError(errorMessage(err));
    } finally {
      setObsLoading(false);
    }
  };

  // Click handler for the Spatial download button. Opens the confirm
  // dialog when any deployment has no site (those would be silently
  // dropped from the export); otherwise goes straight to download.
  const onSpatialClick = () => {
    if (!projectId) return;
    if (noSiteCount > 0) {
      setSpatialConfirmOpen(true);
    } else {
      void runSpatialExport();
    }
  };

  const runSpatialExport = async () => {
    if (!projectId) return;
    setSpatialLoading(true);
    setSpatialError(null);
    const ext: Record<SpatialFormat, string> = {
      geojson: "geojson",
      shapefile: "zip",
      gpkg: "gpkg",
    };
    try {
      const blob = await exportApi.downloadSpatial(projectId, spatialFormat);
      downloadBlob(blob, `spatial-${projectSlug}-${today}.${ext[spatialFormat]}`);
    } catch (err) {
      setSpatialError(errorMessage(err));
    } finally {
      setSpatialLoading(false);
    }
  };

  // CamtrapDP: always open the pre-flight dialog first — the format has
  // hard schema requirements (one camera / one location / one period per
  // deployment) that the user should explicitly confirm are satisfied.
  const onCamtrapDPClick = () => {
    if (!projectId) return;
    setDpConfirmOpen(true);
  };

  const runCamtrapDPExport = async (includeThumbnails: boolean) => {
    if (!projectId) return;
    setDpLoading(true);
    setDpError(null);
    try {
      const { job_id } = await exportApi.prepareCamtrapDP(projectId, includeThumbnails);
      // The progress modal subscribes to the job over WebSocket via
      // useTaskProgress. When it completes, it calls downloadCamtrapDPZip.
      setDpJobId(job_id);
      setDpJobIncludesThumbnails(includeThumbnails);
    } catch (err) {
      setDpError(errorMessage(err));
      setDpLoading(false);
    }
  };

  // Called by the progress modal once the job reports complete.
  const finalizeCamtrapDPExport = async (jobId: string) => {
    if (!projectId) {
      setDpLoading(false);
      setDpJobId(null);
      return;
    }
    try {
      const blob = await exportApi.downloadCamtrapDPZip(projectId, jobId);
      downloadBlob(blob, `camtrap-dp-${projectSlug}-${today}.zip`);
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
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Export</h1>
              <p className="text-sm text-muted-foreground">
                Export project data in standardised formats
              </p>
            </div>
            <DiagnosticReportButton />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Observations</CardTitle>
            <CardDescription>
              Species observations spreadsheet, one row per species per image.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {obsError && <ErrorBanner message={obsError} />}
            <div className="flex items-center justify-between gap-4">
              <TextToggle
                options={OBSERVATION_OPTIONS}
                value={obsFormat}
                onChange={setObsFormat}
              />
              <Button
                onClick={handleDownloadObservations}
                disabled={obsLoading}
                className="flex items-center gap-2"
              >
                {obsLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Preparing export...
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4" />
                    Download {obsFormat.toUpperCase()}
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Spatial</CardTitle>
            <CardDescription>
              Geographic point layers for GIS tools (QGIS, ArcGIS).
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {spatialError && <ErrorBanner message={spatialError} />}
            <div className="flex items-center justify-between gap-4">
              <TextToggle
                options={SPATIAL_OPTIONS}
                value={spatialFormat}
                onChange={setSpatialFormat}
              />
              <Button
                onClick={onSpatialClick}
                disabled={spatialLoading}
                className="flex items-center gap-2"
              >
                {spatialLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Preparing export...
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4" />
                    Download{" "}
                    {SPATIAL_OPTIONS.find((o) => o.value === spatialFormat)?.label}
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Camtrap DP</CardTitle>
            <CardDescription>
              A standardized, community-developed data exchange format to
              and archive camera trap data.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {dpError && <ErrorBanner message={dpError} />}
            <div className="flex items-center justify-between gap-4">
              <TextToggle
                options={CAMTRAP_MEDIA_OPTIONS}
                value={dpMedia}
                onChange={setDpMedia}
              />
              <Button
                onClick={onCamtrapDPClick}
                disabled={dpLoading}
                className="flex items-center gap-2"
              >
                {dpLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Preparing export...
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4" />
                    Download CamTrap DP
                  </>
                )}
              </Button>
            </div>
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
            onProceed={() => void runSpatialExport()}
          />
          <CamtrapDPExportConfirmDialog
            projectId={projectId}
            noSiteCount={noSiteCount}
            open={dpConfirmOpen}
            onOpenChange={setDpConfirmOpen}
            onProceed={() =>
              void runCamtrapDPExport(dpMedia === "thumbnails")
            }
          />

          <CamtrapDPProgressModal
            jobId={dpJobId}
            includesThumbnails={dpJobIncludesThumbnails}
            onComplete={finalizeCamtrapDPExport}
            onError={abortCamtrapDPExport}
          />
        </>
      )}
    </div>
  );
}
