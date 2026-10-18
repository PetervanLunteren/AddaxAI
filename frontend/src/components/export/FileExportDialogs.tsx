/**
 * The three exports that write files into a picked folder: the tables,
 * the spatial layers and Camtrap DP. Each is a set of choices on the
 * shared ExportDialog; they share how a result reads and the warning
 * about deployments that have no site.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { deploymentsApi } from "../../api/deployments";
import {
  exportApi,
  type ExportScope,
  type ObservationFormat,
  type SpatialFormat,
  type WrittenFiles,
} from "../../api/export";
import { projectsApi } from "../../api/projects";
import { NO_SITE_SENTINEL } from "../../lib/filter-url";
import { Callout } from "../ui/callout";
import { CaptionedRadioGroup } from "../ui/captioned-radio-group";
import { ExportDialog, ExportSection, type ExportResultView } from "./ExportDialog";

export interface ExportDialogComponentProps {
  projectId: string;
  onClose: () => void;
}

function describeWritten(data: unknown): ExportResultView {
  const { output_dir, files } = data as WrittenFiles;
  return {
    title: files.length === 1 ? "File written" : `${files.length} files written`,
    folder: output_dir,
    details: (
      <ul className="space-y-1 font-mono text-xs text-muted-foreground">
        {files.map((name) => (
          <li key={name}>{name}</li>
        ))}
      </ul>
    ),
  };
}

const target = (outputDir: string, scope: ExportScope | undefined) => ({
  output_dir: outputDir,
  deployment_ids: scope?.deploymentIds,
});

/** How many of the selected deployments have no site, so no coordinates. */
function useWithoutSite(projectId: string, scope: ExportScope | undefined): number {
  const { data: deployments } = useQuery({
    queryKey: ["deployments", projectId],
    queryFn: () => deploymentsApi.list({ projectId }),
  });
  const selected = scope?.deploymentIds ? new Set(scope.deploymentIds) : null;
  return (deployments ?? []).filter(
    (d) => d.site_id === null && (!selected || selected.has(d.id)),
  ).length;
}

function WithoutSiteWarning({
  projectId,
  count,
  format,
}: {
  projectId: string;
  count: number;
  format: string;
}) {
  if (count === 0) return null;
  return (
    <Callout variant="warning" size="compact">
      {count === 1 ? "1 selected deployment has" : `${count} selected deployments have`} no
      camera site, so no coordinates, and {count === 1 ? "is" : "are"} left out of the{" "}
      {format}. Add sites on the{" "}
      <Link
        to={`/projects/${projectId}/deployments?site_ids=${NO_SITE_SENTINEL}`}
        className="underline"
      >
        Deployments page
      </Link>
      .
    </Callout>
  );
}

const TABLE_FORMATS = [
  {
    value: "xlsx" as const,
    label: "Excel workbook (XLSX)",
    caption: "One file with five sheets: Summary, Counts, Detections, Files and Deployments.",
  },
  {
    value: "csv" as const,
    label: "CSV",
    caption: "The same five tables as five files, for R, Python or any spreadsheet.",
  },
  {
    value: "tsv" as const,
    label: "TSV",
    caption: "As CSV, separated by tabs.",
  },
];

export function SpreadsheetExportDialog({ projectId, onClose }: ExportDialogComponentProps) {
  const [scope, setScope] = useState<ExportScope | undefined>();
  const [format, setFormat] = useState<ObservationFormat>("xlsx");
  return (
    <ExportDialog
      projectId={projectId}
      title="Spreadsheet"
      description="Summary, counts, detections, files and deployments, including species labels and your corrections."
      onClose={onClose}
      scope={scope}
      onScopeChange={setScope}
      start={async (outputDir) => ({
        result: await exportApi.writeTables(projectId, { ...target(outputDir, scope), format }),
      })}
      describeResult={describeWritten}
    >
      <ExportSection title="Format">
        <CaptionedRadioGroup
          name="tables-format"
          value={format}
          onChange={setFormat}
          options={TABLE_FORMATS}
        />
      </ExportSection>
    </ExportDialog>
  );
}

const SPATIAL_FORMATS = [
  {
    value: "gpkg" as const,
    label: "GeoPackage",
    caption: "One file with both layers. Opens directly in QGIS and ArcGIS.",
  },
  {
    value: "geojson" as const,
    label: "GeoJSON",
    caption: "Plain text, for web maps and scripts.",
  },
  {
    value: "shapefile" as const,
    label: "Shapefile",
    caption: "A ZIP with one shapefile per layer, for older GIS tools.",
  },
];

export function SpatialExportDialog({ projectId, onClose }: ExportDialogComponentProps) {
  const [scope, setScope] = useState<ExportScope | undefined>();
  const [format, setFormat] = useState<SpatialFormat>("gpkg");
  const withoutSite = useWithoutSite(projectId, scope);
  return (
    <ExportDialog
      projectId={projectId}
      title="Spatial"
      description="Two point layers for GIS tools: one point per camera, and one per camera per species."
      onClose={onClose}
      scope={scope}
      onScopeChange={setScope}
      start={async (outputDir) => ({
        result: await exportApi.writeSpatial(projectId, { ...target(outputDir, scope), format }),
      })}
      describeResult={describeWritten}
    >
      <ExportSection title="Format">
        <CaptionedRadioGroup
          name="spatial-format"
          value={format}
          onChange={setFormat}
          options={SPATIAL_FORMATS}
        />
      </ExportSection>
      <WithoutSiteWarning projectId={projectId} count={withoutSite} format="spatial layers" />
    </ExportDialog>
  );
}

const CAMTRAP_CONTENTS = [
  {
    value: "metadata" as const,
    label: "Metadata only",
    caption: "Deployments, media and observations as the standard's three tables. Small.",
  },
  {
    value: "thumbnails" as const,
    label: "With thumbnails",
    caption: "Adds a small image of every file, so the package stands on its own. Larger and slower.",
  },
];

export function CamtrapDPExportDialog({ projectId, onClose }: ExportDialogComponentProps) {
  const [scope, setScope] = useState<ExportScope | undefined>();
  const [contents, setContents] = useState<"metadata" | "thumbnails">("metadata");
  const withoutSite = useWithoutSite(projectId, scope);
  const { data: noDate } = useQuery({
    queryKey: ["files-without-date", projectId],
    queryFn: () => projectsApi.getFilesWithoutDate(projectId),
  });
  return (
    <ExportDialog
      projectId={projectId}
      title="Camtrap DP"
      description="A standardised, community-developed exchange format for sharing and archiving camera trap data."
      onClose={onClose}
      scope={scope}
      onScopeChange={setScope}
      start={async (outputDir) => {
        const { job_id } = await exportApi.prepareCamtrapDP(projectId, {
          ...target(outputDir, scope),
          include_thumbnails: contents === "thumbnails",
        });
        return { jobId: job_id };
      }}
      describeResult={describeWritten}
    >
      <ExportSection title="Contents">
        <CaptionedRadioGroup
          name="camtrap-contents"
          value={contents}
          onChange={setContents}
          options={CAMTRAP_CONTENTS}
        />
      </ExportSection>
      <Callout variant="info" size="compact">
        Camtrap DP expects every deployment to be one camera at one place for one period,
        the photos of a single SD card. Check your{" "}
        <Link to={`/projects/${projectId}/deployments`} className="underline">
          deployments
        </Link>{" "}
        before you share the package.
      </Callout>
      <WithoutSiteWarning projectId={projectId} count={withoutSite} format="package" />
      {(noDate?.count ?? 0) > 0 && (
        <Callout variant="warning" size="compact">
          Files without a capture date are left out of the package.
        </Callout>
      )}
    </ExportDialog>
  );
}
