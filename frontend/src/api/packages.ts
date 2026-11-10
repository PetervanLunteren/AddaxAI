/**
 * Project packages: import side. The export side sits in `export.ts`
 * with the other exports (`preparePackage`, `packageSize`).
 *
 * A package is a folder another AddaxAI wrote. `preview` says what
 * importing it would do; `importPackage` starts the job, whose
 * completion payload is a PackageImportResult.
 */

import { api } from "../lib/api-client";

export interface PackageSizeResponse {
  deployments: number;
  files: number;
  media_bytes: number;
  /** An estimate, from the database size. */
  data_bytes: number;
}

export interface PackageExportResult {
  package_dir: string;
  deployments: number;
  files: number;
  includes_media: boolean;
}

export interface PackageDeploymentPreview {
  id: string;
  files: number;
  exists: boolean;
}

export interface PackagePreview {
  project_id: string;
  project_name: string;
  project_exists: boolean;
  app_version: string;
  includes_media: boolean;
  deployments: PackageDeploymentPreview[];
  problems: string[];
  warnings: string[];
}

export interface PackageImportResult {
  project_id: string;
  project_created: boolean;
  deployments_created: number;
  deployments_replaced: number;
  deployments_skipped: number;
}

export const packagesApi = {
  preview: (path: string): Promise<PackagePreview> =>
    api.post<PackagePreview>("/api/packages/preview", { path }),

  importPackage: (
    path: string,
    replaceExisting: boolean,
  ): Promise<{ job_id: string }> =>
    api.post<{ job_id: string }>("/api/packages/import", {
      path,
      replace_existing: replaceExisting,
    }),
};
