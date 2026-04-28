/**
 * FilesTab - file-level verification workflow.
 *
 * Shows one tile per media item (images + videos) for a project, with the
 * same filter surface as the Events tab (sites, dates, labels, verification
 * state), paginated at 48/page. Clicking a tile opens FileDetailModal.
 * Filter state is owned by the parent (VerifyPage) and shared with Events;
 * FilesTab owns only pagination and modal selection.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { CircleHelp, Layers, Loader2 } from "lucide-react";
import { eventsApi } from "../../api/events";
import { filesApi } from "../../api/files";
import { projectsApi } from "../../api/projects";
import { sitesApi } from "../../api/sites";
import { Button } from "../ui/button";
import { Card, CardContent } from "../ui/card";
import {
  setSpeciesContext,
} from "../../utils/species-colors";
import type { EventFilterParams } from "../../api/types";
import { FileCard } from "./FileCard";
import { FileDetailModal } from "./FileDetailModal";
import { FilterChips } from "./FilterChips";
import { FilterPanel } from "./FilterPanel";
import { HelpSheet } from "./HelpSheet";

const PAGE_SIZE = 48;
const FILTER_DEBOUNCE_MS = 300;

function useDebouncedValue<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  const serialized = JSON.stringify(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(JSON.parse(serialized)), delay);
    return () => clearTimeout(timer);
  }, [serialized, delay]);
  return debounced;
}

interface FilesTabProps {
  projectId: string;
  filters: EventFilterParams;
  onFiltersChange: (next: EventFilterParams) => void;
  classificationModelId: string | null;
}

export function FilesTab({
  projectId,
  filters,
  onFiltersChange,
  classificationModelId,
}: FilesTabProps) {
  const [page, setPage] = useState(0);
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);

  const debouncedFilters = useDebouncedValue(filters, FILTER_DEBOUNCE_MS);

  // Reset to page 0 whenever filters change.
  const prevFiltersRef = useRef(filters);
  useEffect(() => {
    if (
      JSON.stringify(prevFiltersRef.current) !== JSON.stringify(filters)
    ) {
      setPage(0);
      prevFiltersRef.current = filters;
    }
  }, [filters]);

  // Project for detection threshold.
  const { data: project } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => projectsApi.get(projectId),
    enabled: !!projectId,
  });
  const detectionThreshold = project?.detection_threshold ?? 0;

  // Total file count (unfiltered baseline).
  const { data: totalCountData } = useQuery({
    queryKey: ["files-count-for-verify", projectId, null],
    queryFn: () => filesApi.countForVerify(projectId),
    enabled: !!projectId,
  });

  // Filtered count.
  const { data: filteredCountData } = useQuery({
    queryKey: ["files-count-for-verify", projectId, debouncedFilters],
    queryFn: () => filesApi.countForVerify(projectId, debouncedFilters),
    enabled: !!projectId,
  });

  // Aggregate verified/total for the filtered set.
  const { data: verificationStats } = useQuery({
    queryKey: [
      "files-verification-stats",
      projectId,
      debouncedFilters,
    ],
    queryFn: () => filesApi.verificationStats(projectId, debouncedFilters),
    enabled: !!projectId,
  });

  // Page of files.
  const {
    data: files,
    isLoading,
    isFetching,
    isPlaceholderData,
  } = useQuery({
    queryKey: ["files-for-verify", projectId, page, debouncedFilters],
    queryFn: () =>
      filesApi.listForVerify({
        project_id: projectId,
        skip: page * PAGE_SIZE,
        limit: PAGE_SIZE,
        filters: debouncedFilters,
      }),
    enabled: !!projectId,
    placeholderData: (prev) => prev,
  });

  // Filter options for label display name mapping in FilterChips.
  const { data: filterOptions } = useQuery({
    queryKey: ["event-filter-options", projectId],
    queryFn: () => eventsApi.getFilterOptions(projectId),
    enabled: !!projectId,
  });

  // Site name lookup for chips.
  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
    enabled: !!projectId && (filters.site_ids?.length ?? 0) > 0,
  });
  const siteNames = useMemo(() => {
    const map: Record<string, string> = {};
    for (const s of sites ?? []) map[s.id] = s.name;
    return map;
  }, [sites]);

  // Register species colors for labels in this page so FileCard chips match
  // the rest of the verify UI.
  useMemo(() => {
    if (files?.length) {
      const allLabels = [...new Set(files.flatMap((f) => f.labels))];
      const aliases: Record<string, string> = {};
      for (const f of files) {
        for (const [uuid, name] of Object.entries(f.display_labels)) {
          aliases[uuid] = name;
        }
      }
      if (allLabels.length > 0) setSpeciesContext(allLabels, aliases);
    }
  }, [files]);

  // Navigation from FileDetailModal dispatches "navigate-file" with the
  // target file id; mirror the pattern Event modal uses for adjacency.
  useEffect(() => {
    const handler = (e: Event) => {
      const targetId = (e as CustomEvent).detail as string | null;
      if (targetId) setSelectedFileId(targetId);
    };
    window.addEventListener("navigate-file", handler);
    return () => window.removeEventListener("navigate-file", handler);
  }, []);

  const totalFiles = totalCountData?.count ?? 0;
  const filteredFiles = filteredCountData?.count ?? totalFiles;
  const isFiltered =
    (filters.site_ids?.length ?? 0) > 0 ||
    !!filters.date_from ||
    !!filters.date_to ||
    (filters.labels?.length ?? 0) > 0 ||
    (!!filters.verification && filters.verification !== "all");
  const hasMore = files && files.length === PAGE_SIZE;

  const pct =
    verificationStats && verificationStats.total_files > 0
      ? (verificationStats.verified_files / verificationStats.total_files) * 100
      : 0;

  const handleClearFilters = useCallback(() => {
    onFiltersChange({});
  }, [onFiltersChange]);

  return (
    <>
      <FilterPanel
        filters={filters}
        onChange={onFiltersChange}
        projectId={projectId}
        isOpen={true}
        onToggle={() => {}}
        classificationModelId={classificationModelId}
        detectionFloor={detectionThreshold}
        countBy="file"
      >
        {isFiltered && (
          <FilterChips
            filters={filters}
            onChange={onFiltersChange}
            filteredCount={filteredFiles}
            totalCount={totalFiles}
            siteNames={siteNames}
            displayLabels={filterOptions?.display_labels}
            detectionFloor={detectionThreshold}
          />
        )}
      </FilterPanel>

      {totalFiles > 0 && (
        <div className="flex flex-wrap items-center gap-3 min-h-12 py-2 px-3 bg-white rounded-lg border shadow-sm">
          <button
            onClick={() => setHelpOpen(true)}
            className="text-muted-foreground hover:text-foreground transition-colors"
            title="Help"
          >
            <CircleHelp className="h-4 w-4" />
          </button>
          <div className="flex items-center gap-3 ml-auto">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <div className="relative h-2 w-20 overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full transition-all duration-500 ease-out rounded-full"
                  style={{ width: `${pct}%`, backgroundColor: "#0f6064" }}
                />
              </div>
              {Math.round(pct)}% files verified
            </div>
          </div>
        </div>
      )}

      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : !files || files.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16 text-center">
            <Layers className="h-12 w-12 text-muted-foreground/50 mb-4" />
            <p className="text-lg font-medium text-muted-foreground">
              {isFiltered ? "No files match your filters" : "No files yet"}
            </p>
            <p className="text-sm text-muted-foreground mt-1 max-w-md">
              {isFiltered
                ? "Try adjusting or clearing your filters to see more files."
                : "Files appear here once you run a deployment analysis."}
            </p>
            {isFiltered && (
              <Button
                variant="outline"
                size="sm"
                className="mt-4"
                onClick={handleClearFilters}
              >
                Clear all filters
              </Button>
            )}
          </CardContent>
        </Card>
      ) : (
        <>
          <div
            className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 transition-opacity ${
              isPlaceholderData ? "opacity-60" : ""
            }`}
          >
            {files.map((file) => (
              <FileCard
                key={file.id}
                file={file}
                detectionThreshold={detectionThreshold}
                onClick={() => setSelectedFileId(file.id)}
              />
            ))}
          </div>

          <div className="flex items-center justify-center gap-4">
            <Button
              variant="outline"
              size="sm"
              disabled={page === 0}
              onClick={() => setPage((p) => p - 1)}
            >
              Previous
            </Button>
            <span className="text-sm text-muted-foreground">
              {isFiltered
                ? `${filteredFiles} of ${totalFiles} files`
                : `${totalFiles} files`}
              {" · "}Page {page + 1}
              {isFetching && " (loading...)"}
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={!hasMore}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </Button>
          </div>
        </>
      )}

      <FileDetailModal
        fileId={selectedFileId}
        projectId={projectId}
        isOpen={!!selectedFileId}
        onClose={() => setSelectedFileId(null)}
        filters={debouncedFilters}
      />

      <HelpSheet open={helpOpen} onOpenChange={setHelpOpen} />
    </>
  );
}
