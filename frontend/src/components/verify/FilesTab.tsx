/**
 * FilesTab - per-file workflow (the "Media" grouping of the Edit view).
 *
 * Shows one tile per still photo and per extracted video frame for a
 * project, with the same filter surface as the Events grouping (sites,
 * dates, labels, verification state), paginated at 48/page. Clicking a
 * tile opens FileDetailModal. Filter state is owned by the parent
 * (VerifyView) and shared with Events; FilesTab owns only pagination
 * and modal selection.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { CircleHelp, Layers, Loader2 } from "lucide-react";
import { eventsApi } from "../../api/events";
import { filesApi } from "../../api/files";
import { projectsApi } from "../../api/projects";
import { speciesLabelMap } from "../../lib/species-name-mode";
import { Button } from "../ui/button";
import { Card, CardContent } from "../ui/card";
import {
  setSpeciesContext,
} from "../../utils/species-colors";
import type {
  EventFilterParams,
  VerifyViewMode,
} from "../../api/types";
import { FileCard } from "./FileCard";
import { FileDetailModal } from "./FileDetailModal";
import { hasAnyActiveFilter } from "./FilterChips";
import { VerifyFilterBar } from "./VerifyFilterBar";
import { VerifyHelpSheet } from "./VerifyHelpSheet";
import { MediaWelcomePopover } from "./MediaWelcomePopover";
import { SortSelector } from "./SortSelector";
import {
  VerifyProgressPill,
  VerifyToolbar,
  VerifyToolbarIcon,
} from "./VerifyToolbar";

const FILES_SORT_MODES = ["newest", "oldest", "random", "cls_low"] as const;

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
  view: VerifyViewMode;
  onViewChange: (view: VerifyViewMode) => void;
}

export function FilesTab({
  projectId,
  filters,
  onFiltersChange,
  classificationModelId,
  view,
  onViewChange,
}: FilesTabProps) {
  const [page, setPage] = useState(0);
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [showWelcome, setShowWelcome] = useState(
    () => !localStorage.getItem("addaxai:mediaWelcomeDismissed")
  );
  const handleDismissWelcome = useCallback(() => {
    setShowWelcome(false);
    localStorage.setItem("addaxai:mediaWelcomeDismissed", "1");
  }, []);

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

  // Verified/total for the progress pill. UNFILTERED on purpose: the
  // bar measures progress against the whole project, not the current
  // filter view (a narrowed view must not read as fully verified).
  // Sourced from the events stats endpoint because every view (Events,
  // Media, Observations) shows the same metric "percent observations
  // verified".
  const { data: verificationStats } = useQuery({
    queryKey: ["events", "verification-stats", projectId],
    queryFn: () => eventsApi.verificationStats(projectId),
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

  // Register species colors for labels in this page so FileCard chips match
  // the rest of the verify UI.
  useMemo(() => {
    if (files?.length) {
      const allLabels = [...new Set(files.flatMap((f) => f.labels))];
      const aliases: Record<string, string> = {};
      for (const f of files) {
        for (const [uuid, name] of Object.entries(speciesLabelMap(f))) {
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
  const isFiltered = hasAnyActiveFilter(filters);
  const hasMore = files && files.length === PAGE_SIZE;

  const pct =
    verificationStats && verificationStats.total_detections > 0
      ? (verificationStats.verified_detections / verificationStats.total_detections) * 100
      : 0;

  const handleClearFilters = useCallback(() => {
    onFiltersChange({});
  }, [onFiltersChange]);

  return (
    <>
      <VerifyFilterBar
        filters={filters}
        onChange={onFiltersChange}
        projectId={projectId}
        classificationModelId={classificationModelId}
        detectionFloor={detectionThreshold}
        countBy="file"
        view={view}
        onViewChange={onViewChange}
      />

      {totalFiles > 0 && (
        <VerifyToolbar>
          <VerifyToolbarIcon
            icon={CircleHelp}
            title="Help"
            onClick={() => setHelpOpen(true)}
          />
          <SortSelector
            sort={filters.sort ?? "newest"}
            seed={filters.seed ?? null}
            availableSorts={FILES_SORT_MODES}
            onChange={(next, seed) => {
              const updated = { ...filters };
              if (next === "newest") delete updated.sort;
              else updated.sort = next;
              if (seed === null) delete updated.seed;
              else updated.seed = seed;
              onFiltersChange(updated);
            }}
          />
          <VerifyProgressPill pct={pct} label="verified" />
        </VerifyToolbar>
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
              {totalFiles === 0
                ? "No files yet"
                : "No files match your filters"}
            </p>
            <p className="text-sm text-muted-foreground mt-1 max-w-md">
              {totalFiles === 0
                ? "Files appear here once you run a deployment analysis."
                : "Try adjusting or clearing your filters to see more files."}
            </p>
            {totalFiles > 0 && (
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
              Showing {page * PAGE_SIZE + 1}-{page * PAGE_SIZE + files.length}
              {" of "}
              {isFiltered ? filteredFiles : totalFiles} files
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

      <VerifyHelpSheet open={helpOpen} onOpenChange={setHelpOpen} />

      <MediaWelcomePopover
        open={showWelcome}
        onDismiss={handleDismissWelcome}
      />
    </>
  );
}
