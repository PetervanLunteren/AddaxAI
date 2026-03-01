/**
 * SimilarityTab - orchestrates the embedding-driven similarity view.
 *
 * Manages sort/search mode via an explicit segmented control,
 * selection model, and coordinates toolbar, grid, bulk actions,
 * and detail sheet.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Loader2, Layers, RefreshCw, Search, X } from "lucide-react";
import { toast } from "sonner";
import { similarityApi } from "../../api/similarity";
import { detectionsApi } from "../../api/detections";
import { projectsApi } from "../../api/projects";
import { Alert, AlertDescription } from "../ui/alert";
import { Button } from "../ui/button";
import { Card, CardContent } from "../ui/card";
import { Slider } from "../ui/slider";
import { API_BASE_URL } from "../../lib/api-client";
import { cn } from "../../lib/utils";
import { CropGrid } from "./CropGrid";
import { BulkActionBar } from "./BulkActionBar";
import { DetectionDetailSheet } from "./DetectionDetailSheet";
import { ReEmbedModal } from "../projects/ReEmbedModal";
import { useLabelOptions } from "../../hooks/useLabelOptions";
import type {
  SortResponse,
  SearchResponse,
  DetectionSummary,
  SimilarityFilters,
  EventFilterParams,
} from "../../api/types";

interface SimilarityTabProps {
  projectId: string;
  filters: EventFilterParams;
  classificationModelId: string | null;
}

/** Convert event filters to similarity filters. */
function toSimilarityFilters(f: EventFilterParams): SimilarityFilters {
  return {
    species: f.species,
    site_ids: f.site_ids,
    date_from: f.date_from,
    date_to: f.date_to,
    min_confidence: f.min_confidence,
  };
}

export function SimilarityTab({
  projectId,
  filters,
  classificationModelId,
}: SimilarityTabProps) {
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();

  // Re-embed state
  const [reEmbedJobId, setReEmbedJobId] = useState<string | null>(null);

  const urlAnchor = searchParams.get("anchor");
  const [anchorId, setAnchorId] = useState<string | null>(urlAnchor);
  const [threshold, setThreshold] = useState(0.7);

  // Explicit view mode — defaults to "search" when URL has anchor
  const [viewMode, setViewMode] = useState<"sort" | "search">(
    urlAnchor ? "search" : "sort"
  );

  // Results
  const [sortResult, setSortResult] = useState<SortResponse | null>(null);
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);

  // Selection
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Mislabel filter
  const [showMislabelsOnly, setShowMislabelsOnly] = useState(false);

  // Detail sheet
  const [detailDetection, setDetailDetection] = useState<DetectionSummary | null>(null);

  // Label options for relabel
  const { options: labelOptions, isLoading: labelOptionsLoading } =
    useLabelOptions(classificationModelId, projectId);

  // Stats query
  const { data: stats } = useQuery({
    queryKey: ["similarity-stats", projectId],
    queryFn: () => similarityApi.stats(projectId),
    enabled: !!projectId,
  });

  // Auto-trigger search when anchor is set from URL on mount
  useEffect(() => {
    if (urlAnchor) {
      setAnchorId(urlAnchor);
    }
  }, [urlAnchor]);

  // Sort mutation
  const sortMutation = useMutation({
    mutationFn: () =>
      similarityApi.sort(projectId, {
        filters: toSimilarityFilters(filters),
      }),
    onSuccess: (data) => {
      setSortResult(data);
      setSelectedIds(new Set());
      setShowMislabelsOnly(false);
    },
    onError: (err: Error) => toast.error(err.message),
  });

  // Stable key for filter comparison — avoids redundant sorts when toggling tabs
  const filtersKey = JSON.stringify(toSimilarityFilters(filters));
  const lastSortFiltersRef = useRef<string | null>(null);

  // Auto-sort on mount and when filters change
  useEffect(() => {
    if (viewMode === "sort" && stats?.embedded_detections && filtersKey !== lastSortFiltersRef.current) {
      lastSortFiltersRef.current = filtersKey;
      sortMutation.mutate();
    }
  }, [viewMode, filtersKey, stats?.embedded_detections]); // eslint-disable-line react-hooks/exhaustive-deps

  // Search mutation
  const searchMutation = useMutation({
    mutationFn: (anchor: string) =>
      similarityApi.search(projectId, {
        anchor_detection_id: anchor,
        filters: toSimilarityFilters(filters),
        limit: 100,
        threshold,
      }),
    onSuccess: (data) => {
      setSearchResult(data);
      setSelectedIds(new Set());
    },
    onError: (err: Error) => toast.error(err.message),
  });

  // Trigger search when anchor or threshold changes
  useEffect(() => {
    if (anchorId) {
      searchMutation.mutate(anchorId);
    }
  }, [anchorId, threshold, filtersKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // Flat detection list for selection model
  const allDetections = useMemo((): DetectionSummary[] => {
    if (viewMode === "sort" && sortResult) {
      const dets = sortResult.detections;
      if (showMislabelsOnly) {
        return dets.filter(
          (d) => d.neighbor_agreement != null && d.neighbor_agreement < 0.5
        );
      }
      return dets;
    }
    if (viewMode === "search" && searchResult) {
      return searchResult.results;
    }
    return [];
  }, [viewMode, sortResult, searchResult, showMislabelsOnly]);

  const handleSelect = useCallback(
    (detectionId: string, e: React.MouseEvent) => {
      setSelectedIds((prev) => {
        const next = new Set(prev);

        if (e.shiftKey && prev.size > 0) {
          // Range select
          const lastSelected = Array.from(prev).pop()!;
          const allIds = allDetections.map((d) => d.detection_id);
          const startIdx = allIds.indexOf(lastSelected);
          const endIdx = allIds.indexOf(detectionId);
          if (startIdx >= 0 && endIdx >= 0) {
            const [lo, hi] =
              startIdx < endIdx
                ? [startIdx, endIdx]
                : [endIdx, startIdx];
            for (let i = lo; i <= hi; i++) {
              next.add(allIds[i]);
            }
          }
        } else if (e.ctrlKey || e.metaKey) {
          // Toggle
          if (next.has(detectionId)) {
            next.delete(detectionId);
          } else {
            next.add(detectionId);
          }
        } else {
          // Single select
          next.clear();
          next.add(detectionId);
        }

        return next;
      });
    },
    [allDetections]
  );

  const handleCardClick = useCallback((detection: DetectionSummary) => {
    setDetailDetection(detection);
  }, []);

  const handleFindSimilar = useCallback(
    (detectionId: string) => {
      setAnchorId(detectionId);
      setViewMode("search");
      setSearchParams(
        (prev) => {
          prev.set("anchor", detectionId);
          return prev;
        },
        { replace: true }
      );
    },
    [setSearchParams]
  );

  const handleCloseSearch = useCallback(() => {
    setAnchorId(null);
    setSearchResult(null);
    setSelectedIds(new Set());
    setViewMode("sort");
    setSearchParams(
      (prev) => {
        prev.delete("anchor");
        return prev;
      },
      { replace: true }
    );
  }, [setSearchParams]);

  const handleActionComplete = useCallback(() => {
    // Re-run the current query to refresh data
    if (viewMode === "sort") {
      sortMutation.mutate();
    } else if (anchorId) {
      searchMutation.mutate(anchorId);
    }
  }, [viewMode, anchorId]); // eslint-disable-line react-hooks/exhaustive-deps

  /** Patch detections in local state without refetching. */
  const patchLocalDetections = useCallback(
    (patchFn: (d: DetectionSummary) => DetectionSummary) => {
      if (viewMode === "sort" && sortResult) {
        setSortResult({
          ...sortResult,
          detections: sortResult.detections.map(patchFn),
        });
      } else if (viewMode === "search" && searchResult) {
        setSearchResult({
          ...searchResult,
          results: searchResult.results.map(patchFn),
        });
      }
    },
    [viewMode, sortResult, searchResult]
  );

  const handleRelabel = useCallback(
    (detectionId: string, species: string, category: string) => {
      detectionsApi
        .bulkRelabel([detectionId], species, category)
        .then((data) => {
          toast.success(`Relabelled ${data.updated_count} detection to ${species}`);
          patchLocalDetections((d) =>
            d.detection_id === detectionId ? { ...d, species, category, verified: true } : d
          );
        })
        .catch((err: Error) => toast.error(err.message));
    },
    [patchLocalDetections]
  );

  const handleBulkRelabel = useCallback(
    (ids: string[], species: string | null, category: string) => {
      const idSet = new Set(ids);
      patchLocalDetections((d) =>
        idSet.has(d.detection_id) ? { ...d, species, category, verified: true } : d
      );
    },
    [patchLocalDetections]
  );

  const handleBulkVerify = useCallback(
    (ids: string[]) => {
      const idSet = new Set(ids);
      patchLocalDetections((d) =>
        idSet.has(d.detection_id) ? { ...d, verified: true } : d
      );
    },
    [patchLocalDetections]
  );

  // Keyboard shortcuts
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      if (e.key === "Escape") {
        if (detailDetection) {
          setDetailDetection(null);
        } else if (selectedIds.size > 0) {
          setSelectedIds(new Set());
        } else if (viewMode === "search") {
          handleCloseSearch();
        }
        return;
      }

      if (e.key === "Enter" && selectedIds.size > 0) {
        e.preventDefault();
        // Verify selected
        const ids = Array.from(selectedIds);
        import("../../api/detections").then(({ detectionsApi }) => {
          detectionsApi
            .bulkVerify(ids, true)
            .then((data) => {
              toast.success(`Verified ${data.updated_count} detections`);
              handleBulkVerify(ids);
              setSelectedIds(new Set());
            });
        });
        return;
      }

      if (e.key === "a" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        // Select all visible
        const allIds = new Set(allDetections.map((d) => d.detection_id));
        setSelectedIds(allIds);
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedIds, detailDetection, allDetections, handleActionComplete, viewMode, handleCloseSearch]);

  // No embeddings state
  if (stats && stats.embedded_detections === 0) {
    return (
      <Card>
        <CardContent className="flex flex-col items-center justify-center py-16 text-center">
          <Layers className="h-12 w-12 text-muted-foreground/50 mb-4" />
          <p className="text-lg font-medium text-muted-foreground">
            No embeddings yet
          </p>
          <p className="text-sm text-muted-foreground mt-1 max-w-md">
            Run an analysis with an embedding model selected to use similarity
            features. Embeddings are computed from detection crops using DINOv2.
          </p>
        </CardContent>
      </Card>
    );
  }

  const isLoading = sortMutation.isPending || searchMutation.isPending;
  const hasResults =
    (viewMode === "sort" && sortResult !== null) ||
    (viewMode === "search" && searchResult !== null);

  const handleEmbedNow = async () => {
    try {
      const { job_id } = await projectsApi.reEmbed(projectId);
      setReEmbedJobId(job_id);
    } catch (err: unknown) {
      toast.error(err instanceof Error ? err.message : "Failed to start embedding");
    }
  };

  return (
    <div className="space-y-2">
      {/* Warning when embeddings are incomplete */}
      {stats && stats.missing_embeddings > 0 && (
        <Alert className="border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
          <AlertTriangle className="h-4 w-4 text-amber-600 dark:text-amber-400" />
          <AlertDescription className="flex items-center gap-2">
            <span>
              {stats.missing_embeddings} detection{stats.missing_embeddings !== 1 ? "s are" : " is"} missing embeddings
              — similarity results may be incomplete.
            </span>
            <Button variant="outline" size="sm" className="ml-auto shrink-0" onClick={handleEmbedNow}>
              Embed now
            </Button>
          </AlertDescription>
        </Alert>
      )}

      <ReEmbedModal
        open={!!reEmbedJobId}
        onOpenChange={(open) => { if (!open) setReEmbedJobId(null); }}
        jobId={reEmbedJobId}
        onComplete={() => {
          queryClient.invalidateQueries({ queryKey: ["similarity-stats", projectId] });
        }}
      />

      {/* Unified toolbar with segmented control */}
      <div className="flex flex-wrap items-center gap-3 py-2">
        {/* Segmented control */}
        <div className="flex rounded-lg bg-muted p-0.5">
          <button
            className={cn(
              "px-3 py-1 text-xs font-medium rounded-md transition-colors",
              viewMode === "sort"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
            onClick={() => {
              setViewMode("sort");
              setSelectedIds(new Set());
            }}
          >
            Sort
          </button>
          <button
            className={cn(
              "px-3 py-1 text-xs font-medium rounded-md transition-colors",
              viewMode === "search"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
            onClick={() => {
              setViewMode("search");
              setSelectedIds(new Set());
            }}
          >
            Search
          </button>
        </div>

        <div className="h-4 w-px bg-border" />

        {/* Sort controls */}
        {viewMode === "sort" && (
          <>
            <button
              onClick={() => sortMutation.mutate()}
              disabled={sortMutation.isPending || !stats?.embedded_detections}
              className="text-muted-foreground hover:text-foreground disabled:opacity-50 transition-colors"
              title="Re-sort"
            >
              <RefreshCw className={cn("h-4 w-4", sortMutation.isPending && "animate-spin")} />
            </button>

            {sortResult && (
              <button
                className={cn(
                  "px-3 py-1.5 text-xs rounded-md border transition-colors",
                  showMislabelsOnly
                    ? "bg-red-50 border-red-300 text-red-700 dark:bg-red-950 dark:border-red-800 dark:text-red-300"
                    : "border-border text-muted-foreground hover:text-foreground"
                )}
                onClick={() => {
                  setShowMislabelsOnly(!showMislabelsOnly);
                  setSelectedIds(new Set());
                }}
              >
                Show mislabels
              </button>
            )}

            {sortResult && (
              <span className="text-xs text-muted-foreground ml-auto">
                {sortResult.total_detections} detection{sortResult.total_detections !== 1 ? "s" : ""}
              </span>
            )}
          </>
        )}

        {/* Search controls */}
        {viewMode === "search" && anchorId && searchResult && (
          <>
            {/* Anchor chip */}
            <div className="flex items-center gap-1.5 bg-muted rounded-md px-2 py-1">
              <img
                src={`${API_BASE_URL}${searchResult.anchor.crop_url}`}
                alt="anchor"
                className="h-8 w-8 rounded object-cover"
              />
              <span className="text-xs capitalize">
                {searchResult.anchor.species || searchResult.anchor.category}
              </span>
            </div>

            <div className="h-4 w-px bg-border" />

            {/* Threshold slider */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Threshold:</span>
              <Slider
                value={[threshold]}
                onValueChange={([v]) => setThreshold(v)}
                min={0}
                max={1}
                step={0.05}
                className="w-[120px]"
              />
              <span className="text-xs font-mono w-[36px]">
                {threshold.toFixed(2)}
              </span>
            </div>

            {/* Result count */}
            <span className="text-xs text-muted-foreground ml-auto">
              {searchResult.total_results} result{searchResult.total_results !== 1 ? "s" : ""}
            </span>

            {/* Close button */}
            <button
              onClick={handleCloseSearch}
              className="text-muted-foreground hover:text-foreground transition-colors"
              title="Close search (Esc)"
            >
              <X className="h-4 w-4" />
            </button>
          </>
        )}
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : !hasResults ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16 text-center">
            <Search className="h-12 w-12 text-muted-foreground/50 mb-4" />
            <p className="text-lg font-medium text-muted-foreground">
              No search active
            </p>
            <p className="text-sm text-muted-foreground mt-1 max-w-md">
              Right-click on a detection and select &quot;Find similar&quot; to search
              for visually similar observations.
            </p>
          </CardContent>
        </Card>
      ) : (
        <CropGrid
          detections={allDetections}
          selectedIds={selectedIds}
          onSelect={handleSelect}
          onCardClick={handleCardClick}
          onFindSimilar={handleFindSimilar}
          onRelabel={handleRelabel}
        />
      )}

      <BulkActionBar
        selectedIds={selectedIds}
        onDeselectAll={() => setSelectedIds(new Set())}
        onFindSimilar={handleFindSimilar}
        labelOptions={labelOptions}
        labelOptionsLoading={labelOptionsLoading}
        onActionComplete={handleActionComplete}
        onRelabel={handleBulkRelabel}
        onVerify={handleBulkVerify}
      />

      <DetectionDetailSheet
        detection={detailDetection}
        open={!!detailDetection}
        onOpenChange={(open) => {
          if (!open) setDetailDetection(null);
        }}
        onFindSimilar={handleFindSimilar}
        onActionComplete={handleActionComplete}
      />
    </div>
  );
}
