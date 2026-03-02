/**
 * SimilarityTab - orchestrates the embedding-driven similarity view.
 *
 * Manages its own filter state (independent from Events tab) via sim_* URL
 * params. Provides sort/search mode via segmented control, selection model,
 * and coordinates toolbar, grid, bulk actions, settings, and detail sheet.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Check, Loader2, Layers, RefreshCw, Search } from "lucide-react";
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
import type { TileSize } from "./CropGrid";
import { BulkActionBar } from "./BulkActionBar";
import { DetectionDetailSheet } from "./DetectionDetailSheet";
import { FilterPanel } from "./FilterPanel";
import { SimilaritySettings } from "./SimilaritySettings";
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
  classificationModelId: string | null;
}

// ── Sim filter state (independent from Events filters) ──────────────────

interface SimilarityFilterState {
  site_ids?: string[];
  date_from?: string;
  date_to?: string;
  species?: string[];
}

/** Parse sim_* params from URL. */
function simFiltersFromSearchParams(sp: URLSearchParams): SimilarityFilterState {
  const f: SimilarityFilterState = {};
  const sites = sp.get("sim_sites");
  if (sites) f.site_ids = sites.split(",");
  const from = sp.get("sim_from");
  if (from) f.date_from = from;
  const to = sp.get("sim_to");
  if (to) f.date_to = to;
  const species = sp.get("sim_species");
  if (species) f.species = species.split(",");
  return f;
}

/** Write sim_* params to URL, preserving non-sim params. */
function simFiltersToSearchParams(
  filters: SimilarityFilterState,
  current: URLSearchParams,
): URLSearchParams {
  const sp = new URLSearchParams(current);
  // Clear all sim_* keys first
  for (const key of [...sp.keys()]) {
    if (key.startsWith("sim_")) sp.delete(key);
  }
  if (filters.site_ids?.length) sp.set("sim_sites", filters.site_ids.join(","));
  if (filters.date_from) sp.set("sim_from", filters.date_from);
  if (filters.date_to) sp.set("sim_to", filters.date_to);
  if (filters.species?.length) sp.set("sim_species", filters.species.join(","));
  return sp;
}

/** Convert SimilarityFilterState → SimilarityFilters for API calls. */
function toSimilarityFilters(f: SimilarityFilterState): SimilarityFilters {
  return {
    species: f.species,
    site_ids: f.site_ids,
    date_from: f.date_from,
    date_to: f.date_to,
  };
}

/** Adapt SimilarityFilterState to EventFilterParams shape for FilterPanel. */
function toFilterPanelFilters(f: SimilarityFilterState): EventFilterParams {
  return {
    site_ids: f.site_ids,
    date_from: f.date_from,
    date_to: f.date_to,
    species: f.species,
  };
}

export function SimilarityTab({
  projectId,
  classificationModelId,
}: SimilarityTabProps) {
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();

  // ── Own filter state from URL sim_* params ──────────────────────────
  const simFilters = useMemo(
    () => simFiltersFromSearchParams(searchParams),
    [searchParams],
  );

  const setSimFilters = useCallback(
    (next: SimilarityFilterState) => {
      setSearchParams(
        (prev) => simFiltersToSearchParams(next, prev),
        { replace: true },
      );
    },
    [setSearchParams],
  );

  /** Handler for FilterPanel onChange (EventFilterParams shape). */
  const handleFilterPanelChange = useCallback(
    (fp: EventFilterParams) => {
      setSimFilters({
        ...simFilters,
        site_ids: fp.site_ids,
        date_from: fp.date_from,
        date_to: fp.date_to,
        species: fp.species,
      });
    },
    [simFilters, setSimFilters],
  );

  // ── Local settings state ────────────────────────────────────────────
  const [reverseSort, setReverseSort] = useState(false);
  const [autoHideVerified, setAutoHideVerified] = useState(false);
  const [tileSize, setTileSize] = useState<TileSize>("M");
  const [showMislabelsOnly, setShowMislabelsOnly] = useState(false);
  const [showSpeciesDividers, setShowSpeciesDividers] = useState(false);

  // Explicit sorting flag — avoids isPending getting stuck in Strict Mode
  const [isSorting, setIsSorting] = useState(false);

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

  // Sort mutation — passes reverse flag
  const sortMutation = useMutation({
    mutationFn: () =>
      similarityApi.sort(projectId, {
        filters: toSimilarityFilters(simFilters),
        reverse: reverseSort,
      }),
    onMutate: () => setIsSorting(true),
    onSuccess: (data) => {
      setSortResult(data);
      setSelectedIds(new Set());
      setShowMislabelsOnly(false);
      setIsSorting(false);
    },
    onError: (err: Error) => {
      toast.error(err.message);
      setIsSorting(false);
    },
  });

  // Stable key for filter + settings comparison
  const filtersKey = JSON.stringify(toSimilarityFilters(simFilters));
  const sortKey = `${filtersKey}|${reverseSort}`;
  const lastSortKeyRef = useRef<string | null>(null);

  // Auto-sort on mount and when filters / reverseSort change
  useEffect(() => {
    if (viewMode === "sort" && stats?.embedded_detections && sortKey !== lastSortKeyRef.current) {
      lastSortKeyRef.current = sortKey;
      sortMutation.mutate();
    }
  }, [viewMode, sortKey, stats?.embedded_detections]); // eslint-disable-line react-hooks/exhaustive-deps

  // Search mutation
  const searchMutation = useMutation({
    mutationFn: (anchor: string) =>
      similarityApi.search(projectId, {
        anchor_detection_id: anchor,
        filters: toSimilarityFilters(simFilters),
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
      let dets = sortResult.detections;
      if (showMislabelsOnly) {
        dets = dets.filter(
          (d) => d.neighbor_agreement != null && d.neighbor_agreement < 0.5
        );
      }
      if (autoHideVerified) {
        dets = dets.filter((d) => !d.verified);
      }
      return dets;
    }
    if (viewMode === "search" && searchResult) {
      let dets = searchResult.results;
      if (autoHideVerified) {
        dets = dets.filter((d) => !d.verified);
      }
      return dets;
    }
    return [];
  }, [viewMode, sortResult, searchResult, showMislabelsOnly, autoHideVerified]);

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

  /** Relabel detections to their neighbor_top_label suggestions, grouped by (label, category). */
  const relabelToSuggestions = useCallback(
    async (dets: DetectionSummary[]) => {
      const withSuggestion = dets.filter(
        (d) => d.neighbor_top_label && d.neighbor_top_label !== d.species
      );
      if (withSuggestion.length === 0) {
        toast.info("No suggestions to accept");
        return;
      }
      // Group by (neighbor_top_label, category)
      const groups = new Map<string, { ids: string[]; species: string; category: string }>();
      for (const d of withSuggestion) {
        const key = `${d.neighbor_top_label}|${d.category}`;
        if (!groups.has(key)) {
          groups.set(key, { ids: [], species: d.neighbor_top_label!, category: d.category });
        }
        groups.get(key)!.ids.push(d.detection_id);
      }
      let totalUpdated = 0;
      for (const { ids, species, category } of groups.values()) {
        try {
          const data = await detectionsApi.bulkRelabel(ids, species, category);
          totalUpdated += data.updated_count;
        } catch (err: unknown) {
          toast.error(err instanceof Error ? err.message : "Relabel failed");
        }
      }
      // Patch local state
      const suggestionMap = new Map(
        withSuggestion.map((d) => [d.detection_id, d.neighbor_top_label!])
      );
      patchLocalDetections((d) =>
        suggestionMap.has(d.detection_id)
          ? { ...d, species: suggestionMap.get(d.detection_id)!, verified: true }
          : d
      );
      setSelectedIds(new Set());
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
              handleBulkVerify(ids);
              setSelectedIds(new Set());
            });
        });
        return;
      }

      if ((e.key === "r" || e.key === "R") && !e.ctrlKey && !e.metaKey && selectedIds.size > 0) {
        e.preventDefault();
        // Relabel selected to their neighbor suggestions
        const selectedDets = allDetections.filter((d) => selectedIds.has(d.detection_id));
        relabelToSuggestions(selectedDets);
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
  }, [selectedIds, detailDetection, allDetections, handleActionComplete, viewMode, handleCloseSearch, relabelToSuggestions]);

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

  const hasResults =
    (viewMode === "sort" && sortResult !== null) ||
    (viewMode === "search" && searchResult !== null);
  // Show spinner only when actively loading AND no results yet.
  const isLoading =
    (isSorting || searchMutation.isPending) && !hasResults;

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
      {/* Filter panel — sites, dates, species only (no verification filter;
          "Hide as I verify" in SimilaritySettings covers that workflow) */}
      <FilterPanel
        filters={toFilterPanelFilters(simFilters)}
        onChange={handleFilterPanelChange}
        projectId={projectId}
        isOpen={true}
        onToggle={() => {}}
        classificationModelId={classificationModelId}
        verificationSection={null}
      />

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
              disabled={isSorting || !stats?.embedded_detections}
              className="text-muted-foreground hover:text-foreground disabled:opacity-50 transition-colors"
              title="Re-sort"
            >
              <RefreshCw className={cn("h-4 w-4", isSorting && "animate-spin")} />
            </button>

            <SimilaritySettings
              reverseSort={reverseSort}
              onReverseSortChange={setReverseSort}
              autoHideVerified={autoHideVerified}
              onAutoHideVerifiedChange={setAutoHideVerified}
              showMislabelsOnly={showMislabelsOnly}
              onShowMislabelsOnlyChange={(v) => {
                setShowMislabelsOnly(v);
                setSelectedIds(new Set());
              }}
              showSpeciesDividers={showSpeciesDividers}
              onShowSpeciesDividersChange={setShowSpeciesDividers}
              tileSize={tileSize}
              onTileSizeChange={setTileSize}
            />

            {/* Accept all suggestions — visible when filtering to suspicious */}
            {showMislabelsOnly && allDetections.length > 0 && (
              <Button
                variant="outline"
                size="sm"
                className="h-7 text-xs gap-1"
                onClick={() => relabelToSuggestions(allDetections)}
              >
                <Check className="h-3.5 w-3.5" />
                Accept suggestions
              </Button>
            )}

            {sortResult && (
              <>
                {/* Agreement quality summary */}
                {(() => {
                  const dets = sortResult.detections;
                  let agreed = 0, suspicious = 0;
                  for (const d of dets) {
                    if (d.neighbor_agreement == null) continue;
                    if (d.neighbor_agreement >= 0.5) agreed++;
                    else suspicious++;
                  }
                  if (agreed + suspicious === 0) return null;
                  return (
                    <div className="flex items-center gap-2 text-xs text-muted-foreground ml-auto">
                      <button
                        className="flex items-center gap-1 hover:text-foreground transition-colors"
                        title="Neighbors agree with current label — click to show all"
                        onClick={() => { setShowMislabelsOnly(false); }}
                      >
                        <span className="inline-block h-2 w-2 rounded-full" style={{ background: "#0f6064" }} />
                        Agreed labels: {agreed}
                      </button>
                      <button
                        className="flex items-center gap-1 hover:text-foreground transition-colors"
                        title="Neighbors disagree with current label — click to filter"
                        onClick={() => { setShowMislabelsOnly(true); setSelectedIds(new Set()); }}
                      >
                        <span className="inline-block h-2 w-2 rounded-full" style={{ background: "#882000" }} />
                        Suspicious labels: {suspicious}
                      </button>
                      <span className="ml-1">
                        {allDetections.length !== sortResult.total_detections
                          ? `${allDetections.length} of ${sortResult.total_detections}`
                          : sortResult.total_detections}{" "}
                        detection{sortResult.total_detections !== 1 ? "s" : ""}
                      </span>
                    </div>
                  );
                })()}

                {/* Fallback count when no agreement data */}
                {sortResult.detections.every((d) => d.neighbor_agreement == null) && (
                  <span className="text-xs text-muted-foreground ml-auto">
                    {allDetections.length !== sortResult.total_detections
                      ? `${allDetections.length} of ${sortResult.total_detections}`
                      : sortResult.total_detections}{" "}
                    detection{sortResult.total_detections !== 1 ? "s" : ""}
                  </span>
                )}
              </>
            )}
          </>
        )}

        {/* Search controls */}
        {viewMode === "search" && anchorId && searchResult && (
          <>
            {/* Anchor chip */}
            <div className="flex items-center gap-1.5 bg-background border rounded-md px-1.5 py-0.5">
              <img
                src={`${API_BASE_URL}${searchResult.anchor.crop_url}`}
                alt="anchor"
                className="h-5 w-5 rounded object-cover"
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
          tileSize={tileSize}
          showSpeciesDividers={viewMode === "sort" && showSpeciesDividers}
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
