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
import { AlertTriangle, Check, CircleHelp, Keyboard, Loader2, Layers, RefreshCw, Search } from "lucide-react";
import { toast } from "sonner";
import { similarityApi } from "../../api/similarity";
import { detectionsApi } from "../../api/detections";
import { projectsApi } from "../../api/projects";
import { Alert, AlertDescription } from "../ui/alert";
import { Button } from "../ui/button";
import { Card, CardContent } from "../ui/card";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { Slider } from "../ui/slider";
import { API_BASE_URL } from "../../lib/api-client";
import { cn } from "../../lib/utils";
import { CropGrid } from "./CropGrid";
import type { TileSize } from "./CropGrid";
import { BulkActionBar } from "./BulkActionBar";
import { DetectionDetailModal } from "./DetectionDetailModal";
import { FilterPanel } from "./FilterPanel";
import { SimilaritySettings } from "./SimilaritySettings";
import { SimilarityHelpSheet } from "./SimilarityHelpSheet";
import { LabelPicker } from "./LabelPicker";
import { SimilarityWelcomePopover } from "./SimilarityWelcomePopover";
import { ReEmbedModal } from "../projects/ReEmbedModal";
import { useLabelOptions, type LabelOption } from "../../hooks/useLabelOptions";
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

  // ── Local settings state (persisted to localStorage) ────────────────
  const LS_KEY = "addaxai:similaritySettings";
  const savedSettings = useMemo(() => {
    try { return JSON.parse(localStorage.getItem(LS_KEY) || "{}"); }
    catch { return {}; }
  }, []);
  const persistSetting = useCallback((key: string, value: unknown) => {
    try {
      const cur = JSON.parse(localStorage.getItem(LS_KEY) || "{}");
      cur[key] = value;
      localStorage.setItem(LS_KEY, JSON.stringify(cur));
    } catch { /* ignore */ }
  }, []);

  const [reverseSort, _setReverseSort] = useState(savedSettings.reverseSort ?? false);
  const setReverseSort = useCallback((v: boolean) => { _setReverseSort(v); persistSetting("reverseSort", v); }, [persistSetting]);
  const [tileSize, _setTileSize] = useState<TileSize>(savedSettings.tileSize ?? "M");
  const setTileSize = useCallback((v: TileSize) => { _setTileSize(v); persistSetting("tileSize", v); }, [persistSetting]);

  type VerificationFilter = "all" | "unverified" | "suspicious";
  const [verificationFilter, _setVerificationFilter] = useState<VerificationFilter>(
    savedSettings.verificationFilter ?? "unverified"
  );
  const setVerificationFilter = useCallback((v: VerificationFilter) => {
    _setVerificationFilter(v);
    persistSetting("verificationFilter", v);
  }, [persistSetting]);
  const [showSpeciesDividers, _setShowSpeciesDividers] = useState(savedSettings.showSpeciesDividers ?? false);
  const setShowSpeciesDividers = useCallback((v: boolean) => { _setShowSpeciesDividers(v); persistSetting("showSpeciesDividers", v); }, [persistSetting]);

  // Help sheet + welcome popover
  const [helpOpen, setHelpOpen] = useState(false);
  const [showWelcome, setShowWelcome] = useState(
    () => !localStorage.getItem("addaxai:similarityWelcomeDismissed")
  );
  const handleDismissWelcome = useCallback(() => {
    setShowWelcome(false);
    localStorage.setItem("addaxai:similarityWelcomeDismissed", "1");
  }, []);

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

  // Project query for shortcut_labels
  const { data: project } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => projectsApi.get(projectId),
  });

  const [shortcutLabels, setShortcutLabels] = useState<Record<number, LabelOption>>({});
  const [showShortcuts, setShowShortcuts] = useState(false);

  useEffect(() => {
    if (project?.shortcut_labels) {
      const parsed: Record<number, LabelOption> = {};
      for (const [k, v] of Object.entries(project.shortcut_labels)) {
        parsed[Number(k)] = v as LabelOption;
      }
      setShortcutLabels(parsed);
    }
  }, [project?.shortcut_labels]);

  const updateShortcutLabels = useCallback(
    (updater: (prev: Record<number, LabelOption>) => Record<number, LabelOption>) => {
      setShortcutLabels((prev) => {
        const next = updater(prev);
        projectsApi.update(projectId, { shortcut_labels: next });
        return next;
      });
    },
    [projectId]
  );

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
    let dets: DetectionSummary[] = [];
    if (viewMode === "sort" && sortResult) dets = sortResult.detections;
    else if (viewMode === "search" && searchResult) dets = searchResult.results;

    if (verificationFilter === "unverified") {
      dets = dets.filter((d) => !d.verified);
    } else if (verificationFilter === "suspicious") {
      dets = dets.filter(
        (d) => !d.verified && d.neighbor_agreement != null && d.neighbor_agreement < 0.7
      );
    }
    return dets;
  }, [viewMode, sortResult, searchResult, verificationFilter]);

  // Unfiltered count to detect "all hidden by filters" vs "genuinely empty"
  const totalCount = useMemo(() => {
    if (viewMode === "sort" && sortResult) return sortResult.detections.length;
    if (viewMode === "search" && searchResult) return searchResult.results.length;
    return 0;
  }, [viewMode, sortResult, searchResult]);

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

      // Skip grid shortcuts when detail sheet is open (sheet handles its own keys)
      if (detailDetection) return;

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
        return;
      }

      if (e.key >= "1" && e.key <= "5" && !e.ctrlKey && !e.metaKey) {
        const slot = parseInt(e.key);
        const label = shortcutLabels[slot];
        if (!label || selectedIds.size === 0) return;
        e.preventDefault();
        const ids = Array.from(selectedIds);
        detectionsApi.bulkRelabel(ids, label.species, label.category).then(() => {
          const idSet = new Set(ids);
          patchLocalDetections((d) =>
            idSet.has(d.detection_id)
              ? { ...d, species: label.species ?? label.category, category: label.category, verified: true }
              : d
          );
          setSelectedIds(new Set());
        });
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedIds, detailDetection, allDetections, handleActionComplete, viewMode, handleCloseSearch, relabelToSuggestions, shortcutLabels, patchLocalDetections]);

  // Click outside grid to deselect
  useEffect(() => {
    if (selectedIds.size === 0) return;
    function handleClick(e: MouseEvent) {
      const el = e.target as HTMLElement;
      if (el.closest("[data-crop-card], button, a, input, select, [role='menu'], [role='dialog'], [data-radix-popper-content-wrapper]")) return;
      setSelectedIds(new Set());
    }
    document.addEventListener("click", handleClick);
    return () => document.removeEventListener("click", handleClick);
  }, [selectedIds.size > 0]); // eslint-disable-line react-hooks/exhaustive-deps

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
      {/* Filter panel — sites, dates, species only (verification filtering
          is handled by the toolbar segmented control) */}
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
      <div className="flex flex-wrap items-center gap-3 py-2 px-3 bg-white rounded-lg border shadow-sm">
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
              showSpeciesDividers={showSpeciesDividers}
              onShowSpeciesDividersChange={setShowSpeciesDividers}
              tileSize={tileSize}
              onTileSizeChange={setTileSize}
            />

            <Popover open={showShortcuts} onOpenChange={setShowShortcuts}>
              <PopoverTrigger asChild>
                <button
                  className="text-muted-foreground hover:text-foreground transition-colors"
                  title="Keyboard shortcuts"
                >
                  <Keyboard className="h-4 w-4" />
                </button>
              </PopoverTrigger>
              <PopoverContent align="end" className="w-auto px-4 py-3">
                <div className="flex gap-8">
                  {/* Left column: grid shortcuts */}
                  <div>
                    {[
                      ["Click", "Open detail"],
                      [navigator.platform.includes("Mac") ? "Cmd + Click" : "Ctrl + Click", "Select"],
                      ["Shift + Click", "Select range"],
                      ["Click outside", "Deselect all"],
                      ["Enter", "Verify selected"],
                      ["R", "Accept suggestions"],
                      [navigator.platform.includes("Mac") ? "Cmd + A" : "Ctrl + A", "Select all"],
                      ["Esc", "Deselect / close"],
                    ].map(([key, action]) => (
                      <div key={key} className="flex items-center text-xs gap-3 h-7">
                        <code className="bg-zinc-100 text-zinc-500 px-1.5 py-0.5 rounded text-[11px] w-24 shrink-0 text-center whitespace-nowrap">{(key as string).split("+").map((part, i, arr) => <span key={i}>{part}{i < arr.length - 1 && <span className="text-[#bbbbc1]">+</span>}</span>)}</code>
                        <span>{action}</span>
                      </div>
                    ))}
                  </div>

                  {/* Right column: label shortcuts 1-5 */}
                  <div>
                    {[1, 2, 3, 4, 5].map((n) => (
                      <div key={n} className="flex items-center text-xs gap-3 h-7">
                        <code className="bg-zinc-100 text-zinc-500 px-1.5 py-0.5 rounded text-[11px] w-24 shrink-0 text-center whitespace-nowrap">{n}</code>
                        <span>Change selected to</span>
                        <LabelPicker
                          value={shortcutLabels[n]?.value ?? null}
                          onSelect={(option) =>
                            updateShortcutLabels((prev) => ({ ...prev, [n]: option }))
                          }
                          options={labelOptions}
                          isLoading={labelOptionsLoading}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              </PopoverContent>
            </Popover>

            <button
              onClick={() => setHelpOpen(true)}
              className="text-muted-foreground hover:text-foreground transition-colors"
              title="Help"
            >
              <CircleHelp className="h-4 w-4" />
            </button>

            {sortResult && (() => {
              const dets = sortResult.detections;
              const total = dets.length;
              const unverified = dets.filter((d) => !d.verified).length;
              const suspicious = dets.filter(
                (d) => !d.verified && d.neighbor_agreement != null && d.neighbor_agreement < 0.7
              ).length;
              const hasAgreement = dets.some((d) => d.neighbor_agreement != null);
              const verified = total - unverified;
              const verifiedPct = total > 0 ? (verified / total) * 100 : 0;

              return (
                <div className="flex items-center gap-3 ml-auto">
                  <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    Verified
                    <div className="relative h-2 w-20 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full transition-all duration-500 ease-out rounded-full"
                        style={{ width: `${verifiedPct}%`, backgroundColor: "#0f6064" }}
                      />
                    </div>
                    {Math.round(verifiedPct)}%
                  </div>
                  <div className="h-4 w-px bg-border" />
                  <div className="flex items-center rounded-lg bg-muted p-0.5 text-xs">
                    <button
                      className={cn(
                        "px-2.5 py-1 rounded-md transition-colors font-medium flex items-center gap-1.5",
                        verificationFilter === "all"
                          ? "bg-background text-foreground shadow-sm"
                          : "text-muted-foreground hover:text-foreground"
                      )}
                      onClick={() => { setVerificationFilter("all"); setSelectedIds(new Set()); }}
                    >
                      <span className="inline-block h-2 w-2 rounded-full" style={{ background: "#0f6064" }} />
                      All ({total})
                    </button>
                    <button
                      className={cn(
                        "px-2.5 py-1 rounded-md transition-colors font-medium flex items-center gap-1.5",
                        verificationFilter === "unverified"
                          ? "bg-background text-foreground shadow-sm"
                          : "text-muted-foreground hover:text-foreground"
                      )}
                      onClick={() => { setVerificationFilter("unverified"); setSelectedIds(new Set()); }}
                    >
                      <span className="inline-block h-2 w-2 rounded-full" style={{ background: "#71b7ba" }} />
                      Unverified ({unverified})
                    </button>
                    {hasAgreement && (
                      <button
                        className={cn(
                          "px-2.5 py-1 rounded-md transition-colors font-medium flex items-center gap-1.5",
                          verificationFilter === "suspicious"
                            ? "bg-background text-foreground shadow-sm"
                            : "text-muted-foreground hover:text-foreground"
                        )}
                        onClick={() => { setVerificationFilter("suspicious"); setSelectedIds(new Set()); }}
                      >
                        <span className="inline-block h-2 w-2 rounded-full" style={{ background: "#882000" }} />
                        Suspicious ({suspicious})
                      </button>
                    )}
                  </div>
                </div>
              );
            })()}
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
      ) : allDetections.length === 0 && totalCount > 0 && verificationFilter === "suspicious" ? (
        <div className="flex flex-col items-center justify-center py-20 text-center text-muted-foreground">
          <p className="text-sm">No suspicious labels in the current selection.</p>
          <p className="text-xs mt-1">All detections have been verified or have high neighbor agreement.</p>
        </div>
      ) : allDetections.length === 0 && totalCount > 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-center text-muted-foreground">
          <Check className="h-8 w-8 mb-3 text-muted-foreground/60" />
          <p className="text-sm">All {totalCount} detections in this view are verified.</p>
          <p className="text-xs mt-1">Switch to &quot;All&quot; to see them.</p>
        </div>
      ) : (
        <div style={{ paddingBottom: selectedIds.size > 0 ? 80 : 0 }}>
          <CropGrid
            detections={allDetections}
            selectedIds={selectedIds}
            onSelect={handleSelect}
            onCardClick={handleCardClick}
            onFindSimilar={handleFindSimilar}
            onRelabel={handleRelabel}
            onBackgroundClick={() => setSelectedIds(new Set())}
            tileSize={tileSize}
            showSpeciesDividers={viewMode === "sort" && showSpeciesDividers}
          />
        </div>
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
        suggestionCount={
          allDetections.filter(
            (d) => selectedIds.has(d.detection_id) && !d.verified && d.neighbor_top_label && d.neighbor_top_label !== d.species
          ).length
        }
        onAcceptSuggestions={() => {
          const selectedDets = allDetections.filter((d) => selectedIds.has(d.detection_id));
          relabelToSuggestions(selectedDets);
        }}
      />

      <SimilarityWelcomePopover open={showWelcome} onDismiss={handleDismissWelcome} />
      <SimilarityHelpSheet open={helpOpen} onOpenChange={setHelpOpen} />

      <DetectionDetailModal
        detection={detailDetection}
        open={!!detailDetection}
        onOpenChange={(open) => {
          if (!open) setDetailDetection(null);
        }}
        onFindSimilar={handleFindSimilar}
        onActionComplete={handleActionComplete}
        projectId={projectId}
        onRelabel={(detectionId, species, category) => {
          patchLocalDetections((d) =>
            d.detection_id === detectionId
              ? { ...d, species, category, verified: true }
              : d
          );
        }}
        onVerify={(detectionId) => {
          patchLocalDetections((d) =>
            d.detection_id === detectionId ? { ...d, verified: true } : d
          );
        }}
        position={
          detailDetection
            ? `${allDetections.findIndex((d) => d.detection_id === detailDetection.detection_id) + 1} / ${allDetections.length}`
            : undefined
        }
        onNavigate={(direction) => {
          if (!detailDetection) return false;
          const idx = allDetections.findIndex(
            (d) => d.detection_id === detailDetection.detection_id
          );
          if (idx === -1) return false;

          if (direction === "nextUnverified") {
            // Find next unverified after current index, wrapping around
            for (let i = 1; i <= allDetections.length; i++) {
              const candidate = allDetections[(idx + i) % allDetections.length];
              if (!candidate.verified) {
                setDetailDetection(candidate);
                return true;
              }
            }
            // All verified — close the sheet
            toast.success("All detections verified");
            setDetailDetection(null);
            return false;
          }

          const nextIdx =
            direction === "next"
              ? Math.min(idx + 1, allDetections.length - 1)
              : Math.max(idx - 1, 0);
          if (nextIdx === idx) return false;
          setDetailDetection(allDetections[nextIdx]);
          return true;
        }}
      />
    </div>
  );
}

