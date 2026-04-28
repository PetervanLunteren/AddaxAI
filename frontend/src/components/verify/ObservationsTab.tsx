/**
 * ObservationsTab - orchestrates the embedding-driven observation grid.
 *
 * Manages its own filter state (independent from Events / Files tabs) via
 * obs_* URL params. Provides sort/search mode via segmented control,
 * selection model, and coordinates toolbar, grid, bulk actions, settings,
 * and detail sheet.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Check,
  CircleHelp,
  Layers,
  Loader2,
  RefreshCw,
  Search,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { observationsApi } from "../../api/observations";
import { detectionsApi } from "../../api/detections";
import { eventsApi } from "../../api/events";
import { projectsApi } from "../../api/projects";
import { sitesApi } from "../../api/sites";
import { Button } from "../ui/button";
import { Card, CardContent } from "../ui/card";
import { Slider } from "../ui/slider";
import { API_BASE_URL } from "../../lib/api-client";
import { invalidateProjectData } from "../../lib/invalidate-project";
import { CropGrid } from "./CropGrid";
import type { TileSize } from "./CropGrid";
import { BulkActionBar } from "./BulkActionBar";
import { DetectionDetailModal } from "./DetectionDetailModal";
import { FilterChips, hasAnyActiveFilter } from "./FilterChips";
import { VerifyFilterBar, type VerificationOption } from "./VerifyFilterBar";
import { SortSelector } from "./SortSelector";
import {
  VerifyProgressPill,
  VerifyToolbar,
  VerifyToolbarIcon,
} from "./VerifyToolbar";
import { getDetectionDisplayName } from "../../lib/detection-utils";
import { ObservationsSettings } from "./ObservationsSettings";
import { ObservationsKeyboardPopover } from "./ObservationsKeyboardPopover";
import { ObservationsHelpSheet } from "./ObservationsHelpSheet";
import { ObservationsWelcomePopover } from "./ObservationsWelcomePopover";
import { ReEmbedModal } from "../projects/ReEmbedModal";
import { useLabelOptions, type LabelOption } from "../../hooks/useLabelOptions";
import type {
  SortResponse,
  SearchResponse,
  DetectionSummary,
  ObservationFilters,
  ObservationSort,
  EventFilterParams,
  VerifySort,
} from "../../api/types";

const OBSERVATIONS_SORT_MODES: readonly VerifySort[] = [
  "similarity",
  "similarity_reverse",
  "newest",
  "oldest",
  "cls_low",
];

interface ObservationsTabProps {
  projectId: string;
  classificationModelId: string | null;
}

// ── Observations filter state (independent from Events / Files filters) ──

type ObservationsVerification = "all" | "unverified" | "suspicious";

interface ObservationsFilterState {
  site_ids?: string[];
  date_from?: string;
  date_to?: string;
  labels?: string[];
  min_confidence?: number;
  max_confidence?: number;
  min_label_confidence?: number;
  max_label_confidence?: number;
  /** Default "unverified" when omitted — verified detections are usually
   *  not what the user is looking at on this tab. */
  verification?: ObservationsVerification;
}

/** Parse obs_* params from URL. */
function obsFiltersFromSearchParams(sp: URLSearchParams): ObservationsFilterState {
  const f: ObservationsFilterState = {};
  const sites = sp.get("obs_sites");
  if (sites) f.site_ids = sites.split(",");
  const from = sp.get("obs_from");
  if (from) f.date_from = from;
  const to = sp.get("obs_to");
  if (to) f.date_to = to;
  const labels = sp.get("obs_labels");
  if (labels) f.labels = labels.split(",");
  const minC = sp.get("obs_min_confidence");
  if (minC !== null) f.min_confidence = parseFloat(minC);
  const maxC = sp.get("obs_max_confidence");
  if (maxC !== null) f.max_confidence = parseFloat(maxC);
  const minLC = sp.get("obs_min_label_confidence");
  if (minLC !== null) f.min_label_confidence = parseFloat(minLC);
  const maxLC = sp.get("obs_max_label_confidence");
  if (maxLC !== null) f.max_label_confidence = parseFloat(maxLC);
  const ver = sp.get("obs_verification");
  if (ver === "all" || ver === "unverified" || ver === "suspicious") {
    f.verification = ver;
  }
  return f;
}

/** Write obs_* params to URL, preserving non-obs params. */
function obsFiltersToSearchParams(
  filters: ObservationsFilterState,
  current: URLSearchParams,
): URLSearchParams {
  const sp = new URLSearchParams(current);
  for (const key of [...sp.keys()]) {
    if (key.startsWith("obs_")) sp.delete(key);
  }
  if (filters.site_ids?.length) sp.set("obs_sites", filters.site_ids.join(","));
  if (filters.date_from) sp.set("obs_from", filters.date_from);
  if (filters.date_to) sp.set("obs_to", filters.date_to);
  if (filters.labels?.length) sp.set("obs_labels", filters.labels.join(","));
  if (filters.min_confidence !== undefined)
    sp.set("obs_min_confidence", String(filters.min_confidence));
  if (filters.max_confidence !== undefined)
    sp.set("obs_max_confidence", String(filters.max_confidence));
  if (filters.min_label_confidence !== undefined)
    sp.set("obs_min_label_confidence", String(filters.min_label_confidence));
  if (filters.max_label_confidence !== undefined)
    sp.set("obs_max_label_confidence", String(filters.max_label_confidence));
  // "unverified" is the implicit default — no URL param when set to that.
  if (filters.verification && filters.verification !== "unverified") {
    sp.set("obs_verification", filters.verification);
  }
  return sp;
}

/** Convert ObservationsFilterState → ObservationFilters for API calls. */
function toObservationFilters(f: ObservationsFilterState): ObservationFilters {
  return {
    labels: f.labels,
    site_ids: f.site_ids,
    date_from: f.date_from,
    date_to: f.date_to,
    min_confidence: f.min_confidence,
    max_confidence: f.max_confidence,
    min_label_confidence: f.min_label_confidence,
    max_label_confidence: f.max_label_confidence,
  };
}

/** Adapt ObservationsFilterState to the EventFilterParams shape that
 *  VerifyFilterBar reads. The verified select lives on the bar and
 *  emits its value into `filters.verification`. */
function toFilterBarFilters(f: ObservationsFilterState): EventFilterParams {
  return {
    site_ids: f.site_ids,
    date_from: f.date_from,
    date_to: f.date_to,
    labels: f.labels,
    min_confidence: f.min_confidence,
    max_confidence: f.max_confidence,
    min_label_confidence: f.min_label_confidence,
    max_label_confidence: f.max_label_confidence,
    verification: f.verification ?? "unverified",
  };
}

export function ObservationsTab({
  projectId,
  classificationModelId,
}: ObservationsTabProps) {
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();

  // ── Own filter state from URL obs_* params ──────────────────────────
  const obsFilters = useMemo(
    () => obsFiltersFromSearchParams(searchParams),
    [searchParams],
  );

  const setObsFilters = useCallback(
    (next: ObservationsFilterState) => {
      setSearchParams(
        (prev) => obsFiltersToSearchParams(next, prev),
        { replace: true },
      );
    },
    [setSearchParams],
  );

  /** Handler for VerifyFilterBar onChange (EventFilterParams shape).
   *
   *  The bar collapses "all" → undefined upstream because Events / Files
   *  treat undefined as "no filter". On Observations the implicit default
   *  is "unverified", so we have to record "all" explicitly when the user
   *  picks it; otherwise the state falls back to the unverified default
   *  and the dropdown silently reverts. */
  const handleFilterBarChange = useCallback(
    (fp: EventFilterParams) => {
      const v = fp.verification;
      const verification: ObservationsVerification =
        v === "unverified" || v === "suspicious" ? v : "all";
      setObsFilters({
        ...obsFilters,
        site_ids: fp.site_ids,
        date_from: fp.date_from,
        date_to: fp.date_to,
        labels: fp.labels,
        min_confidence: fp.min_confidence,
        max_confidence: fp.max_confidence,
        min_label_confidence: fp.min_label_confidence,
        max_label_confidence: fp.max_label_confidence,
        verification,
      });
    },
    [obsFilters, setObsFilters],
  );

  // ── Local settings state (persisted to localStorage) ────────────────
  const LS_KEY = "addaxai:observationsSettings";
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

  const isObservationSort = (v: unknown): v is ObservationSort =>
    v === "similarity" ||
    v === "similarity_reverse" ||
    v === "newest" ||
    v === "oldest" ||
    v === "cls_low";

  const initialSort: ObservationSort = isObservationSort(savedSettings.sort)
    ? savedSettings.sort
    : savedSettings.reverseSort // migrate the dropped reverseSort flag
      ? "similarity_reverse"
      : "similarity";
  const [obsSort, _setObsSort] = useState<ObservationSort>(initialSort);
  const setObsSort = useCallback(
    (v: ObservationSort) => {
      _setObsSort(v);
      persistSetting("sort", v);
    },
    [persistSetting],
  );

  const [tileSize, _setTileSize] = useState<TileSize>(savedSettings.tileSize ?? "M");
  const setTileSize = useCallback((v: TileSize) => { _setTileSize(v); persistSetting("tileSize", v); }, [persistSetting]);

  // Verification filter is the bar's "Verified" select; default unverified.
  const verificationFilter: ObservationsVerification =
    obsFilters.verification ?? "unverified";

  const [showLabelDividers, _setShowLabelDividers] = useState(savedSettings.showLabelDividers ?? false);
  const setShowLabelDividers = useCallback((v: boolean) => { _setShowLabelDividers(v); persistSetting("showLabelDividers", v); }, [persistSetting]);

  // Toolbar sheet/popover state (welcome popover only; keyboard and
  // settings are self-contained popovers anchored to their toolbar
  // icons, so they own their own open state).
  const [helpOpen, setHelpOpen] = useState(false);
  const [relabelOpen, setRelabelOpen] = useState(false);
  const [showWelcome, setShowWelcome] = useState(
    () => !localStorage.getItem("addaxai:observationsWelcomeDismissed")
  );
  const handleDismissWelcome = useCallback(() => {
    setShowWelcome(false);
    localStorage.setItem("addaxai:observationsWelcomeDismissed", "1");
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
  const selectionAnchorRef = useRef<string | null>(null);

  const clearSelection = useCallback(() => {
    setSelectedIds(new Set());
    selectionAnchorRef.current = null;
  }, []);

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

  // Site + filter-options queries: needed so the chip row below can
  // resolve site IDs and label IDs into human-readable names. Same
  // query keys VerifyFilterBar uses so react-query dedupes the work.
  const { data: sites } = useQuery({
    queryKey: ["sites", projectId],
    queryFn: () => sitesApi.list(projectId),
    enabled: !!projectId && (obsFilters.site_ids?.length ?? 0) > 0,
  });
  const siteNames = useMemo(() => {
    const map: Record<string, string> = {};
    for (const s of sites ?? []) map[s.id] = s.name;
    return map;
  }, [sites]);

  const { data: filterOptions } = useQuery({
    queryKey: ["event-filter-options", projectId],
    queryFn: () => eventsApi.getFilterOptions(projectId),
    enabled: !!projectId,
  });

  const [shortcutLabels, setShortcutLabels] = useState<Record<number, LabelOption>>({});

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
    queryKey: ["observations-stats", projectId],
    queryFn: () => observationsApi.stats(projectId),
    enabled: !!projectId,
  });

  // Auto-trigger search when anchor is set from URL on mount
  useEffect(() => {
    if (urlAnchor) {
      setAnchorId(urlAnchor);
    }
  }, [urlAnchor]);

  // Sort mutation — passes the chosen sort enum.
  const sortMutation = useMutation({
    mutationFn: () =>
      observationsApi.sort(projectId, {
        filters: toObservationFilters(obsFilters),
        sort: obsSort,
      }),
    onMutate: () => setIsSorting(true),
    onSuccess: (data) => {
      setSortResult(data);
      clearSelection();
      setIsSorting(false);
    },
    onError: (err: Error) => {
      toast.error(err.message);
      setIsSorting(false);
    },
  });

  // Stable key for filter + sort comparison; drives auto re-sort.
  const filtersKey = JSON.stringify(toObservationFilters(obsFilters));
  const sortKey = `${filtersKey}|${obsSort}`;
  const lastSortKeyRef = useRef<string | null>(null);

  // Auto-sort on mount and when filters or sort mode change.
  useEffect(() => {
    if (viewMode === "sort" && stats?.embedded_detections && sortKey !== lastSortKeyRef.current) {
      lastSortKeyRef.current = sortKey;
      sortMutation.mutate();
    }
  }, [viewMode, sortKey, stats?.embedded_detections]); // eslint-disable-line react-hooks/exhaustive-deps

  // Search mutation
  const searchMutation = useMutation({
    mutationFn: (anchor: string) =>
      observationsApi.search(projectId, {
        anchor_detection_id: anchor,
        filters: toObservationFilters(obsFilters),
        limit: 100,
        threshold,
      }),
    onSuccess: (data) => {
      setSearchResult(data);
      clearSelection();
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

  // Pre-computed index lookup for O(1) range selection
  const idIndexMap = useMemo(() => {
    const map = new Map<string, number>();
    for (let i = 0; i < allDetections.length; i++) {
      map.set(allDetections[i].detection_id, i);
    }
    return map;
  }, [allDetections]);

  // Unfiltered count to detect "all hidden by filters" vs "genuinely empty"
  const totalCount = useMemo(() => {
    if (viewMode === "sort" && sortResult) return sortResult.detections.length;
    if (viewMode === "search" && searchResult) return searchResult.results.length;
    return 0;
  }, [viewMode, sortResult, searchResult]);

  const handleSelect = useCallback(
    (detectionId: string, e: React.MouseEvent) => {
      if (e.shiftKey && selectionAnchorRef.current) {
        // Shift+Click: select range from anchor to target
        setSelectedIds((prev) => {
          const startIdx = idIndexMap.get(selectionAnchorRef.current!);
          const endIdx = idIndexMap.get(detectionId);
          if (startIdx != null && endIdx != null) {
            const [lo, hi] = startIdx < endIdx ? [startIdx, endIdx] : [endIdx, startIdx];
            const next = new Set(prev);
            for (let i = lo; i <= hi; i++) next.add(allDetections[i].detection_id);
            return next;
          }
          return prev;
        });
        // anchor stays — allows repeated Shift+Click to adjust range
      } else if (e.ctrlKey || e.metaKey) {
        // Ctrl/Cmd+Click: toggle individual card
        setSelectedIds((prev) => {
          const next = new Set(prev);
          if (next.has(detectionId)) {
            next.delete(detectionId);
          } else {
            next.add(detectionId);
          }
          return next;
        });
        // Move anchor to this card so Shift+Click extends from here
        selectionAnchorRef.current = detectionId;
      } else {
        // Plain click: select only this card, deselect all others
        selectionAnchorRef.current = detectionId;
        setSelectedIds(new Set([detectionId]));
      }
    },
    [allDetections, idIndexMap]
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
    clearSelection();
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
    queryClient.invalidateQueries({ queryKey: ["label-tree"] });
  }, [viewMode, anchorId, queryClient]); // eslint-disable-line react-hooks/exhaustive-deps

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
      // Keep the detail modal in sync
      setDetailDetection((prev) => (prev ? patchFn(prev) : prev));
    },
    [viewMode, sortResult, searchResult]
  );

  const handleRelabel = useCallback(
    (detectionId: string, label: string, category: string) => {
      detectionsApi
        .bulkRelabel([detectionId], label, category)
        .then(() => {
          patchLocalDetections((d) =>
            d.detection_id === detectionId
              ? { ...d, label, category, display_name: null, verified: true }
              : d
          );
        })
        .catch((err: Error) => toast.error(err.message));
    },
    [patchLocalDetections]
  );

  const handleBulkRelabel = useCallback(
    (ids: string[], label: string | null, category: string, displayName: string) => {
      const idSet = new Set(ids);
      patchLocalDetections((d) =>
        idSet.has(d.detection_id)
          ? { ...d, label, category, display_name: displayName, verified: true }
          : d
      );
      queryClient.invalidateQueries({ queryKey: ["label-tree"] });
    },
    [patchLocalDetections, queryClient]
  );

  const handleMarkFalse = useCallback(
    (ids: string[]) => {
      const idSet = new Set(ids);
      detectionsApi
        .bulkRelabel(ids, "false detection", undefined)
        .then(() => {
          patchLocalDetections((d) =>
            idSet.has(d.detection_id)
              ? { ...d, label: "false detection", display_name: "False detection", verified: true }
              : d
          );
          clearSelection();
          queryClient.invalidateQueries({ queryKey: ["label-tree"] });
        })
        .catch((err: Error) => toast.error(err.message));
    },
    [patchLocalDetections, clearSelection, queryClient]
  );

  const handleBulkMarkFalse = useCallback(
    (ids: string[]) => {
      const idSet = new Set(ids);
      patchLocalDetections((d) =>
        idSet.has(d.detection_id)
          ? { ...d, label: "false detection", verified: true }
          : d
      );
      queryClient.invalidateQueries({ queryKey: ["label-tree"] });
    },
    [patchLocalDetections, queryClient]
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
        (d) => d.neighbor_top_label && d.neighbor_top_label !== d.label
      );
      if (withSuggestion.length === 0) {
        toast.info("No suggestions to accept");
        return;
      }
      // Group by (neighbor_top_label, category)
      const groups = new Map<string, { ids: string[]; label: string; category: string }>();
      for (const d of withSuggestion) {
        const key = `${d.neighbor_top_label}|${d.category}`;
        if (!groups.has(key)) {
          groups.set(key, { ids: [], label: d.neighbor_top_label!, category: d.category });
        }
        groups.get(key)!.ids.push(d.detection_id);
      }
      let totalUpdated = 0;
      for (const { ids, label, category } of groups.values()) {
        try {
          const data = await detectionsApi.bulkRelabel(ids, label, category);
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
          ? { ...d, label: suggestionMap.get(d.detection_id)!, verified: true }
          : d
      );
      clearSelection();
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
          clearSelection();
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
              clearSelection();
            });
        });
        return;
      }

      if ((e.key === "r" || e.key === "R") && !e.ctrlKey && !e.metaKey && selectedIds.size > 0) {
        e.preventDefault();
        setRelabelOpen((prev) => !prev);
        return;
      }

      if ((e.key === "f" || e.key === "F") && !e.ctrlKey && !e.metaKey && selectedIds.size > 0) {
        e.preventDefault();
        const first = selectedIds.values().next().value;
        if (first) handleFindSimilar(first);
        return;
      }

      if ((e.key === "a" || e.key === "A") && !e.ctrlKey && !e.metaKey && selectedIds.size > 0) {
        e.preventDefault();
        const selectedDets = allDetections.filter((d) => selectedIds.has(d.detection_id));
        relabelToSuggestions(selectedDets);
        return;
      }

      if ((e.key === "x" || e.key === "X") && !e.ctrlKey && !e.metaKey && selectedIds.size > 0) {
        e.preventDefault();
        handleMarkFalse(Array.from(selectedIds));
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
        detectionsApi.bulkRelabel(ids, label.label, label.category).then(() => {
          const idSet = new Set(ids);
          patchLocalDetections((d) =>
            idSet.has(d.detection_id)
              ? { ...d, label: label.label ?? label.category, category: label.category, verified: true }
              : d
          );
          clearSelection();
        });
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedIds, detailDetection, allDetections, handleActionComplete, viewMode, handleCloseSearch, relabelToSuggestions, shortcutLabels, patchLocalDetections, handleFindSimilar, handleMarkFalse]);

  // Click outside grid to deselect
  useEffect(() => {
    if (selectedIds.size === 0) return;
    function handleClick(e: MouseEvent) {
      const el = e.target as HTMLElement;
      if (el.closest("[data-crop-card], button, a, input, select, [role='menu'], [role='dialog'], [data-radix-popper-content-wrapper]")) return;
      clearSelection();
    }
    document.addEventListener("click", handleClick);
    return () => document.removeEventListener("click", handleClick);
  }, [selectedIds.size > 0]); // eslint-disable-line react-hooks/exhaustive-deps

  // Verified select options. "Suspicious" is only meaningful when at
  // least one detection in the current sort result has neighbor
  // agreement data (i.e. the result was produced by the similarity
  // pipeline). Hooks need to run before any early return below.
  const hasAgreementData = useMemo(() => {
    const dets = sortResult?.detections ?? [];
    return dets.some((d) => d.neighbor_agreement != null);
  }, [sortResult]);

  // Verified-detections progress for the toolbar pill.
  const verifiedPct = useMemo(() => {
    const dets = sortResult?.detections ?? [];
    if (dets.length === 0) return 0;
    const verified = dets.filter((d) => d.verified).length;
    return (verified / dets.length) * 100;
  }, [sortResult]);

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
            Run an analysis with an embedding model selected to use this tab.
            Embeddings are computed from detection crops using DINOv2.
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

  // "Suspicious" only makes sense when the result has neighbour
  // agreement data (i.e. came from the similarity pipeline). Hide it
  // otherwise so users don't pick a filter that selects nothing.
  const verificationOptions: VerificationOption[] = [
    { value: "all", label: "All" },
    { value: "unverified", label: "Unverified" },
    ...(hasAgreementData
      ? [{ value: "suspicious" as const, label: "Suspicious" }]
      : []),
  ];

  return (
    <div className="space-y-4">
      <VerifyFilterBar
        filters={toFilterBarFilters(obsFilters)}
        onChange={handleFilterBarChange}
        projectId={projectId}
        classificationModelId={classificationModelId}
        detectionFloor={project?.detection_threshold ?? 0}
        countBy="detection"
        verificationOptions={verificationOptions}
        showLikedFlaggedEmpty={false}
      />

      {hasAnyActiveFilter(toFilterBarFilters(obsFilters)) && (
        <FilterChips
          filters={toFilterBarFilters(obsFilters)}
          onChange={handleFilterBarChange}
          filteredCount={allDetections.length}
          totalCount={totalCount}
          siteNames={siteNames}
          displayLabels={filterOptions?.display_labels}
          detectionFloor={project?.detection_threshold ?? 0}
        />
      )}

      {/* Warning when embeddings are incomplete */}
      {stats && stats.missing_embeddings > 0 && (
        <div className="flex items-center gap-3 rounded-lg border border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200 px-4 py-3">
          <AlertTriangle className="h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400" />
          <div className="flex-1">
            <p className="text-sm font-medium">
              {stats.missing_embeddings} detection
              {stats.missing_embeddings !== 1 ? "s are" : " is"} not shown
            </p>
            <p className="text-xs text-amber-800 dark:text-amber-300">
              This grid only shows detections that have an embedding, no matter which sort mode you pick. Embeddings can be missing when embedding was switched off in settings, an error occurred during analysis, or detections were added manually via event verification. Click 'Embed now' to fix this.
            </p>
          </div>
          <Button variant="outline" size="sm" className="shrink-0" onClick={handleEmbedNow}>
            Embed now
          </Button>
        </div>
      )}

      <ReEmbedModal
        open={!!reEmbedJobId}
        onOpenChange={(open) => { if (!open) setReEmbedJobId(null); }}
        jobId={reEmbedJobId}
        onComplete={() => invalidateProjectData(queryClient, projectId)}
        onError={() => invalidateProjectData(queryClient, projectId)}
      />

      <VerifyToolbar>
        <VerifyToolbarIcon
          icon={CircleHelp}
          title="Help"
          onClick={() => setHelpOpen(true)}
        />
        <ObservationsKeyboardPopover
          shortcutLabels={shortcutLabels}
          onShortcutLabelsChange={updateShortcutLabels}
          labelOptions={labelOptions}
          labelOptionsLoading={labelOptionsLoading}
          projectId={projectId}
        />
        <ObservationsSettings
          showLabelDividers={showLabelDividers}
          onShowLabelDividersChange={setShowLabelDividers}
          tileSize={tileSize}
          onTileSizeChange={setTileSize}
          similaritySort={obsSort === "similarity" || obsSort === "similarity_reverse"}
        />
        <VerifyToolbarIcon
          icon={RefreshCw}
          title="Refresh"
          onClick={() => sortMutation.mutate()}
          spinning={isSorting}
          disabled={!stats?.embedded_detections || isSorting}
        />
        <SortSelector
          sort={obsSort}
          seed={null}
          availableSorts={OBSERVATIONS_SORT_MODES}
          onChange={(next) => {
            if (isObservationSort(next)) setObsSort(next);
          }}
        />
        {sortResult && (
          <VerifyProgressPill pct={verifiedPct} label="detections verified" />
        )}
      </VerifyToolbar>

      {viewMode === "search" && anchorId && searchResult && (
        <div className="flex flex-wrap items-center gap-3 min-h-12 py-2 px-3 bg-white rounded-lg border shadow-sm">
          <span className="text-xs font-medium text-muted-foreground">Searching</span>
          <div className="flex items-center gap-1.5 bg-background border rounded-md px-1.5 py-0.5">
            <img
              src={`${API_BASE_URL}${searchResult.anchor.crop_url}`}
              alt="anchor"
              className="h-5 w-5 rounded object-cover"
            />
            <span className="text-xs capitalize">
              {getDetectionDisplayName(searchResult.anchor)}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Threshold</span>
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
          <span className="text-xs text-muted-foreground">
            {searchResult.total_results} result
            {searchResult.total_results !== 1 ? "s" : ""}
          </span>
          <button
            type="button"
            onClick={handleCloseSearch}
            className="ml-auto text-muted-foreground hover:text-foreground transition-colors"
            title="Exit search"
            aria-label="Exit search"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : !hasResults ? (
        viewMode === "search" ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-16 text-center">
              <Search className="h-12 w-12 text-muted-foreground/50 mb-4" />
              <p className="text-lg font-medium text-muted-foreground">
                No search active
              </p>
              <p className="text-sm text-muted-foreground mt-1 max-w-md">
                Select a detection and press <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">F</code>,
                or click <code className="bg-zinc-100 px-1 py-0.5 rounded text-xs">Find similar</code> in
                the selection bar or the detail window, to search for
                visually similar observations.
              </p>
            </CardContent>
          </Card>
        ) : (
          // Sort mode auto-runs, so !hasResults here means the mutation
          // has not fired yet (e.g. stats still loading). Show a spinner
          // rather than the misleading "No search active" card.
          <div className="flex items-center justify-center h-64">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        )
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
      ) : allDetections.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16 text-center">
            <Layers className="h-12 w-12 text-muted-foreground/50 mb-4" />
            <p className="text-lg font-medium text-muted-foreground">
              No detections match your filters
            </p>
            <p className="text-sm text-muted-foreground mt-1 max-w-md">
              Try adjusting or clearing your filters to see more detections.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div style={{ paddingBottom: selectedIds.size > 0 ? 80 : 0 }}>
          <CropGrid
            detections={allDetections}
            selectedIds={selectedIds}
            onSelect={handleSelect}
            onDoubleClick={handleCardClick}
            onFindSimilar={handleFindSimilar}
            onRelabel={handleRelabel}
            onMarkFalse={(detectionId) => handleMarkFalse([detectionId])}
            onBackgroundClick={clearSelection}
            tileSize={tileSize}
            showLabelDividers={viewMode === "sort" && showLabelDividers}
          />
        </div>
      )}

      <BulkActionBar
        selectedIds={selectedIds}
        onDeselectAll={clearSelection}
        onFindSimilar={handleFindSimilar}
        labelOptions={labelOptions}
        labelOptionsLoading={labelOptionsLoading}
        onActionComplete={handleActionComplete}
        onRelabel={handleBulkRelabel}
        onVerify={handleBulkVerify}
        onMarkFalse={handleBulkMarkFalse}
        projectId={projectId}
        relabelOpen={relabelOpen}
        onRelabelOpenChange={setRelabelOpen}
        suggestionCount={
          allDetections.filter(
            (d) => selectedIds.has(d.detection_id) && !d.verified && d.neighbor_top_label && d.neighbor_top_label !== d.label
          ).length
        }
        onAcceptSuggestions={() => {
          const selectedDets = allDetections.filter((d) => selectedIds.has(d.detection_id));
          relabelToSuggestions(selectedDets);
        }}
      />

      <ObservationsWelcomePopover open={showWelcome} onDismiss={handleDismissWelcome} />
      <ObservationsHelpSheet open={helpOpen} onOpenChange={setHelpOpen} />

      <DetectionDetailModal
        detection={detailDetection}
        open={!!detailDetection}
        onOpenChange={(open) => {
          if (!open) setDetailDetection(null);
        }}
        onFindSimilar={handleFindSimilar}
        onActionComplete={handleActionComplete}
        projectId={projectId}
        labelOptions={labelOptions}
        labelOptionsLoading={labelOptionsLoading}
        onRelabel={(detectionId, label, category) => {
          patchLocalDetections((d) =>
            d.detection_id === detectionId
              ? { ...d, label, category, verified: true }
              : d
          );
        }}
        onMarkFalse={(detectionId) => {
          patchLocalDetections((d) =>
            d.detection_id === detectionId
              ? { ...d, label: "false detection", verified: true }
              : d
          );
        }}
        onVerify={(detectionId, verified = true) => {
          patchLocalDetections((d) =>
            d.detection_id === detectionId ? { ...d, verified } : d
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


