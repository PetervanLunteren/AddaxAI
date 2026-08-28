/**
 * LabelsTab - orchestrates the embedding-driven label grid.
 *
 * Manages its own filter state (independent from Events / Files tabs) via
 * lbl_* URL params. Provides sort/search mode via segmented control,
 * selection model, and coordinates toolbar, grid, bulk actions, settings,
 * and detail sheet.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  FILTER_DEBOUNCE_MS,
  useDebouncedValue,
} from "../../hooks/useDebouncedValue";
import {
  AlertTriangle,
  Check,
  Layers,
  Loader2,
  Maximize2,
  Minimize2,
} from "lucide-react";
import { toast } from "sonner";
import {
  labelsApi,
  type LabelsProgressEvent,
} from "../../api/labels";
import { detectionsApi } from "../../api/detections";
import { projectsApi } from "../../api/projects";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
import { Card, CardContent } from "../ui/card";
import { Progress } from "../ui/progress";
import {
  DEFAULT_CLASSIFICATION_GATE,
  MD_OUTPUT_CONFIDENCE_THRESHOLD,
} from "../../lib/confidence";
import { invalidateProjectData } from "../../lib/invalidate-project";
import { resolveSpeciesName } from "../../lib/species-name-mode";
import {
  fromFilterBarFilters,
  lblFiltersFromSearchParams,
  lblFiltersToSearchParams,
  toFilterBarFilters,
  type LabelsFilterState,
  type LabelsVerification,
} from "./labels-filters";
import { nextAfterActed, selectOnClick } from "./grid-selection";
import { GridEmptyState } from "./GridEmptyState";
import { CropGrid } from "./CropGrid";
import type { CropGridHandle } from "./CropGrid";
import { BulkActionBar } from "./BulkActionBar";
import { DetectionDetailModal } from "./DetectionDetailModal";
import { SuggestionsToolbarPill } from "./SuggestionsToolbarPill";
import { VerifyFilterBar } from "./VerifyFilterBar";
import { SortSelector } from "./SortSelector";
import { useWideModeControls } from "./wide-mode";
import {
  VerifyGuideLink,
  VerifyProgressPill,
  VerifyToolbar,
  VerifyToolbarIcon,
} from "./VerifyToolbar";
import { LabelsSettings } from "./LabelsSettings";
import { LabelsKeyboardPopover } from "./LabelsKeyboardPopover";
import { MOD, type Shortcut } from "./shortcuts";
import {
  persistLabelsSetting,
  readLabelsSettings,
  useTileSize,
} from "./labels-settings";
import { LabelsWelcomePopover } from "./LabelsWelcomePopover";
import { ReEmbedModal } from "../projects/ReEmbedModal";
import { useLabelsProgress } from "./useLabelsProgress";
import { useLabelOptions, type LabelOption } from "../../hooks/useLabelOptions";
import type {
  CohortItem,
  SortResponse,
  DetectionSummary,
  LabelFilters,
  LabelSort,
  EventFilterParams,
  VerifySort,
} from "../../api/types";
import { invalidateLabelQueries } from "../../lib/invalidate-label-queries";

// Labels dropdown, kept deliberately small: browse look-alikes or
// review by event. Low-confidence triage is the confidence slider's
// job, not a sort mode. Suggestions is a third mode reached via the
// toolbar pill, not this list.
const CROPS_SHORTCUTS: readonly Shortcut[] = [
  ["Click", "Select"],
  [`${MOD} + Click`, "Toggle select"],
  ["Shift + Click", "Extend range"],
  ["Double-click", "Open detail"],
  ["Click outside", "Deselect all"],
  ["Enter", "Verify selected"],
  ["X", "Mark false detection"],
  ["U", "Mark unknown (unidentifiable)"],
  ["R", "Relabel selected"],
  ["M", "Relabel to most common in selection"],
  [`${MOD} + A`, "Select all"],
  ["E", "Select next event to check (event sort)"],
  [`${MOD} + Z`, "Undo last action"],
  ["Esc", "Deselect / close"],
];

const LABELS_SORT_MODES: readonly VerifySort[] = [
  "similarity",
  "events",
];

/**
 * IDs of the first event that still has an unverified crop in view.
 *
 * The "By event" sort lays each event's crops out contiguously, so we
 * walk the list, take each contiguous run of one event_id, and return
 * the first run that contains an unverified crop. Returns every in-view
 * crop of that event (matching what the user sees under the filters),
 * so the existing bulk shortcuts act on the whole event at once. Empty
 * when nothing is left to verify.
 */
function firstUnverifiedEventDetectionIds(
  detections: DetectionSummary[],
): string[] {
  let i = 0;
  while (i < detections.length) {
    const eventId = detections[i].event_id;
    let j = i;
    while (j < detections.length && detections[j].event_id === eventId) j++;
    const run = detections.slice(i, j);
    if (run.some((d) => !d.verified)) return run.map((d) => d.detection_id);
    i = j;
  }
  return [];
}

interface LabelsTabProps {
  projectId: string;
  classificationModelId: string | null;
  /** Fires when the size of the active bulk selection changes. The
   *  folder-run Labels step uses it to hide its sticky Back / Continue
   *  bar while a selection is live, so the BulkActionBar doesn't sit
   *  on top of it and the user can't accidentally advance mid-action. */
  onSelectionChange?: (count: number) => void;
  /** Extra control rendered at the start of the toolbar row. The
   *  folder-run Labels step slots its "Analysis settings" button here. */
  toolbarExtra?: ReactNode;
  /** Bumping this re-runs the sort even when filters are unchanged.
   *  Used to refresh the grid after a reprocess rewrites labels. */
  refreshSignal?: number;
  /** Labels not yet checked in the Empties tab, for the pointer shown
   *  when this grid runs out. */
  otherTabLeft?: number;
  /** Unverified labels in this tab, and every label in scope. Both feed
   *  the empty state so it can tell "you finished" from "your filters
   *  are hiding the rest". */
  thisTabLeft?: number;
  totalLabels?: number;
  onSwitchTab?: () => void;
}

// ── Labels filter state (independent from Events / Files filters) ──

function toLabelFilters(f: LabelsFilterState): LabelFilters {
  // The Verified filter scopes the sort server-side so the cap counts the
  // pool the user is actually looking at. "unverified" is the default;
  // "all" omits the clause.
  const ver = f.verification ?? "unverified";
  return {
    labels: f.labels,
    site_ids: f.site_ids,
    date_from: f.date_from,
    date_to: f.date_to,
    min_confidence: f.min_confidence,
    max_confidence: f.max_confidence,
    min_label_confidence: f.min_label_confidence,
    max_label_confidence: f.max_label_confidence,
    verified: ver === "all" ? undefined : ver === "verified",
  };
}

interface SelectionMajority {
  count: number;
  label: string;
  category: string;
  common_name: string | null;
  scientific_name: string | null;
}

/** Most common label among the selected detections.
 *
 *  Returns null when nothing in the selection carries a label.
 *  Ties resolve to the first label encountered, which is deterministic
 *  given the grid's iteration order. Drives both the Match-majority
 *  action and the label shown on its button, so the two never diverge.
 */
function selectionMajority(
  detections: DetectionSummary[],
  idSet: Set<string>,
): SelectionMajority | null {
  const counts = new Map<string, SelectionMajority>();
  for (const d of detections) {
    if (!idSet.has(d.detection_id) || !d.label) continue;
    const entry = counts.get(d.label);
    if (entry) {
      entry.count += 1;
    } else {
      counts.set(d.label, {
        count: 1,
        label: d.label,
        category: d.category,
        common_name: d.common_name,
        scientific_name: d.scientific_name,
      });
    }
  }
  let mode: SelectionMajority | null = null;
  for (const entry of counts.values()) {
    if (!mode || entry.count > mode.count) mode = entry;
  }
  return mode;
}

// Only suggest narrowing once the dataset is big enough that the wait
// is real and narrowing is genuinely the fix. Below this the tip is
// noise, and it gives bad advice: similarity ordering and suggestions
// both get better with more crops, so we don't want to nudge users on
// small sets toward a smaller, worse-quality view. Site is left out of
// the wording on purpose: a folder run is effectively one site, so
// only species and date narrow it there.
const NARROW_TIP_MIN_TOTAL = 5000;

/**
 * Loading state for the Labels grid. Shows a real progress bar
 * while the subprocess streams `progress` events; falls back to an
 * indeterminate spinner during the brief window before the first
 * event arrives.
 *
 * The subprocess emits three phases (load, sort, neighbors), each
 * with its own 0 → N counter. To avoid the bar resetting to 0%
 * between phases (which reads like a flicker), each phase is mapped
 * to one slice of the overall 0 → 100% bar.
 */
function LabelsLoadingState({
  progress,
}: {
  progress: LabelsProgressEvent | null;
}) {
  const phaseLabel = progress
    ? PHASE_LABELS[progress.phase] ?? progress.phase
    : null;
  const overallPct = progress ? overallProgressPct(progress) : 0;
  const showNarrowTip = (progress?.total ?? 0) >= NARROW_TIP_MIN_TOTAL;

  return (
    <div className="flex flex-col items-center gap-3 w-full max-w-sm mx-auto">
      {progress ? (
        <>
          <Progress value={overallPct} className="h-2" />
          <p className="text-xs text-muted-foreground">
            {phaseLabel} ({overallPct}%)
          </p>
        </>
      ) : (
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      )}
      {showNarrowTip && (
        <p className="text-xs text-muted-foreground">
          Narrow the filters to speed this up.
        </p>
      )}
    </div>
  );
}

const PHASE_LABELS: Record<LabelsProgressEvent["phase"], string> = {
  detections: "Loading detections",
  load: "Loading embeddings",
  sort: "Ordering by similarity",
  neighbors: "Comparing neighbours",
};

// Slice each phase into a chunk of the overall 0-100% bar so the
// per-phase resets don't flicker the bar back to 0%. Order matches
// the subprocess: load → sort → neighbors. Weights are 1/3 each;
// not perfectly proportional to wall-clock time but close enough
// that the bar moves smoothly forward.
const PHASE_RANGES: Record<
  LabelsProgressEvent["phase"],
  { start: number; end: number }
> = {
  // The metadata sorts (event / time) emit only this phase, so it spans
  // the whole bar. The embedding sorts use load → sort → neighbours below.
  detections: { start: 0, end: 100 },
  load: { start: 0, end: 33 },
  sort: { start: 33, end: 66 },
  neighbors: { start: 66, end: 100 },
};

function overallProgressPct(progress: LabelsProgressEvent): number {
  const range = PHASE_RANGES[progress.phase];
  if (!range) return 0;
  const phaseFrac =
    progress.total > 0 ? Math.min(1, progress.done / progress.total) : 0;
  return Math.round(range.start + (range.end - range.start) * phaseFrac);
}

export function LabelsTab({
  projectId,
  classificationModelId,
  onSelectionChange,
  toolbarExtra,
  refreshSignal,
  otherTabLeft = 0,
  thisTabLeft = 0,
  totalLabels = 0,
  onSwitchTab,
}: LabelsTabProps) {
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();

  // ── Own filter state from URL lbl_* params ──────────────────────────
  const lblFilters = useMemo(
    () => lblFiltersFromSearchParams(searchParams),
    [searchParams],
  );

  const setLblFilters = useCallback(
    (next: LabelsFilterState) => {
      setSearchParams(
        (prev) => lblFiltersToSearchParams(next, prev),
        { replace: true },
      );
    },
    [setSearchParams],
  );

  /** Handler for VerifyFilterBar onChange (EventFilterParams shape).
   *
   *  The bar collapses the page default ("unverified" here, via
   *  verificationDefault) to undefined, so undefined means "default"
   *  and any set value ("verified" / "all") is an explicit filter.
   *  Straight passthrough. */
  const handleFilterBarChange = useCallback(
    (fp: EventFilterParams) => {
      setLblFilters(fromFilterBarFilters(fp, lblFilters));
    },
    [lblFilters, setLblFilters],
  );

  // ── Local settings state (persisted to localStorage) ────────────────
  // Shared with the Empties tab; see `labels-settings.ts`.
  const savedSettings = useMemo(() => readLabelsSettings(), []);

  const isLabelSort = (v: unknown): v is LabelSort =>
    v === "similarity" ||
    v === "events" ||
    v === "suggestions";

  // Default to event sort: it always produces a usable grid (similarity-
  // clustered events with embeddings, chronological without) and enables
  // the E->M keyboard flow, so a new user never hits the "needs
  // embeddings" card. A user's own choice persists and wins here; the
  // fallback also catches saved sorts that are no longer offered (older
  // builds had time / confidence modes here).
  const initialSort: LabelSort = isLabelSort(savedSettings.sort)
    ? savedSettings.sort
    : "events";
  const [lblSort, _setLblSort] = useState<LabelSort>(initialSort);
  const setLblSort = useCallback(
    (v: LabelSort) => {
      _setLblSort(v);
      persistLabelsSetting("sort", v);
    },
    [],
  );

  const [tileSize, setTileSize] = useTileSize();
  const { wide, toggle: toggleWide } = useWideModeControls();

  // Verification filter is the bar's "Verified" select; default unverified.
  const verificationFilter: LabelsVerification =
    lblFilters.verification ?? "unverified";


  // Toolbar sheet/popover state (welcome popover only; keyboard and
  // settings are self-contained popovers anchored to their toolbar
  // icons, so they own their own open state).
  const [relabelOpen, setRelabelOpen] = useState(false);
  const [showWelcome, setShowWelcome] = useState(
    () => !localStorage.getItem("addaxai:labelsWelcomeDismissed")
  );
  const handleDismissWelcome = useCallback(() => {
    setShowWelcome(false);
    localStorage.setItem("addaxai:labelsWelcomeDismissed", "1");
  }, []);

  // Explicit sorting flag — avoids isPending getting stuck in Strict Mode
  const [isSorting, setIsSorting] = useState(false);
  // A "background" sort refreshes data after an in-grid action (verify /
  // relabel) and must NOT blank the grid. Every other sort (filter change,
  // sort switch, load more, try again) is blocking: it shows the loading
  // state, like a fresh load, so the reload is never silent.
  const [isBackgroundSort, setIsBackgroundSort] = useState(false);
  // Last sort error (e.g. the max-detections cap). Held so the grid
  // body can show an explicit, persistent error card instead of a
  // toast that fades and leaves a spinner spinning forever.
  const [sortError, setSortError] = useState<string | null>(null);

  // Re-embed state
  const [reEmbedJobId, setReEmbedJobId] = useState<string | null>(null);

  // Results
  const [sortResult, setSortResult] = useState<SortResponse | null>(null);
  // Sort mode that produced the current sortResult. The dividers prop
  // tracks this rather than lblSort so the brief window after a sort
  // switch — where the old result lingers until the new sort lands —
  // does not paint cohort dividers over similarity data (which would
  // collapse everything that shares (label, "", category) into a
  // single "(no label)" cohort).
  const [resultSort, setResultSort] = useState<LabelSort | null>(null);

  // Selection
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const selectionAnchorRef = useRef<string | null>(null);
  // Imperative handle to the grid so the "E" shortcut can scroll the
  // newly selected event into view.
  const cropGridRef = useRef<CropGridHandle>(null);
  // Detection to scroll to after the next render. The post-action advance
  // sets this instead of scrolling inline: the action removes the acted
  // cards, so scrolling has to wait until the grid has re-rendered with
  // the shorter list, or it lands on a stale offset. Consumed by the
  // effect below.
  const pendingScrollRef = useRef<string | null>(null);

  // Undo stack: one entry per label action, holding just the affected
  // detection ids. Undo reverts them to the model's original prediction
  // (the original_* columns), so no pre-state snapshot is needed.
  // In-memory / session-scoped. `undoDepth` mirrors the stack length to
  // drive the button (a ref alone wouldn't re-render).
  const undoStackRef = useRef<string[][]>([]);
  const [undoDepth, setUndoDepth] = useState(0);
  const pushUndo = useCallback((ids: string[]) => {
    if (ids.length === 0) return;
    undoStackRef.current = [...undoStackRef.current, ids];
    setUndoDepth(undoStackRef.current.length);
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedIds(new Set());
    selectionAnchorRef.current = null;
  }, []);

  // Notify the parent when the selection size changes (and once on
  // unmount, so it can restore a hidden step nav).
  useEffect(() => {
    onSelectionChange?.(selectedIds.size);
    return () => {
      onSelectionChange?.(0);
    };
  }, [selectedIds.size, onSelectionChange]);

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

  // Stats query — embedded-detection counts for the missing-embeddings
  // banner and the "no embeddings yet" empty state.
  const { data: stats } = useQuery({
    queryKey: ["labels-stats", projectId],
    queryFn: () => labelsApi.stats(projectId),
    enabled: !!projectId,
  });

  // Streaming progress reported by the subprocess (load → sort → neighbors).
  // Cleared whenever a new sort starts and when results land.
  const [progress, setProgress] = useState<LabelsProgressEvent | null>(
    null,
  );

  // Sort mutation — takes the sort mode as the mutation argument so
  // `onSuccess` can pin the resulting data to it. Otherwise a rapid
  // sort-mode flip would race the in-flight result against the latest
  // `lblSort` state and paint the wrong dividers on the response.
  const sortMutation = useMutation({
    mutationFn: (vars: { sort: LabelSort; background?: boolean }) =>
      labelsApi.sortStream(
        projectId,
        {
          filters: toLabelFilters(lblFilters),
          sort: vars.sort,
        },
        setProgress,
      ),
    onMutate: (vars) => {
      setIsSorting(true);
      setIsBackgroundSort(!!vars.background);
      setProgress(null);
      setSortError(null);
    },
    onSuccess: (data, vars) => {
      setSortResult(data);
      setResultSort(vars.sort);
      clearSelection();
      setIsSorting(false);
      setProgress(null);
      setSortError(null);
    },
    onError: (err: Error) => {
      setSortError(err.message);
      setIsSorting(false);
      setProgress(null);
    },
  });

  // Stable key for filter + sort comparison; drives auto re-sort. The
  // Verified filter is inside toLabelFilters, so switching it re-sorts
  // (the cap counts the current verified pool, server-side).
  const filtersKey = JSON.stringify(toLabelFilters(lblFilters));
  const sortKey = `${filtersKey}|${lblSort}|${refreshSignal ?? 0}`;
  const lastSortKeyRef = useRef<string | null>(null);

  // Debounce the key so a slider drag (or any rapid filter change)
  // fires ONE sort after it settles, not a subprocess spawn per step.
  // Matches the Counts page's debounced filters. The handle, % readout,
  // chips, and URL all update live off the undebounced state.
  const debouncedSortKey = useDebouncedValue(sortKey, FILTER_DEBOUNCE_MS);

  // Event sort needs no embeddings (it orders by capture time / event),
  // so it may run even when embedded_detections is 0. Similarity still
  // requires embeddings.
  const canSort =
    lblSort === "events" ? !!stats : (stats?.embedded_detections ?? 0) > 0;

  // Auto-sort on mount and when filters or sort mode settle.
  useEffect(() => {
    if (canSort && debouncedSortKey !== lastSortKeyRef.current) {
      lastSortKeyRef.current = debouncedSortKey;
      sortMutation.mutate({ sort: lblSort });
    }
  }, [debouncedSortKey, canSort]); // eslint-disable-line react-hooks/exhaustive-deps

  // Flat detection list for selection model
  const allDetections = useMemo((): DetectionSummary[] => {
    let dets: DetectionSummary[] = sortResult?.detections ?? [];

    if (verificationFilter === "unverified") {
      dets = dets.filter((d) => !d.verified);
    } else if (verificationFilter === "verified") {
      dets = dets.filter((d) => d.verified);
    }
    return dets;
  }, [sortResult, verificationFilter]);

  // The grid's visual order, which is what a shift-click range is read
  // from. Also the input to `nextAfterActed`.
  const orderedDetectionIds = useMemo(
    () => allDetections.map((d) => d.detection_id),
    [allDetections],
  );

  // Unfiltered count to detect "all hidden by filters" vs "genuinely empty"
  const totalCount = useMemo(
    () => sortResult?.detections.length ?? 0,
    [sortResult],
  );

  // The subprocess caps a huge selection to the newest slice (a memory
  // guard); `total_matching` is the uncapped pool. When it exceeds what we
  // loaded, we show a notice and let the user reload.
  //
  // Compare against `total_loaded`, never `total_detections`. The
  // suggestions sort returns only cohort members, so `total_detections` is
  // a small subset of a pool that was never capped, and comparing it here
  // fired the notice on every suggestions run.
  const loadedCount = sortResult?.total_loaded ?? 0;
  const totalMatching = sortResult?.total_matching ?? 0;
  const isCapped = totalMatching > loadedCount;
  // Name the actual cap: similarity / suggestions walk embeddings (cap
  // 20k), event sort renders detections (cap 50k). Keyed off the sort
  // that produced the current result, not the dropdown.
  const capNoun =
    resultSort === "similarity" || resultSort === "suggestions"
      ? "embedded detections"
      : "detections";
  // Re-run the current sort. It's a reload, not pagination: the Verified
  // filter scopes the sort server-side, so once you've verified some crops
  // they drop out of the pool and the newest ones that didn't fit come in.
  const reload = useCallback(
    () => sortMutation.mutate({ sort: lblSort }),
    [sortMutation, lblSort],
  );

  // Latest grid order, read by advanceSelectionAfter so it can stay stable.
  const allDetectionsRef = useRef<DetectionSummary[]>(allDetections);
  allDetectionsRef.current = allDetections;

  // After an action consumes the current selection, advance to the next card
  // where you were — the one that slides into the freed slot — instead of
  // clearing, so repeated verify / relabel / etc. need no mouse re-click. This
  // only moves the highlight; it never acts on the new card. Reads the
  // pre-action order (state updates are async) and never picks an acted card,
  // so the chosen one survives the strip / verified-filter. Selecting on the
  // tail falls back to the card just before the acted block.
  const advanceSelectionAfter = useCallback(
    (actedIds: string[]) => {
      const order = allDetectionsRef.current.map((d) => d.detection_id);
      const next = nextAfterActed(order, actedIds);
      if (next === null) {
        clearSelection();
        return;
      }
      selectionAnchorRef.current = next;
      setSelectedIds(new Set([next]));
      // Keep the advanced card in view so a keyboard-only pass never
      // needs the mouse. Deferred to the post-render effect because the
      // acted cards are being removed this same tick; "auto" then only
      // scrolls when it would otherwise be off-screen.
      pendingScrollRef.current = next;
    },
    [clearSelection],
  );

  // Runs the queued advance-scroll after the grid has re-rendered with the
  // post-action rows. Child (grid) effects run before this parent effect,
  // so the virtualizer has re-measured and the target offset is correct.
  useEffect(() => {
    const id = pendingScrollRef.current;
    if (id == null) return;
    pendingScrollRef.current = null;
    cropGridRef.current?.scrollToDetection(id, "auto");
  });

  // Click / shift-range / cmd-toggle live in `grid-selection.ts`, shared
  // with the Empties grid so the two feel identical.
  const handleSelect = useCallback(
    (detectionId: string, e: React.MouseEvent) => {
      setSelectedIds((prev) => {
        const result = selectOnClick(
          orderedDetectionIds,
          selectionAnchorRef.current,
          detectionId,
          e,
          prev,
        );
        selectionAnchorRef.current = result.anchor;
        return result.ids;
      });
    },
    [orderedDetectionIds]
  );

  // The large view navigates a list pinned when it opens. The grid list
  // drops a crop the moment it is verified (default filter "Unverified",
  // applied server-side on the refetch), so navigating the live list made
  // "previous" skip the crop just verified and never return to the other
  // animal in the same photo (Grant Hiebert, 2026-08-25). The pin holds the
  // crops themselves, patched by every action below, so they stay
  // reachable and current until the view closes.
  const detailNavRef = useRef<{
    ids: string[];
    byId: Map<string, DetectionSummary>;
  } | null>(null);

  // A refetched grid row (canonical names after a relabel) wins over the
  // pinned copy; a crop the refetch no longer returns keeps its pinned,
  // patched copy.
  const detailNavList = useMemo((): DetectionSummary[] => {
    const pinned = detailNavRef.current;
    if (!detailDetection || !pinned) return [];
    const live = new Map(
      (sortResult?.detections ?? []).map((d) => [d.detection_id, d]),
    );
    return pinned.ids.flatMap((id) => {
      const d = live.get(id) ?? pinned.byId.get(id);
      return d ? [d] : [];
    });
  }, [detailDetection, sortResult]);

  const handleCardClick = useCallback((detection: DetectionSummary) => {
    const grid = allDetectionsRef.current;
    detailNavRef.current = {
      ids: grid.map((d) => d.detection_id),
      byId: new Map(grid.map((d) => [d.detection_id, d])),
    };
    setDetailDetection(detection);
  }, []);

  const handleActionComplete = useCallback(() => {
    // Re-run the current sort to refresh data. Background: this fires after
    // an in-grid verify/relabel, so it must reconcile without blanking the
    // grid the user is working in.
    sortMutation.mutate({ sort: lblSort, background: true });
    invalidateLabelQueries(queryClient);
    // Cohort counts feed the toolbar pill; any relabel / verify path
    // can change which detections still belong in a cohort. Invalidate
    // here so the pill catches up after every bulk action, not just
    // the divider's Accept button.
    queryClient.invalidateQueries({ queryKey: ["cohorts", projectId] });
    // Cascade to the Media / Events views (File.verified rollup) and the
    // verified-progress pill — see applyDetectionAction.
    queryClient.invalidateQueries({ queryKey: ["events"] });
  }, [lblSort, queryClient, projectId]); // eslint-disable-line react-hooks/exhaustive-deps

  /** Re-embed (Embed now / Process now) finished: the grid renders from
   *  the sort mutation, so query invalidation alone can't show the newly
   *  embedded detections. Refresh the embedding-coverage stats (note the
   *  labels-stats key — invalidateProjectData uses a different one) and
   *  the unprocessed-tail count, then re-run the sort so the new crops
   *  appear without a hard refresh. */
  const handleReEmbedComplete = useCallback(() => {
    invalidateProjectData(queryClient, projectId);
    queryClient.invalidateQueries({ queryKey: ["labels-stats", projectId] });
    queryClient.invalidateQueries({
      queryKey: ["labels-unprocessed", projectId],
    });
    sortMutation.mutate({ sort: lblSort });
  }, [lblSort, queryClient, projectId]); // eslint-disable-line react-hooks/exhaustive-deps

  /** Patch detections in local state without refetching. */
  const patchLocalDetections = useCallback(
    (patchFn: (d: DetectionSummary) => DetectionSummary) => {
      if (sortResult) {
        setSortResult({
          ...sortResult,
          detections: sortResult.detections.map(patchFn),
        });
      }
      const pinned = detailNavRef.current;
      if (pinned) {
        for (const [id, d] of pinned.byId) pinned.byId.set(id, patchFn(d));
      }
      // Keep the detail modal in sync
      setDetailDetection((prev) => (prev ? patchFn(prev) : prev));
    },
    [sortResult]
  );

  /** Apply a single-card or bulk action to local state.
   *
   *  In similarity / metadata sorts, the affected detections stay
   *  visible with their new fields (patch in place).
   *
   *  In suggestions mode, the affected detections have been reviewed
   *  and leave the grid. Mutating in place would change CropGrid's
   *  cohortKey (label + neighbor_top_label + category) for that
   *  detection, tear it out of its cohort row, and leave a phantom
   *  singleton inside an otherwise-intact cohort. Strip the ids
   *  instead, matching what the cohort divider's Accept button does.
   *
   *  Also invalidates the cohorts query so the toolbar pill catches up
   *  regardless of which action path triggered the change.
   */
  const applyDetectionAction = useCallback(
    (ids: string[], patch: (d: DetectionSummary) => DetectionSummary) => {
      const idSet = new Set(ids);
      if (lblSort === "suggestions") {
        setSortResult((prev) =>
          prev
            ? {
                ...prev,
                detections: prev.detections.filter(
                  (d) => !idSet.has(d.detection_id),
                ),
              }
            : prev,
        );
        setDetailDetection((prev) =>
          prev && idSet.has(prev.detection_id) ? null : prev,
        );
      } else {
        patchLocalDetections((d) =>
          idSet.has(d.detection_id) ? patch(d) : d,
        );
      }
      queryClient.invalidateQueries({ queryKey: ["cohorts", projectId] });
      // The page's progress bar counts photos, and verifying the last
      // box on one flips File.verified server-side. Its key does not
      // start with "event", so the predicate below never reaches it: it
      // needs saying explicitly, or the bar only moves on a reload.
      queryClient.invalidateQueries({
        queryKey: ["labels-progress", projectId],
      });
      // Relabelling rebuilds the event's observations server-side (and
      // cascades up to File.verified / event verification). Refresh every
      // event-family query so the Counts page reflects it: the event list
      // (["events", ...]), the single-event modal (["event", id]), the
      // count badges (["event-count", ...]) and the filter options. A
      // plain ["events"] prefix would miss the singular ["event", ...] and
      // ["event-count", ...] keys, which is why the Counts view and its
      // event modal used to stay stale after a relabel.
      queryClient.invalidateQueries({
        predicate: (q) => {
          const key = q.queryKey[0];
          return typeof key === "string" && key.startsWith("event");
        },
      });
    },
    [lblSort, patchLocalDetections, projectId, queryClient],
  );

  // Same optimistic patch as applyDetectionAction, but also records the
  // action on the undo stack. Every label action (verify / relabel /
  // mark) goes through this; only handleUndo itself uses the plain
  // applyDetectionAction, so undoing never re-stacks.
  const applyUndoableAction = useCallback(
    (ids: string[], patch: (d: DetectionSummary) => DetectionSummary) => {
      applyDetectionAction(ids, patch);
      pushUndo(ids);
    },
    [applyDetectionAction, pushUndo],
  );

  // Undo the most recent label action: revert its detections to the
  // model's original prediction and pop the stack. Works with no active
  // selection (it acts on history, not the current selection).
  const handleUndo = useCallback(async () => {
    const stack = undoStackRef.current;
    if (stack.length === 0) return;
    const ids = stack[stack.length - 1];
    try {
      const { reverted } = await detectionsApi.bulkRevertToOriginal(ids);
      const byId = new Map(reverted.map((r) => [r.detection_id, r]));
      applyDetectionAction(ids, (d) => {
        const r = byId.get(d.detection_id);
        if (!r) return d;
        return {
          ...d,
          label: r.label,
          category: r.category,
          label_confidence: r.label_confidence,
          label_taxonomy_id: r.label_taxonomy_id,
          scientific_name: r.scientific_name,
          common_name: r.common_name,
          verified: r.verified,
          neighbor_top_label: null,
          neighbor_top_scientific_name: null,
        };
      });
      undoStackRef.current = stack.slice(0, -1);
      setUndoDepth(undoStackRef.current.length);
      // Bring the reverted crops back into view so the user can re-act.
      // No success toast: the crops reappearing (unverified) is the
      // confirmation, and the verify workflow is toast-fatigued already.
      setSelectedIds(new Set(ids));
    } catch (err: unknown) {
      toast.error(err instanceof Error ? err.message : "Undo failed");
    }
  }, [applyDetectionAction]);

  const handleBulkRelabel = useCallback(
    (ids: string[], label: string | null, category: string, displayName: string) => {
      applyUndoableAction(ids, (d) => ({
        ...d,
        label,
        category,
        // Set BOTH names: the display prefers common_name, so leaving the
        // old one in place would keep showing the previous label until a
        // refetch.
        common_name: displayName,
        scientific_name: displayName,
        label_taxonomy_id: null,
        neighbor_top_label: null,
        neighbor_top_scientific_name: null,
        verified: true,
      }));
      invalidateLabelQueries(queryClient);
      advanceSelectionAfter(ids);
    },
    [applyDetectionAction, advanceSelectionAfter, queryClient]
  );

  // Bulk-relabel a cohort straight from the divider button — no
  // confirm modal. The cohort header already shows the count, the
  // current label, the suggested label, and a tooltip spells out the
  // "+ verify" consequence on hover; a second prompt was just
  // restating the same information.
  //
  // Optimistically strips the relabelled detections from the local
  // sortResult (so the cohort card vanishes immediately) and
  // invalidates the cohort + label-tree caches so the toolbar pill's
  // count catches up.
  const relabelCohort = useCallback(
    async (cohort: CohortItem) => {
      try {
        await detectionsApi.bulkRelabel(
          cohort.detection_ids,
          cohort.suggested_label,
          cohort.category ?? undefined,
        );
        const idSet = new Set(cohort.detection_ids);
        setSortResult((prev) =>
          prev
            ? {
                ...prev,
                detections: prev.detections.filter(
                  (d) => !idSet.has(d.detection_id),
                ),
              }
            : prev,
        );
        queryClient.invalidateQueries({ queryKey: ["cohorts", projectId] });
        invalidateLabelQueries(queryClient);
        // Refresh every event-family query (list, single-event modal,
        // count badges, filter options) so the Counts page reflects the
        // rebuilt observations, not just the ["events"] list prefix.
        queryClient.invalidateQueries({
          predicate: (q) => {
            const key = q.queryKey[0];
            return typeof key === "string" && key.startsWith("event");
          },
        });
        toast.success(
          `Relabelled ${cohort.count} label${
            cohort.count === 1 ? "" : "s"
          } to ${cohort.suggested_scientific_name || cohort.suggested_label}.`,
        );
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Relabel failed");
      }
    },
    [projectId, queryClient],
  );

  // Dismiss a whole cohort straight from the divider button — no confirm
  // modal, mirroring relabelCohort. Dismiss is non-destructive: it sets
  // `suggestion_dismissed` so the cohort stops appearing in suggestions,
  // but leaves every crop's label and verified state untouched. The crops
  // remain in the normal sorts for later relabelling.
  //
  // Optimistically strips the dismissed detections from the local
  // sortResult (so the cohort card vanishes immediately) and invalidates
  // the cohorts query so the toolbar pill's count catches up. Only the
  // cohorts cache is touched — nothing about labels or verification
  // changed. A toast offers Undo, which clears the flag again.
  const dismissCohort = useCallback(
    async (cohort: CohortItem) => {
      try {
        await detectionsApi.bulkDismiss(cohort.detection_ids, true);
        const idSet = new Set(cohort.detection_ids);
        setSortResult((prev) =>
          prev
            ? {
                ...prev,
                detections: prev.detections.filter(
                  (d) => !idSet.has(d.detection_id),
                ),
              }
            : prev,
        );
        queryClient.invalidateQueries({ queryKey: ["cohorts", projectId] });
        toast.success(
          `Dismissed ${cohort.count} suggestion${
            cohort.count === 1 ? "" : "s"
          }.`,
          {
            action: {
              label: "Undo",
              onClick: () => {
                detectionsApi
                  .bulkDismiss(cohort.detection_ids, false)
                  .then(() => {
                    queryClient.invalidateQueries({
                      queryKey: ["cohorts", projectId],
                    });
                  })
                  .catch((err: Error) => toast.error(err.message));
              },
            },
          },
        );
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Dismiss failed");
      }
    },
    [projectId, queryClient],
  );

  // The sort the user was on before entering suggestions. Exiting
  // returns them there instead of always dropping them into similarity,
  // which they may not use (e.g. they came from "By event").
  const preSuggestionsSortRef = useRef<LabelSort>("similarity");
  const exitSuggestionsMode = useCallback(
    () => setLblSort(preSuggestionsSortRef.current),
    [setLblSort],
  );

  const handleMarkFalse = useCallback(
    (ids: string[]) => {
      detectionsApi
        .bulkRelabel(ids, "false detection", undefined)
        .then(() => {
          applyUndoableAction(ids, (d) => ({
            ...d,
            label: "false detection",
            common_name: "False detection",
            scientific_name: "False detection",
            label_taxonomy_id: null,
            neighbor_top_label: null,
            neighbor_top_scientific_name: null,
            verified: true,
          }));
          advanceSelectionAfter(ids);
          invalidateLabelQueries(queryClient);
        })
        .catch((err: Error) => toast.error(err.message));
    },
    [applyDetectionAction, advanceSelectionAfter, queryClient]
  );

  const handleBulkMarkFalse = useCallback(
    (ids: string[]) => {
      applyUndoableAction(ids, (d) => ({
        ...d,
        label: "false detection",
        common_name: "False detection",
        scientific_name: "False detection",
        label_taxonomy_id: null,
        neighbor_top_label: null,
        neighbor_top_scientific_name: null,
        verified: true,
      }));
      invalidateLabelQueries(queryClient);
      advanceSelectionAfter(ids);
    },
    [applyDetectionAction, advanceSelectionAfter, queryClient]
  );

  // "Unknown" mirrors mark-false but the label is a real observation
  // (kept in counts/exports) and the category is left untouched. The
  // server auto-creates the "unknown" custom taxonomy row and resolves
  // both names to "Unknown"; the optimistic patch pre-fills them so the
  // tile updates immediately. Grid-shortcut variant owns the API call.
  const handleMarkUnknown = useCallback(
    (ids: string[]) => {
      detectionsApi
        .bulkRelabel(ids, "unknown", undefined)
        .then(() => {
          applyUndoableAction(ids, (d) => ({
            ...d,
            label: "unknown",
            common_name: "Unknown",
            scientific_name: "Unknown",
            label_taxonomy_id: null,
            neighbor_top_label: null,
            neighbor_top_scientific_name: null,
            verified: true,
          }));
          advanceSelectionAfter(ids);
          invalidateLabelQueries(queryClient);
        })
        .catch((err: Error) => toast.error(err.message));
    },
    [applyDetectionAction, advanceSelectionAfter, queryClient]
  );

  // Patch-only variant: the BulkActionBar already ran the API call.
  const handleBulkMarkUnknown = useCallback(
    (ids: string[]) => {
      applyUndoableAction(ids, (d) => ({
        ...d,
        label: "unknown",
        common_name: "Unknown",
        scientific_name: "Unknown",
        label_taxonomy_id: null,
        neighbor_top_label: null,
        neighbor_top_scientific_name: null,
        verified: true,
      }));
      invalidateLabelQueries(queryClient);
      advanceSelectionAfter(ids);
    },
    [applyDetectionAction, advanceSelectionAfter, queryClient]
  );

  const handleBulkVerify = useCallback(
    (ids: string[]) => {
      applyUndoableAction(ids, (d) => ({ ...d, verified: true }));
      advanceSelectionAfter(ids);
    },
    [applyDetectionAction, advanceSelectionAfter]
  );

  /** Relabel the selection to its most common label and verify.
   *
   *  Useful pattern in similarity-sorted grids: a long run of one
   *  species (e.g. 200 turkeys) is interrupted by a few stragglers
   *  predicted at the broader rank (e.g. "Aves"). Plain Verify keeps
   *  the stragglers wrong; the relabel picker is one tap too many for
   *  a self-evident majority. This snaps everyone to the mode in one
   *  click / keystroke and verifies.
   *
   *  Ties: first label encountered wins (deterministic given grid
   *  iteration order). No-ops with a toast if nothing in the selection
   *  carries a label.
   */
  const handleMatchMajority = useCallback(
    (ids: string[]) => {
      if (ids.length === 0) return;
      const idSet = new Set(ids);
      const mode = selectionMajority(allDetections, idSet);
      if (!mode) {
        toast.info("No labels in selection to apply");
        return;
      }
      const { label: modeLabel, category: modeCategory } = mode;
      detectionsApi
        .bulkRelabel(ids, modeLabel, modeCategory)
        .then(() => {
          applyUndoableAction(ids, (d) => ({
            ...d,
            label: modeLabel,
            category: modeCategory,
            common_name: mode.common_name,
            scientific_name: mode.scientific_name,
            label_taxonomy_id: null,
            neighbor_top_label: null,
            neighbor_top_scientific_name: null,
            verified: true,
          }));
          advanceSelectionAfter(ids);
          // No success toast: this fires on every E->M in a run of
          // hundreds of events. The grid updating (crops verified /
          // relabelled) is the confirmation; errors still toast.
        })
        .catch((err: Error) => toast.error(err.message));
    },
    [allDetections, applyDetectionAction, advanceSelectionAfter],
  );

  // Keyboard shortcuts
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      // Undo: Cmd+Z on macOS, Ctrl+Z on Windows/Linux (the standard
      // everywhere). Shift is excluded to leave room for a future redo.
      if (
        (e.metaKey || e.ctrlKey) &&
        !e.shiftKey &&
        (e.key === "z" || e.key === "Z")
      ) {
        e.preventDefault();
        handleUndo();
        return;
      }

      if (e.key === "Escape") {
        if (detailDetection) {
          setDetailDetection(null);
        } else if (selectedIds.size > 0) {
          clearSelection();
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
            .then(() => {
              handleBulkVerify(ids);
            });
        });
        return;
      }

      if ((e.key === "r" || e.key === "R") && !e.ctrlKey && !e.metaKey && selectedIds.size > 0) {
        e.preventDefault();
        setRelabelOpen((prev) => !prev);
        return;
      }

      if ((e.key === "x" || e.key === "X") && !e.ctrlKey && !e.metaKey && selectedIds.size > 0) {
        e.preventDefault();
        handleMarkFalse(Array.from(selectedIds));
        return;
      }

      if ((e.key === "u" || e.key === "U") && !e.ctrlKey && !e.metaKey && selectedIds.size > 0) {
        e.preventDefault();
        handleMarkUnknown(Array.from(selectedIds));
        return;
      }

      if ((e.key === "m" || e.key === "M") && !e.ctrlKey && !e.metaKey && selectedIds.size > 0) {
        e.preventDefault();
        handleMatchMajority(Array.from(selectedIds));
        return;
      }

      // "E" selects the first event that still needs work, so the bulk
      // shortcuts above (Enter, M, R, X, 1-5) can act on the whole event.
      // Only meaningful in "By event" sort, where each event's crops are
      // contiguous; a no-op in other sorts.
      if (
        (e.key === "e" || e.key === "E") &&
        !e.ctrlKey &&
        !e.metaKey &&
        resultSort === "events"
      ) {
        e.preventDefault();
        const ids = firstUnverifiedEventDetectionIds(allDetections);
        if (ids.length > 0) {
          setSelectedIds(new Set(ids));
          // Bring the newly selected event into view (it is usually
          // below the fold after verifying the ones above it).
          cropGridRef.current?.scrollToDetection(ids[0]);
        }
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
          applyUndoableAction(ids, (d) => ({
            ...d,
            label: label.label ?? label.category,
            category: label.category,
            common_name: label.displayName,
            scientific_name: label.displayName,
            label_taxonomy_id: null,
            neighbor_top_label: null,
            neighbor_top_scientific_name: null,
            verified: true,
          }));
          advanceSelectionAfter(ids);
        });
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedIds, detailDetection, allDetections, resultSort, handleActionComplete, shortcutLabels, applyUndoableAction, handleMarkFalse, handleMarkUnknown, handleMatchMajority, handleBulkVerify, advanceSelectionAfter, handleUndo]);

  // Click outside grid to deselect
  useEffect(() => {
    if (selectedIds.size === 0) return;
    function handleClick(e: MouseEvent) {
      // A modal layer is open, so nothing behind it is a grid background
      // click. Its overlay is not part of the grid, and a drag that
      // starts inside the dialog and ends on the overlay fires `click`
      // on their common ancestor, which is <body>. Both reached
      // clearSelection below, which unmounts the BulkActionBar and the
      // add-custom-label slideout living inside it: the sheet vanished
      // mid-edit with no exit animation and no onOpenChange. Selecting
      // text right to left in the sheet hit this constantly, because
      // the sheet's left edge is the only boundary a drag can cross and
      // it sits ~25px from the fields.
      if (document.querySelector("[role='dialog'],[role='alertdialog']")) return;
      const el = e.target as HTMLElement;
      if (el.closest("[data-crop-card], button, a, input, select, [role='menu'], [role='dialog'], [data-radix-popper-content-wrapper]")) return;
      clearSelection();
    }
    document.addEventListener("click", handleClick);
    return () => document.removeEventListener("click", handleClick);
  }, [selectedIds.size > 0]); // eslint-disable-line react-hooks/exhaustive-deps

  // Verified-detections progress for the toolbar pill.
  // Progress over the whole dataset (unfiltered project stats), not the
  // currently-loaded / filtered detections, so a narrowed view does
  // not read as 100% verified. Same source as the Events and Media
  // pills, so all three views report the same number.
  // Counted in photos, shared with the Empties tab. The two tabs show
  // different units but are one job, and a bar that jumped when you
  // switched tabs would suggest otherwise.
  const pageProgress = useLabelsProgress(projectId, lblFilters);

  // Majority label of the current selection, shown on the Match-majority
  // button so the action is previewable ("Set to Corvus") instead
  // of a blind relabel. Null when the selection carries no labels — the
  // button hides in that case.
  const majorityLabel = useMemo(() => {
    if (selectedIds.size === 0) return null;
    const mode = selectionMajority(allDetections, selectedIds);
    return mode ? resolveSpeciesName(mode) : null;
  }, [selectedIds, allDetections]);

  const hasResults = sortResult !== null;
  // Show the loading view when a sort is running AND we have nothing
  // Show the loading state (no grid, just the progress bar) for every
  // blocking sort — first load, sort switch, filter change, load more,
  // try again — so a reload is never silent. Only a background sort (the
  // post-action reconcile in handleActionComplete) keeps the current grid
  // in place, so verifying doesn't blank the grid you're working in.
  const isLoading = isSorting && !isBackgroundSort;

  const handleEmbedNow = async () => {
    try {
      const { job_id } = await projectsApi.reEmbed(projectId);
      setReEmbedJobId(job_id);
    } catch (err: unknown) {
      toast.error(err instanceof Error ? err.message : "Failed to start embedding");
    }
  };

  // ── Unprocessed low-confidence tail ─────────────────────────────
  // MegaDetector runs untresholded; detections below the run's
  // classification gate were deliberately never embedded, so they
  // cannot appear in this grid. When the user's range slider digs
  // below the gate, count what exists there and offer the backfill.
  // Purely data-driven (embedding-existence, not settings), so it is
  // also correct for projects whose deployments ran under different
  // gates.
  const classificationGate =
    project?.classification_gate ?? DEFAULT_CLASSIFICATION_GATE;
  const detectionFloorValue = project?.counting_threshold ?? 0;
  const effectiveFloor = Math.min(
    detectionFloorValue,
    lblFilters.min_confidence ?? detectionFloorValue,
  );
  const unprocessedRangeMax = Math.min(
    classificationGate,
    lblFilters.max_confidence ?? 1,
  );
  const { data: unprocessed } = useQuery({
    queryKey: [
      "labels-unprocessed",
      projectId,
      effectiveFloor,
      unprocessedRangeMax,
    ],
    queryFn: () =>
      labelsApi.unprocessedCount(
        projectId,
        Math.max(effectiveFloor, MD_OUTPUT_CONFIDENCE_THRESHOLD),
        unprocessedRangeMax,
      ),
    enabled:
      !!project?.embedding_model_id &&
      effectiveFloor < classificationGate,
  });
  const unprocessedCount = unprocessed?.count ?? 0;

  // Similarity sort with no embeddings: rendered as a content-area
  // branch below (NOT a full early return), so the toolbar with the
  // sort dropdown stays visible and the user can switch to event sort.
  const similarityNeedsEmbeddings =
    !!stats && stats.embedded_detections === 0 && lblSort !== "events";

  const handleProcessUnprocessed = async () => {
    try {
      const { job_id } = await projectsApi.reEmbed(projectId, {
        min_confidence: Math.max(
          effectiveFloor,
          MD_OUTPUT_CONFIDENCE_THRESHOLD,
        ),
      });
      setReEmbedJobId(job_id);
    } catch (err: unknown) {
      toast.error(
        err instanceof Error ? err.message : "Failed to start processing",
      );
    }
  };

  return (
    <div className="space-y-4">
      <VerifyFilterBar
        filters={toFilterBarFilters(lblFilters)}
        onChange={handleFilterBarChange}
        projectId={projectId}
        classificationModelId={classificationModelId}
        detectionFloor={project?.counting_threshold ?? 0}
        countBy="detection"
        showLikedFlaggedEmpty={false}
        confidenceFloorMode="open"
        verificationDefault="unverified"
      />

      <ReEmbedModal
        open={!!reEmbedJobId}
        onOpenChange={(open) => { if (!open) setReEmbedJobId(null); }}
        jobId={reEmbedJobId}
        onComplete={handleReEmbedComplete}
        onError={() => invalidateProjectData(queryClient, projectId)}
      />

      <VerifyToolbar>
        {/* Left: the controls that change what happens (analysis
            settings, sort, suggestions). Right: meta icons + progress,
            the glance-and-forget zone. */}
        {toolbarExtra}
        {/* Hide the dropdown in suggestions mode: it's a focused review
            workflow with its own entry / exit via the pill below. The
            pill itself is rendered in any sort mode because the count
            signal is still useful when the user is browsing normally. */}
        {lblSort !== "suggestions" && (
          <SortSelector
            sort={lblSort}
            seed={null}
            availableSorts={LABELS_SORT_MODES}
            onChange={(next) => {
              if (isLabelSort(next)) setLblSort(next);
            }}
          />
        )}
        <SuggestionsToolbarPill
          projectId={projectId}
          isActive={lblSort === "suggestions"}
          onEnter={() => {
            if (lblSort !== "suggestions") preSuggestionsSortRef.current = lblSort;
            setLblSort("suggestions");
          }}
          onExit={exitSuggestionsMode}
        />
        <div className="ml-auto flex items-center gap-1">
          <VerifyToolbarIcon
            icon={wide ? Minimize2 : Maximize2}
            title={wide ? "Exit full width" : "Full width"}
            onClick={toggleWide}
            active={wide}
          />
          <VerifyGuideLink step="labels" />
          <LabelsKeyboardPopover
            shortcuts={CROPS_SHORTCUTS}
            footer="After an action the next detection is selected, so you can keep going."
            labelSlots={{
              shortcutLabels,
              onShortcutLabelsChange: updateShortcutLabels,
              labelOptions,
              labelOptionsLoading,
              projectId,
            }}
          />
          <LabelsSettings
            tileSize={tileSize}
            onTileSizeChange={setTileSize}
          />
          {sortResult && (
            <div className="ml-2">
              <VerifyProgressPill
                pct={pageProgress.pct}
                label="verified"
                title={pageProgress.title}
              />
            </div>
          )}
        </div>
      </VerifyToolbar>

      {/* Warning when embeddings are PARTIAL: the grid shows the
          embedded ones and this offers to fill the rest. Requires
          embedded > 0 so it never overlaps the zero-embedding empty
          card below, and so "Embed now" is only offered when a working
          embedding model exists. Suppressed in event mode (shows all). */}
      {lblSort !== "events" &&
        stats &&
        stats.embedded_detections > 0 &&
        stats.missing_embeddings > 0 && (
        <Callout
          variant="warning"
          title={`${stats.missing_embeddings} detection${
            stats.missing_embeddings !== 1 ? "s are" : " is"
          } not shown`}
          action={
            <Button variant="outline" size="sm" className="shrink-0" onClick={handleEmbedNow}>
              Embed now
            </Button>
          }
        >
          Sort by similarity only shows detections that have an embedding. A
          box you draw yourself does not come with one, and neither does
          anything analysed while embedding was switched off in settings.
          Switch to Sort by event to see all of them, or click "Embed now" to
          embed them.
        </Callout>
      )}

      {/* Unprocessed low-confidence tail in the selected range. Same as
          above: only meaningful for Sort by similarity. */}
      {lblSort !== "events" && unprocessedCount > 0 && (
        <Callout
          variant="info"
          title={`${unprocessedCount.toLocaleString()} more detection${
            unprocessedCount !== 1 ? "s" : ""
          } available in this confidence range`}
          action={
            <Button
              variant="outline"
              size="sm"
              className="shrink-0"
              onClick={handleProcessUnprocessed}
            >
              Process now
            </Button>
          }
        >
          These detections sit below the confidence at which this
          analysis identified species, so they were never processed for
          review. Process them to show them here as unlabeled animals.
          Your data exports include them either way.
        </Callout>
      )}

      {sortError ? (
        // Persistent error card. The common case is the max-detections
        // cap: the subprocess refuses to load a result set larger than
        // the user's limit. Without this the body would fall through to
        // the loading branch and spin forever after the error.
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16 text-center">
            <AlertTriangle className="h-10 w-10 text-amber-500 mb-4" />
            <p className="text-lg font-medium text-muted-foreground">
              Could not load this view
            </p>
            <p className="text-sm text-muted-foreground mt-1 max-w-md">
              {sortError}
            </p>
            <Button
              variant="outline"
              size="sm"
              className="mt-4"
              onClick={() => sortMutation.mutate({ sort: lblSort })}
            >
              Try again
            </Button>
          </CardContent>
        </Card>
      ) : similarityNeedsEmbeddings ? (
        // No embeddings: similarity has nothing to walk. Event sort
        // works without them, so the sort dropdown stays visible above.
        // Simple rule: a model is set -> offer to embed; no model ->
        // nudge to where you pick one (settings for a project, the setup
        // step for a folder run), with no action.
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16 text-center">
            <Layers className="h-12 w-12 text-muted-foreground/50 mb-4" />
            <p className="text-lg font-medium text-muted-foreground">
              Sort by similarity needs embeddings
            </p>
            <p className="text-sm text-muted-foreground mt-1 max-w-md">
              {project?.embedding_model_id
                ? "Switch to Sort by event to review these detections, or embed them to enable similarity sorting."
                : project?.mode === "folder_run"
                  ? "Switch to Sort by event to review these detections, or run an analysis with an embedding model to enable similarity sorting."
                  : "Switch to Sort by event to review these detections, or pick an embedding model in the project settings to enable similarity sorting."}
            </p>
            {project?.embedding_model_id && (
              <Button
                variant="outline"
                size="sm"
                className="mt-4"
                onClick={handleEmbedNow}
              >
                Embed now
              </Button>
            )}
          </CardContent>
        </Card>
      ) : isLoading || !hasResults ? (
        // Sort auto-runs, so !hasResults means the mutation has not
        // fired yet (e.g. stats still loading). Either way, show the
        // loading state.
        <div className="flex items-center justify-center h-64">
          <LabelsLoadingState progress={progress} />
        </div>
      ) : lblSort === "suggestions" && allDetections.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16 text-center">
            <Check className="h-8 w-8 mb-3 text-primary" />
            <p className="text-lg font-medium">All suggestions reviewed</p>
            <p className="text-sm text-muted-foreground mt-1 max-w-md">
              There are no more label suggestions to check right now. Go
              back to your labels to keep verifying.
            </p>
            <Button
              type="button"
              className="mt-4"
              onClick={exitSuggestionsMode}
            >
              Exit suggestions
            </Button>
          </CardContent>
        </Card>
      ) : allDetections.length === 0 && totalCount > 0 && isCapped ? (
        // Its own case: the grid is empty because the result was
        // truncated, not because the work is done, and the fix is a
        // reload rather than a filter change.
        <div className="flex flex-col items-center justify-center py-20 text-center text-muted-foreground">
          <Check className="h-8 w-8 mb-3 text-muted-foreground/60" />
          <p className="text-sm">
            You&apos;ve verified all {loadedCount.toLocaleString()} loaded
            labels.
          </p>
          <p className="text-xs mt-1">
            {(totalMatching - loadedCount).toLocaleString()} more aren&apos;t
            loaded yet.
          </p>
          <Button variant="outline" size="sm" className="mt-4" onClick={reload}>
            Reload
          </Button>
        </div>
      ) : allDetections.length === 0 ? (
        <GridEmptyState
          thisTabLeft={thisTabLeft}
          otherTabLeft={otherTabLeft}
          totalLabels={totalLabels}
          viewFinished={totalCount > 0}
          viewCount={totalCount}
          tabHasNothing={totalLabels > 0 && thisTabLeft === 0 && totalCount === 0}
          noun="detections"
          otherNoun="empty files"
          otherTabName="Empties"
          onClearFilters={() => setLblFilters({})}
          onSwitchTab={onSwitchTab}
        />
      ) : (
        <div style={{ paddingBottom: selectedIds.size > 0 ? 80 : 0 }}>
          {isCapped && (
            <Callout
              variant="info"
              size="compact"
              className="mb-3"
              action={
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7"
                  onClick={reload}
                >
                  Reload
                </Button>
              }
            >
              {resultSort === "suggestions" ? (
                <>
                  Suggestions were found in the newest{" "}
                  {loadedCount.toLocaleString()} of{" "}
                  {totalMatching.toLocaleString()} {capNoun}, capped to stay
                  responsive. Verify some and reload to search the rest, or
                  narrow the filters.
                </>
              ) : (
                <>
                  Showing the newest {loadedCount.toLocaleString()} of{" "}
                  {totalMatching.toLocaleString()} {capNoun}, capped to stay
                  responsive. Verify some and reload to bring in more, or narrow
                  the filters.
                </>
              )}
            </Callout>
          )}
          <CropGrid
            ref={cropGridRef}
            detections={allDetections}
            selectedIds={selectedIds}
            onSelect={handleSelect}
            onDoubleClick={handleCardClick}
            onBackgroundClick={clearSelection}
            onRelabelCohort={relabelCohort}
            onDismissCohort={dismissCohort}
            onSelectEvent={(ids) => setSelectedIds(new Set(ids))}
            tileSize={tileSize}
            dividers={
              // Tie dividers to the sort that PRODUCED the current
              // result, not the user's dropdown selection. Otherwise the
              // brief window after switching modes, where the old result
              // lingers until the new sort lands, paints the wrong
              // dividers over the old data.
              resultSort === "suggestions"
                ? "cohort"
                : resultSort === "events"
                  ? "event"
                  : "none"
            }
          />
        </div>
      )}

      <BulkActionBar
        selectedIds={selectedIds}
        onDeselectAll={clearSelection}
        labelOptions={labelOptions}
        labelOptionsLoading={labelOptionsLoading}
        onActionComplete={handleActionComplete}
        onRelabel={handleBulkRelabel}
        onVerify={handleBulkVerify}
        onMarkFalse={handleBulkMarkFalse}
        onMarkUnknown={handleBulkMarkUnknown}
        onMatchMajority={handleMatchMajority}
        majorityLabel={majorityLabel}
        projectId={projectId}
        relabelOpen={relabelOpen}
        onRelabelOpenChange={setRelabelOpen}
        onUndo={handleUndo}
        canUndo={undoDepth > 0}
      />

      <LabelsWelcomePopover open={showWelcome} onDismiss={handleDismissWelcome} />
      <DetectionDetailModal
        detection={detailDetection}
        open={!!detailDetection}
        onOpenChange={(open) => {
          if (!open) setDetailDetection(null);
        }}
        onActionComplete={handleActionComplete}
        projectId={projectId}
        labelOptions={labelOptions}
        labelOptionsLoading={labelOptionsLoading}
        onRelabel={(detectionId, label, category) => {
          applyUndoableAction([detectionId], (d) => ({
            ...d,
            label,
            category,
            // Clear BOTH names so the display falls back to the new
            // label; leaving the old common_name would keep showing the
            // previous label until a refetch fills the canonical names.
            common_name: null,
            scientific_name: null,
            label_taxonomy_id: null,
            neighbor_top_label: null,
            neighbor_top_scientific_name: null,
            verified: true,
          }));
        }}
        onMarkFalse={(detectionId) => {
          applyUndoableAction([detectionId], (d) => ({
            ...d,
            label: "false detection",
            common_name: "False detection",
            scientific_name: "False detection",
            label_taxonomy_id: null,
            neighbor_top_label: null,
            neighbor_top_scientific_name: null,
            verified: true,
          }));
        }}
        onMarkUnknown={(detectionId) => {
          applyUndoableAction([detectionId], (d) => ({
            ...d,
            label: "unknown",
            common_name: "Unknown",
            scientific_name: "Unknown",
            label_taxonomy_id: null,
            neighbor_top_label: null,
            neighbor_top_scientific_name: null,
            verified: true,
          }));
        }}
        onVerify={(detectionId, verified = true) => {
          applyUndoableAction([detectionId], (d) => ({ ...d, verified }));
        }}
        position={
          detailDetection
            ? `${detailNavList.findIndex((d) => d.detection_id === detailDetection.detection_id) + 1} / ${detailNavList.length}`
            : undefined
        }
        onNavigate={(direction) => {
          if (!detailDetection) return false;
          const idx = detailNavList.findIndex(
            (d) => d.detection_id === detailDetection.detection_id
          );
          if (idx === -1) return false;

          if (direction === "nextUnverified") {
            // Find next unverified after current index, wrapping around
            for (let i = 1; i <= detailNavList.length; i++) {
              const candidate = detailNavList[(idx + i) % detailNavList.length];
              if (!candidate.verified) {
                setDetailDetection(candidate);
                return true;
              }
            }
            // All verified — close the sheet
            toast.success("All labels verified");
            setDetailDetection(null);
            return false;
          }

          const nextIdx =
            direction === "next"
              ? Math.min(idx + 1, detailNavList.length - 1)
              : Math.max(idx - 1, 0);
          if (nextIdx === idx) return false;
          setDetailDetection(detailNavList[nextIdx]);
          return true;
        }}
      />
    </div>
  );
}


