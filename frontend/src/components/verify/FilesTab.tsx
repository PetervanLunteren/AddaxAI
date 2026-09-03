/**
 * FilesTab - the Files half of the Labels page.
 *
 * One tile per file, with its visible boxes drawn on it. The job here is
 * different from the crop grid's: not "is this label right?" but "is
 * this picture right?". So the unit is the file, the sorts are about
 * where a file sits rather than what it looks like, and the verdict is
 * one: Verify means the boxes you can see are all there is. Weak boxes
 * below the threshold are set aside as false detections, the visible
 * ones are signed off, and a box you draw first is one of them.
 *
 * The Empty select in More filters narrows to the files where nothing
 * passed the threshold (the old Empties tab) or to the files where
 * something did. It rests on "all".
 *
 * Shares its filter state with the crop grid through the `lbl_*` URL
 * params (see `labels-filters.ts`), so switching tabs keeps the site,
 * date and confidence the user set.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { Loader2, Maximize2, Minimize2 } from "lucide-react";
import { toast } from "sonner";

import { detectionsApi } from "../../api/detections";
import { filesApi } from "../../api/files";
import { labelsApi } from "../../api/labels";
import { projectsApi } from "../../api/projects";
import { useLabelOptions } from "../../hooks/useLabelOptions";
import { useShortcutLabels } from "../../hooks/useShortcutLabels";
import { shouldDrawBbox } from "../../lib/detection-utils";
import { resolveSpeciesName } from "../../lib/species-name-mode";
import { labelMajority } from "./label-majority";
import { BulkActionBar } from "./BulkActionBar";
import { formatConfidencePct } from "../../lib/confidence";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
import { FilesGrid } from "./FilesGrid";
import { GridEmptyState } from "./GridEmptyState";
import { LabelsKeyboardPopover } from "./LabelsKeyboardPopover";
import { MOD, type Shortcut } from "./shortcuts";
import { nextAfterActed, selectOnClick } from "./grid-selection";
import { FileDetailModal } from "./FileDetailModal";
import { LabelsSettings } from "./LabelsSettings";
import { SortSelector } from "./SortSelector";
import { useFilesSort, useTileSize } from "./labels-settings";
import type { FilesSort } from "./labels-settings";
import { VerifyFilterBar } from "./VerifyFilterBar";
import {
  VerifyGuideLink,
  VerifyProgressPill,
  VerifyToolbar,
  VerifyToolbarIcon,
} from "./VerifyToolbar";
import {
  fromFilterBarFilters,
  lblFiltersFromSearchParams,
  lblFiltersToSearchParams,
  toFilterBarFilters,
  type LabelsFilterState,
} from "./labels-filters";
import { useLabelsProgress } from "./useLabelsProgress";
import { warmFiles } from "./warm-files";
import { useWideModeControls } from "./wide-mode";
import type {
  LabelsFileItem,
  LabelsFilesParams,
  LabelsFilesResponse,
  VerifySort,
} from "../../api/types";
import type { ReactNode } from "react";

const PAGE_SIZE = 48;

// The crop grid's keys, at file scope: a label action reaches every
// visible box of the selected files. In the viewer the same keys act on
// the selected box, or on every box of the picture when none is
// selected. Viewer-only extras at the end.
const FILES_SHORTCUTS: readonly Shortcut[] = [
  ["Click", "Select"],
  [`${MOD} + Click`, "Toggle select"],
  ["Shift + Click", "Extend range"],
  ["Double-click", "Open file"],
  ["Click outside", "Deselect all"],
  ["Enter", "Verify"],
  [`${MOD} + A`, "Select all on this page"],
  ["E", "Select next event to check (event sort)"],
  ["X / Backspace", "Mark false detection"],
  ["U", "Mark unknown (unidentifiable)"],
  ["R", "Relabel"],
  ["M", "Relabel to most common"],
  ["1 - 5", "Apply a saved label"],
  [`${MOD} + Z`, "Undo last label action"],
  ["Esc", "Deselect"],
  // Viewer only.
  ["← / →", "Previous / next file"],
  ["D", "Draw a box"],
  ["B", "Hide or show the boxes"],
  ["F", "Flag for review"],
  ["P", "Play a video / back to its frame"],
  ["Tab / Shift + Tab", "Select the next / previous box"],
];
const FILES_SORT_MODES: readonly VerifySort[] = [
  "path",
  "events",
  "newest",
  "oldest",
  "random",
];

interface FilesTabProps {
  projectId: string;
  toolbarExtra?: ReactNode;
  onSelectionChange?: (count: number) => void;
  /** Boxes not yet checked in the Detections tab, for the pointer shown
   *  when this grid runs out. */
  otherTabLeft?: number;
  /** Files not signed off in this tab, and every label in scope, so the
   *  empty state can tell "you finished" from "your filters are hiding
   *  the rest". */
  thisTabLeft?: number;
  totalLabels?: number;
  onSwitchTab?: () => void;
  /** Take the user to where the detection threshold is set. Supplied by
   *  the host because the control is a route in projects mode and a
   *  slideout in a folder run. Without it the note still names the
   *  setting, it just cannot offer the trip. */
  onEditThreshold?: () => void;
}

export function FilesTab({
  projectId,
  toolbarExtra,
  onSelectionChange,
  otherTabLeft = 0,
  thisTabLeft = 0,
  totalLabels = 0,
  onSwitchTab,
  onEditThreshold,
}: FilesTabProps) {
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const { wide, toggle: toggleWide } = useWideModeControls();

  const [sort, setSort] = useFilesSort();
  const [seed, setSeed] = useState<number | null>(null);
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const anchorRef = useRef<string | null>(null);
  const [openIndex, setOpenIndex] = useState<number | null>(null);
  const [tileSize, setTileSize] = useTileSize();

  const { data: project } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => projectsApi.get(projectId),
  });

  const filters = useMemo(() => {
    const f = lblFiltersFromSearchParams(searchParams);
    // The slider here is clamped at the project threshold ("empty" is
    // defined by that one setting, never by a transient control), so a
    // below-floor minimum can only be a leftover from digging down on
    // the Detections tab. Dropped rather than displayed: a chip for a
    // filter that does nothing reads as working.
    const thr = project?.counting_threshold;
    if (
      thr !== undefined &&
      f.min_confidence !== undefined &&
      f.min_confidence <= thr
    ) {
      delete f.min_confidence;
    }
    return f;
  }, [searchParams, project?.counting_threshold]);
  const progress = useLabelsProgress(projectId, filters);

  // A file handed over by the Detections modal's "Open in files view":
  // ask the list where the file sits under the current filters and
  // sort (`find`), land the grid on that page and open the viewer
  // there, so next and previous continue through the list exactly as
  // if the tile had been clicked. A file the view does not hold
  // (usually: already verified while the view shows unverified) widens
  // the verification filter to "all files", visibly, and asks again.
  // A file even that cannot reach (a box filter it no longer matches)
  // falls back to a one-file viewer, so the link always shows the
  // file. One-shot: the param is consumed and removed so back/reload
  // does not reopen it.
  const [soloItem, setSoloItem] = useState<LabelsFileItem | null>(null);
  const [pendingOpenId, setPendingOpenId] = useState<string | null>(null);
  const openSolo = useCallback((fid: string) => {
    filesApi
      .get(fid)
      .then((f) => {
        setSoloItem({
          id: f.id,
          deployment_id: f.deployment_id,
          file_path: f.file_path,
          file_type: f.file_type,
          captured_at_local: f.captured_at_local,
          verified: f.verified,
          width_px: f.width_px ?? null,
          height_px: f.height_px ?? null,
          event_id: null,
        });
      })
      .catch((err: Error) => toast.error(err.message));
  }, []);

  const setFilters = useCallback(
    (next: LabelsFilterState) => {
      setSearchParams((prev) => lblFiltersToSearchParams(next, prev), {
        replace: true,
      });
      setPage(0);
    },
    [setSearchParams],
  );

  // "all" is the page default: every file, empty or not.
  const empty = filters.empty ?? "all";

  // Both memoised so they can be dependencies without rebuilding every
  // caller that reads them once a render.
  const baseParams: LabelsFilesParams = useMemo(
    () => ({
      site_ids: filters.site_ids,
      date_from: filters.date_from,
      date_to: filters.date_to,
      labels: filters.labels,
      min_confidence: filters.min_confidence,
      max_confidence: filters.max_confidence,
      min_label_confidence: filters.min_label_confidence,
      max_label_confidence: filters.max_label_confidence,
      empty,
      flagged: filters.flagged,
      favorited: filters.favorited,
    }),
    [filters, empty],
  );

  const listParams: LabelsFilesParams = useMemo(
    () => ({
      ...baseParams,
      // "unverified" is the page default: the job is what you have not
      // looked at yet.
      verification: filters.verification ?? "unverified",
      sort,
      seed,
      skip: page * PAGE_SIZE,
      limit: PAGE_SIZE,
    }),
    [baseParams, filters.verification, sort, seed, page],
  );

  const { data, isLoading, isFetching, isPlaceholderData } = useQuery({
    queryKey: ["labels-files", projectId, listParams],
    queryFn: () => labelsApi.files(projectId, listParams),
    // Page, sort and filters are all in the key, so touching any of them
    // is a new query and the grid would otherwise blank out and remount
    // 48 tiles, each re-requesting its thumbnail. Hold the old page
    // until the new one lands, dimmed, with the spinner in the toolbar
    // saying why. Same as the Counts grid (`VerifyView`).
    placeholderData: (prev) => prev,
  });

  // Warm the next page while this one is being worked: its list row,
  // then each file's detail and thumbnail (`warmFiles`). Clicking Next
  // then paints instantly with the boxes already drawn, instead of a
  // wall of grey. Skipped while the current page is still the held-over
  // placeholder, so the prefetch never competes with the fetch the user
  // is actually waiting on.
  useEffect(() => {
    if (!data || isPlaceholderData) return;
    const nextSkip = (page + 1) * PAGE_SIZE;
    if (nextSkip >= data.total) return;
    const nextParams = { ...listParams, skip: nextSkip };
    queryClient
      .prefetchQuery({
        queryKey: ["labels-files", projectId, nextParams],
        queryFn: () => labelsApi.files(projectId, nextParams),
      })
      .then(() => {
        const next = queryClient.getQueryData<LabelsFilesResponse>([
          "labels-files",
          projectId,
          nextParams,
        ]);
        warmFiles(
          queryClient,
          (next?.items ?? []).map((i) => i.id),
        );
      });
  }, [data, isPlaceholderData, page, listParams, projectId, queryClient]);

  const items = useMemo(() => data?.items ?? [], [data]);

  // The consumption of `lbl_file` (see the comment on `soloItem`).
  // Lives below the list query because it probes with the very
  // `listParams` the grid uses; the guard makes reruns harmless once
  // the param is gone.
  useEffect(() => {
    const fid = searchParams.get("lbl_file");
    if (!fid) return;
    setSearchParams(
      (prev) => {
        const sp = new URLSearchParams(prev);
        sp.delete("lbl_file");
        return sp;
      },
      { replace: true },
    );
    const probe = { ...listParams, skip: 0, limit: 1, find: fid };
    (async () => {
      try {
        let res = await labelsApi.files(projectId, probe);
        if (
          res.find_index == null &&
          (filters.verification ?? "unverified") !== "all"
        ) {
          res = await labelsApi.files(projectId, {
            ...probe,
            verification: "all",
          });
          if (res.find_index != null) {
            setFilters({ ...filters, verification: "all" });
          }
        }
        if (res.find_index != null) {
          setPage(Math.floor(res.find_index / PAGE_SIZE));
          setPendingOpenId(fid);
        } else {
          openSolo(fid);
        }
      } catch (err) {
        toast.error((err as Error).message);
      }
    })();
  }, [
    searchParams,
    setSearchParams,
    listParams,
    filters,
    setFilters,
    projectId,
    openSolo,
  ]);

  // The page the handoff picked has to land before the viewer can point
  // at an index in it. If the file is not on the loaded page after all
  // (its rank moved between the probe and the fetch), show it alone
  // rather than opening the viewer on a neighbour.
  useEffect(() => {
    if (!pendingOpenId || !data || isPlaceholderData) return;
    const idx = items.findIndex((i) => i.id === pendingOpenId);
    setPendingOpenId(null);
    if (idx >= 0) setOpenIndex(idx);
    else openSolo(pendingOpenId);
  }, [pendingOpenId, data, isPlaceholderData, items, openSolo]);

  // How many the view holds ignoring the checked filter. Only fetched
  // when the grid has come back empty, which is the one moment the
  // difference matters: "you verified all 12 here" versus "nothing
  // matched at all" are different messages and this is what tells them
  // apart. Costs nothing while there is anything to show.
  const { data: viewIgnoringChecked } = useQuery({
    queryKey: ["labels-files-view-total", projectId, baseParams],
    queryFn: () => labelsApi.files(projectId, { ...baseParams, limit: 1 }),
    enabled: !isLoading && (data?.total ?? 0) === 0,
  });
  const viewCountIgnoringChecked = viewIgnoringChecked?.total ?? 0;
  const orderedIds = useMemo(() => items.map((i) => i.id), [items]);
  const total = data?.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const floor = data?.floor ?? project?.counting_threshold ?? 0;

  const refresh = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["labels-files", projectId] });
    queryClient.invalidateQueries({ queryKey: ["labels-progress", projectId] });
  }, [projectId, queryClient]);

  // The viewer changes files while it is open, and some of those
  // changes take the file out of this list: sign it off and it is no
  // longer unverified. Refetching there would pull the file out from
  // under the person still working on it, so the counts update straight
  // away and the list waits for the viewer to close.
  const listDirtyRef = useRef(false);
  const refreshCountsNow = useCallback(() => {
    listDirtyRef.current = true;
    queryClient.invalidateQueries({ queryKey: ["labels-progress", projectId] });
  }, [projectId, queryClient]);
  const closeViewer = useCallback(() => {
    setOpenIndex(null);
    if (listDirtyRef.current) {
      listDirtyRef.current = false;
      queryClient.invalidateQueries({ queryKey: ["labels-files", projectId] });
    }
  }, [projectId, queryClient]);

  /**
   * The viewer reached the end of this page. Fetch what comes next and
   * hand it a fresh list instead of shutting it, so a long run of files
   * is one pass rather than 48 files and a reopen.
   *
   * The viewer holds a spinner while this runs, and that is what makes
   * it safe: its index points into `items`, so the two would disagree
   * for a render if the list swapped underneath it.
   *
   * On the page default, "unverified", the files just signed off leave
   * the result set, so the same page number refills with what follows
   * and nothing is skipped. On the other verification filters they stay
   * put, so there is nothing new to show and the viewer closes, exactly
   * as it did before.
   */
  const [loadingMore, setLoadingMore] = useState(false);
  const continueViewer = useCallback(async () => {
    setLoadingMore(true);
    try {
      // Verifies are fired and not awaited, so that holding Enter moves
      // between files at typing speed rather than at network speed. The
      // last of them is therefore still in flight when we get here, and
      // fetching now asks the server a question it cannot answer yet:
      // measured, the GET went out 14 ms before its own PATCH, so the
      // file just signed off came back at the top of the next batch and
      // the run stepped over it a second time. Bounded, because a wedged
      // request must not strand the viewer on a spinner; falling through
      // costs a repeat, never a lost verdict.
      for (let i = 0; i < 40 && queryClient.isMutating() > 0; i++) {
        await new Promise((resolve) => setTimeout(resolve, 50));
      }
      const fresh = await labelsApi.files(projectId, listParams);
      queryClient.setQueryData(["labels-files", projectId, listParams], fresh);
      listDirtyRef.current = false;
      const next = fresh.items.findIndex((i) => !i.verified);
      setOpenIndex(next === -1 ? null : next);
    } catch (err) {
      // Fall back to what used to happen: shut the viewer and let the
      // grid refresh itself. `closeViewer`, not `setOpenIndex(null)`,
      // because the list is still dirty from the verifies just made.
      toast.error((err as Error).message);
      closeViewer();
    } finally {
      setLoadingMore(false);
    }
  }, [projectId, listParams, queryClient, closeViewer]);

  const updateSelection = useCallback(
    (next: Set<string>) => {
      setSelected(next);
      onSelectionChange?.(next.size);
    },
    [onSelectionChange],
  );

  // Click / shift-range / cmd-toggle come from `grid-selection.ts`, the
  // same module the crop grid uses, so the gestures are identical.
  const handleSelect = useCallback(
    (fileId: string, e: React.MouseEvent) => {
      const result = selectOnClick(
        orderedIds,
        anchorRef.current,
        fileId,
        e,
        selected,
      );
      anchorRef.current = result.anchor;
      updateSelection(result.ids);
    },
    [orderedIds, selected, updateSelection],
  );

  const clearSelection = useCallback(() => {
    anchorRef.current = null;
    updateSelection(new Set());
  }, [updateSelection]);

  /** After a bulk action, select the file sliding into the freed slot,
   *  so a repeated pass never needs the mouse again. Same rule as the
   *  crop grid. */
  const advanceAfter = useCallback(
    (actedIds: string[]) => {
      // `orderedIds` is still the pre-action order here: the refetch
      // that removes the acted rows is kicked off after this runs.
      const next = nextAfterActed(orderedIds, actedIds);
      if (next === null) {
        clearSelection();
        return;
      }
      anchorRef.current = next;
      updateSelection(new Set([next]));
    },
    [orderedIds, clearSelection, updateSelection],
  );

  const markChecked = useMutation({
    mutationFn: (ids: string[]) => filesApi.bulkVerify(ids),
    onSuccess: (_r, ids) => {
      // No success toast: this is the repeated action on this page, and
      // the result is already on screen. The files leave the grid, the
      // selection moves to the next one, and the progress bar ticks up.
      // Matches the crop grid, which stays quiet on a verify too. Errors
      // still toast.
      undoStackRef.current.push({ kind: "verify", fileIds: ids });
      setCanUndo(true);
      advanceAfter(ids);
      refresh();
      // The tiles draw from the file detail, which the verify changed.
      for (const id of ids) {
        queryClient.invalidateQueries({ queryKey: ["file", id] });
      }
    },
    onError: (err: Error) => toast.error(err.message),
  });

  // ── Bulk box actions, the crop grid's set at file scope ──────────
  // One mechanism behind X / U / M / R / 1-5 and their buttons: apply a
  // label to every visible box of the selected files. The tiles already
  // hold each file's detail in the query cache, so collecting the box
  // ids costs nothing beyond a cache read; `fetchQuery` covers a tile
  // whose fetch has not landed yet. Undo mirrors the crop grid's:
  // Cmd+Z reverts the last action's boxes to the model's own call.
  const threshold = project?.counting_threshold ?? 0;
  const { options: labelOptions, isLoading: labelOptionsLoading } =
    useLabelOptions(project?.classification_model_id ?? null, projectId);
  const { shortcutLabels } = useShortcutLabels(projectId);
  const [relabelOpen, setRelabelOpen] = useState(false);
  // Label entries revert boxes to the model's call; verify entries
  // undo with an untick (`bulkVerify(ids, false)`), which is verify's
  // exact inverse: every box unverified, and the next reprocess hands
  // the rejected ones back to the AI. Same stack, so Cmd+Z walks back
  // through verifies and relabels in the order they happened.
  const undoStackRef = useRef<
    (
      | { kind: "label"; boxIds: string[]; fileIds: string[] }
      | { kind: "verify"; fileIds: string[] }
    )[]
  >([]);
  // Fix for keyboard double-fire: the bar's buttons disable while a
  // mutation is pending, but M/X/2 on the keyboard have no such gate.
  const bulkBusyRef = useRef(false);
  const [canUndo, setCanUndo] = useState(false);

  const visibleBoxesOf = useCallback(
    async (fileIds: string[]) => {
      const files = await Promise.all(
        fileIds.map((id) =>
          queryClient.fetchQuery({
            queryKey: ["file", id],
            queryFn: () => filesApi.get(id),
          }),
        ),
      );
      return files.flatMap((f) =>
        f.detections.filter((d) => shouldDrawBbox(d, f, threshold)),
      );
    },
    [queryClient, threshold],
  );

  const applyLabelToSelection = useCallback(
    async (label: string | null, category: string | undefined) => {
      const fileIds = [...selected];
      if (fileIds.length === 0) return;
      if (bulkBusyRef.current) return;
      bulkBusyRef.current = true;
      try {
        const boxes = await visibleBoxesOf(fileIds);
        if (boxes.length === 0) {
          toast.info("No visible boxes in the selection");
          return;
        }
        const boxIds = boxes.map((b) => b.id);
        await detectionsApi.bulkRelabel(boxIds, label, category);
        undoStackRef.current.push({ kind: "label", boxIds, fileIds });
        setCanUndo(true);
        // Relabelling verifies the boxes, so the files roll up to
        // verified and leave the default (unverified) view; advance the
        // selection exactly as a verify does.
        advanceAfter(fileIds);
        refresh();
        for (const id of fileIds) {
          queryClient.invalidateQueries({ queryKey: ["file", id] });
        }
      } catch (err) {
        toast.error((err as Error).message);
      } finally {
        bulkBusyRef.current = false;
      }
    },
    [selected, visibleBoxesOf, advanceAfter, refresh, queryClient],
  );

  // The selection's majority, from the page cache, for the bar button's
  // own text; the shared helper keeps the rule identical to the crop
  // grid's Match majority and the viewer's M.
  const selectionMajority = useMemo(() => {
    const boxes = [];
    for (const id of selected) {
      const f = queryClient.getQueryData<
        Awaited<ReturnType<typeof filesApi.get>>
      >(["file", id]);
      if (!f) continue;
      for (const d of f.detections) {
        if (shouldDrawBbox(d, f, threshold)) boxes.push(d);
      }
    }
    return labelMajority(boxes);
  }, [selected, threshold, queryClient]);

  const matchMajority = useCallback(async () => {
    const fileIds = [...selected];
    if (fileIds.length === 0 || bulkBusyRef.current) return;
    // The bar button's own text comes from the tile cache
    // (`selectionMajority`), but the action must not: tiles still
    // fetching would make that a majority of whatever happened to be
    // loaded. Fetch the full set (cached tiles cost nothing) and let
    // `applyLabelToSelection` reuse it from the cache.
    const boxes = await visibleBoxesOf(fileIds);
    const mode = labelMajority(boxes);
    if (!mode) {
      toast.info("No labels in selection to apply");
      return;
    }
    applyLabelToSelection(mode.label, mode.category);
  }, [selected, visibleBoxesOf, applyLabelToSelection]);

  const handleUndo = useCallback(async () => {
    const entry = undoStackRef.current.pop();
    setCanUndo(undoStackRef.current.length > 0);
    if (!entry) return;
    try {
      if (entry.kind === "verify") {
        await filesApi.bulkVerify(entry.fileIds, false);
      } else {
        await detectionsApi.bulkRevertToOriginal(entry.boxIds);
      }
      refresh();
      for (const id of entry.fileIds) {
        queryClient.invalidateQueries({ queryKey: ["file", id] });
      }
    } catch (err) {
      toast.error((err as Error).message);
    }
  }, [refresh, queryClient]);

  // Grid keyboard, matching the crop grid: Escape clears the selection,
  // Enter acts on it. Skipped while the viewer is open, which owns the
  // keyboard for the file it is showing.
  useEffect(() => {
    if (openIndex !== null) return;
    const onKey = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }
      if (e.key === "Escape" && selected.size > 0) {
        e.preventDefault();
        clearSelection();
      } else if (e.key === "Enter" && selected.size > 0) {
        e.preventDefault();
        if (!markChecked.isPending) markChecked.mutate([...selected]);
      } else if (e.key === "a" && (e.metaKey || e.ctrlKey)) {
        // Select everything on screen, like the crop grid. "On screen"
        // is this page, not all 229: the action would otherwise reach
        // files the user has not looked at and cannot see.
        e.preventDefault();
        anchorRef.current = orderedIds[0] ?? null;
        updateSelection(new Set(orderedIds));
      } else if (e.key === "z" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        handleUndo();
      } else if (
        (e.key === "x" || e.key === "X" || e.key === "Backspace" ||
          e.key === "Delete") &&
        !e.ctrlKey && !e.metaKey && selected.size > 0
      ) {
        e.preventDefault();
        applyLabelToSelection("false detection", undefined);
      } else if (
        (e.key === "u" || e.key === "U") &&
        !e.ctrlKey && !e.metaKey && selected.size > 0
      ) {
        e.preventDefault();
        applyLabelToSelection("unknown", undefined);
      } else if (
        (e.key === "m" || e.key === "M") &&
        !e.ctrlKey && !e.metaKey && selected.size > 0
      ) {
        e.preventDefault();
        matchMajority();
      } else if (
        (e.key === "r" || e.key === "R") &&
        !e.ctrlKey && !e.metaKey && selected.size > 0
      ) {
        e.preventDefault();
        setRelabelOpen((v) => !v);
      } else if (
        e.key >= "1" && e.key <= "5" &&
        !e.ctrlKey && !e.metaKey && selected.size > 0
      ) {
        const slot = shortcutLabels[parseInt(e.key)];
        if (slot) {
          e.preventDefault();
          applyLabelToSelection(slot.label, slot.category);
        }
      } else if (
        (e.key === "e" || e.key === "E") &&
        !e.ctrlKey &&
        !e.metaKey &&
        sort === "events"
      ) {
        // Same key as the crop grid: select the first event on this
        // page that still has an unverified file, whole burst at once,
        // and bring it into view. A no-op in the other sorts, where
        // events are not contiguous.
        e.preventDefault();
        let i = 0;
        while (i < items.length) {
          const eventId = items[i].event_id;
          let j = i;
          while (j < items.length && items[j].event_id === eventId) j++;
          const run = items.slice(i, j);
          if (run.some((f) => !f.verified)) {
            const ids = run.map((f) => f.id);
            anchorRef.current = ids[0];
            updateSelection(new Set(ids));
            // The event's divider to the top of the viewport, exactly
            // where the crop grid's E puts it (`scrollToDetection`
            // targets the divider row with align "start"), so the two
            // tabs land the eye in the same place.
            const divider = document.querySelector(
              `[data-event-divider="${eventId ?? "none"}"]`,
            );
            (divider ?? document.querySelector(`[data-file-id="${ids[0]}"]`))
              ?.scrollIntoView({ block: "start" });
            break;
          }
          i = j;
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    openIndex,
    selected,
    clearSelection,
    markChecked,
    orderedIds,
    updateSelection,
    sort,
    items,
    handleUndo,
    applyLabelToSelection,
    matchMajority,
    shortcutLabels,
  ]);

  const countNoun =
    empty === "show_only"
      ? " with nothing found"
      : empty === "hide"
        ? " with detections"
        : "";

  // The "Count detections above" setting, named as a link when the host
  // can navigate to it (a route in projects mode, a slideout in a
  // folder run). Shared by the empties callout and the slider's clamp
  // caption, so the two can never name it differently.
  const thresholdSettingLink = onEditThreshold ? (
    <button
      type="button"
      onClick={onEditThreshold}
      className="underline underline-offset-2 hover:no-underline"
    >
      Count detections above
    </button>
  ) : (
    "Count detections above"
  );

  return (
    <div className="space-y-4">
      <VerifyFilterBar
        filters={toFilterBarFilters(filters)}
        onChange={(fp) => setFilters(fromFilterBarFilters(fp, filters))}
        projectId={projectId}
        classificationModelId={project?.classification_model_id ?? null}
        detectionFloor={project?.counting_threshold ?? 0}
        countBy="file"
        // Liked / flagged as on Counts, plus the Empty select and the
        // confidence ranges in More filters. Each range means "at
        // least one box in it": the Detections rules lifted to files
        // (see `get_labels_files`). Clamped at the project threshold,
        // unlike Detections: this surface can never show a
        // sub-threshold box (Verify signs off what is drawn), so the
        // slider must not pretend to reach below.
        showLikedFlaggedEmpty
        showEmpty
        emptyDefault="all"
        confidenceFloorMode="clamp"
        clampReason={
          <>
            The minimum is your {thresholdSettingLink} setting (
            {formatConfidencePct(project?.counting_threshold ?? 0)}).
          </>
        }
        verificationDefault="unverified"
      />

      <VerifyToolbar>
        {toolbarExtra}
        <SortSelector
          sort={sort}
          seed={seed}
          availableSorts={FILES_SORT_MODES}
          onChange={(next, nextSeed) => {
            setSort(next as FilesSort);
            setSeed(nextSeed ?? null);
            setPage(0);
          }}
        />
        {/* Same icons in the same order as the Detections tab: full width,
            help, keyboard, tile size, then the progress pill. */}
        <div className="ml-auto flex items-center gap-1">
          <VerifyToolbarIcon
            icon={wide ? Minimize2 : Maximize2}
            title={wide ? "Exit full width" : "Full width"}
            onClick={toggleWide}
            active={wide}
          />
          <VerifyGuideLink step="labels" />
          <LabelsKeyboardPopover
            shortcuts={FILES_SHORTCUTS}
            footer="After verifying, the next file is selected, so you can keep going."
          />
          <LabelsSettings tileSize={tileSize} onTileSizeChange={setTileSize} />
          <div className="ml-2">
            <VerifyProgressPill
              pct={progress.pct}
              label="verified"
              title={progress.title}
            />
          </div>
        </div>
      </VerifyToolbar>

      {/* Only while the Empty filter is narrowing to empties: that is
          the one view where "empty" needs defining. One sentence, because
          the setting's own name says what it does. Naming where it lives
          is gone too: it is a link, and that is a location the note
          would otherwise have to word per mode. */}
      {empty === "show_only" && (
        <Callout variant="info" size="compact">
          {/* The project's threshold, not `floor`. `floor` is the
              effective one, which the confidence slider drags down, so
              quoting it made the sentence claim the setting was 1% when
              it was really 20%. The slider is a lens; this sentence is
              about the setting it names. */}
          A file counts as empty when nothing was found with a detection
          confidence above{" "}
          {formatConfidencePct(project?.counting_threshold ?? floor)}, the
          threshold set with {thresholdSettingLink}.
        </Callout>
      )}

      {isLoading ? (
        <div className="flex h-64 items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : items.length === 0 ? (
        <GridEmptyState
          thisTabLeft={thisTabLeft}
          otherTabLeft={otherTabLeft}
          totalLabels={totalLabels}
          viewFinished={viewCountIgnoringChecked > 0}
          viewCount={viewCountIgnoringChecked}
          tabHasNothing={viewCountIgnoringChecked === 0}
          noun="files"
          otherNoun="detections"
          otherTabName="Detections"
          onClearFilters={() => setFilters({})}
          onSwitchTab={onSwitchTab}
        />
      ) : (
        <>
          <FilesGrid
            // Dimmed while the tiles on screen belong to the previous
            // page, so held-over content never reads as the answer to
            // what you just asked for. Same cue as the Counts grid.
            className={isPlaceholderData ? "opacity-60" : undefined}
            items={items}
            selectedIds={selected}
            onSelect={handleSelect}
            detectionThreshold={project?.counting_threshold ?? 0}
            tileSize={tileSize}
            groupByEvent={sort === "events"}
            onSelectEvent={(ids) => updateSelection(new Set(ids))}
            onBackgroundClick={clearSelection}
            onOpen={(item: LabelsFileItem) =>
              setOpenIndex(items.findIndex((i) => i.id === item.id))
            }
          />

          {/* Count and paging share one row, the Counts grid's idiom
              ("Showing 1-50 of 1040 events"). The count used to be its
              own line above the grid, which duplicated the tab chip a
              few pixels higher and pushed the tiles down for one
              number. Always rendered: on a single filtered page,
              "Showing 1-6 of 6 files with nothing found" is the only
              place the filter's exact result appears. */}
          <div className="flex items-center justify-center gap-3 pt-2">
            <Button
              variant="outline"
              size="sm"
              disabled={page === 0}
              onClick={() => setPage((p) => Math.max(0, p - 1))}
            >
              Previous
            </Button>
            <span className="text-sm text-muted-foreground">
              Showing {(page * PAGE_SIZE + 1).toLocaleString()}-
              {(page * PAGE_SIZE + items.length).toLocaleString()}
              {" of "}
              {total.toLocaleString()} file{total === 1 ? "" : "s"}
              {countNoun}
              {isFetching && " (loading...)"}
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={page >= pageCount - 1}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </Button>
          </div>
        </>
      )}

      {/* The Detections grid's bar, verbatim: `performAction` maps each
          button to this tab's file-scoped implementation (every visible
          box of the selected files) while the presentation stays one
          component, so the two grids cannot drift. Verify is immediate,
          like Enter: the old confirm dialog guarded an irreversible
          delete that no longer exists (untick a file and the next
          reprocess hands its rejected boxes back to the AI). */}
      <BulkActionBar
        selectedIds={selected}
        onDeselectAll={clearSelection}
        labelOptions={labelOptions}
        labelOptionsLoading={labelOptionsLoading}
        onActionComplete={() => {}}
        performAction={(action) => {
          if (action === "verify") {
            markChecked.mutate([...selected]);
          } else if (action === "false") {
            return applyLabelToSelection("false detection", undefined);
          } else if (action === "unknown") {
            return applyLabelToSelection("unknown", undefined);
          } else {
            return applyLabelToSelection(
              action.relabel.label,
              action.relabel.category,
            );
          }
        }}
        onMatchMajority={matchMajority}
        majorityLabel={
          selectionMajority ? resolveSpeciesName(selectionMajority) : null
        }
        projectId={projectId}
        relabelOpen={relabelOpen}
        onRelabelOpenChange={setRelabelOpen}
        onUndo={handleUndo}
        canUndo={canUndo}
      />

      <FileDetailModal
        projectId={projectId}
        items={soloItem ? [soloItem] : items}
        index={soloItem ? 0 : openIndex}
        onIndexChange={soloItem ? () => {} : setOpenIndex}
        onClose={soloItem ? () => setSoloItem(null) : closeViewer}
        onExhausted={soloItem ? () => setSoloItem(null) : continueViewer}
        loadingMore={soloItem ? false : loadingMore}
        onChanged={refreshCountsNow}
      />
    </div>
  );
}
