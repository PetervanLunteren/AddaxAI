/**
 * EmptiesTab - the Empties half of the Labels page.
 *
 * One tile per photo where nothing passed the detection threshold. The
 * job here is different from the crop grid's: not "is this label
 * right?" but "did the detector miss something?". So the unit is the
 * photo, the sorts are about where a photo sits rather than what it
 * looks like, and the only verdicts are "yes it is empty" (checked) or
 * a drawn box, which takes the photo out of this list entirely.
 *
 * Shares its filter state with the crop grid through the `lbl_*` URL
 * params (see `labels-filters.ts`), so switching tabs keeps the site,
 * date and confidence the user set.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import {
  CheckCheck,
  Loader2,
  Maximize2,
  Minimize2,
  X,
} from "lucide-react";
import { toast } from "sonner";

import { filesApi } from "../../api/files";
import { labelsApi } from "../../api/labels";
import { projectsApi } from "../../api/projects";
import { formatConfidencePct } from "../../lib/confidence";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "../ui/alert-dialog";
import { EmptiesGrid } from "./EmptiesGrid";
import { GridEmptyState } from "./GridEmptyState";
import { LabelsKeyboardPopover } from "./LabelsKeyboardPopover";
import { MOD, type Shortcut } from "./shortcuts";
import { nextAfterActed, selectOnClick } from "./grid-selection";
import { EmptyPhotoModal } from "./EmptyPhotoModal";
import { LabelsSettings } from "./LabelsSettings";
import { SortSelector } from "./SortSelector";
import { useTileSize } from "./labels-settings";
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
import { useWideModeControls } from "./wide-mode";
import type { EmptiesParams, EmptyFileItem, VerifySort } from "../../api/types";
import type { ReactNode } from "react";

const PAGE_SIZE = 48;

const EMPTIES_SHORTCUTS: readonly Shortcut[] = [
  ["Click", "Select"],
  [`${MOD} + Click`, "Toggle select"],
  ["Shift + Click", "Extend range"],
  ["Double-click", "Open file"],
  ["Click outside", "Deselect all"],
  ["Enter", "Verify"],
  [`${MOD} + A`, "Select all on this page"],
  ["Esc", "Deselect"],
];
const EMPTIES_SORT_MODES: readonly VerifySort[] = [
  "path",
  "newest",
  "oldest",
  "random",
];

type EmptiesSort = "path" | "newest" | "oldest" | "random";

interface EmptiesTabProps {
  projectId: string;
  toolbarExtra?: ReactNode;
  onSelectionChange?: (count: number) => void;
  /** Labels not yet checked in the Detections tab, for the pointer shown when
   *  this grid runs out. */
  otherTabLeft?: number;
  /** Unverified labels in this tab, and every label in scope, so the
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

export function EmptiesTab({
  projectId,
  toolbarExtra,
  onSelectionChange,
  otherTabLeft = 0,
  thisTabLeft = 0,
  totalLabels = 0,
  onSwitchTab,
  onEditThreshold,
}: EmptiesTabProps) {
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const { wide, toggle: toggleWide } = useWideModeControls();

  const [sort, setSort] = useState<EmptiesSort>("path");
  const [seed, setSeed] = useState<number | null>(null);
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const anchorRef = useRef<string | null>(null);
  const [openIndex, setOpenIndex] = useState<number | null>(null);
  const [tileSize, setTileSize] = useTileSize();
  // Only the bulk path asks. In the viewer you are looking at the photo
  // when you decide; here Cmd+A selects 47 files you have not opened,
  // and verifying removes the detector's boxes from every one of them,
  // with no undo (unlike the Detections grid, which has an undo stack).
  // Holds the ids, not a flag. The dialog's own text would otherwise
  // read from the live selection, which empties as the action runs: it
  // re-rendered to "Verify 0 files?" mid-close, and that re-render left
  // Radix's exit transition stranded so the panel never unmounted.
  const [confirmBulk, setConfirmBulk] = useState<string[] | null>(null);

  const filters = useMemo(
    () => lblFiltersFromSearchParams(searchParams),
    [searchParams],
  );
  const progress = useLabelsProgress(projectId, filters);

  const setFilters = useCallback(
    (next: LabelsFilterState) => {
      setSearchParams((prev) => lblFiltersToSearchParams(next, prev), {
        replace: true,
      });
      setPage(0);
    },
    [setSearchParams],
  );

  const { data: project } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => projectsApi.get(projectId),
  });

  // Both memoised so they can be dependencies without rebuilding every
  // caller that reads them once a render.
  const baseParams: EmptiesParams = useMemo(
    () => ({
      site_ids: filters.site_ids,
      date_from: filters.date_from,
      date_to: filters.date_to,
      min_confidence: filters.min_confidence,
    }),
    [filters],
  );

  const listParams: EmptiesParams = useMemo(
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
    queryKey: ["empties", projectId, listParams],
    queryFn: () => labelsApi.empties(projectId, listParams),
    // Page, sort and filters are all in the key, so touching any of them
    // is a new query and the grid would otherwise blank out and remount
    // 48 tiles, each re-requesting its thumbnail. Hold the old page
    // until the new one lands, dimmed, with the spinner in the toolbar
    // saying why. Same as the Counts grid (`VerifyView`).
    placeholderData: (prev) => prev,
  });

  const items = useMemo(() => data?.items ?? [], [data]);

  // How many the view holds ignoring the checked filter. Only fetched
  // when the grid has come back empty, which is the one moment the
  // difference matters: "you verified all 12 here" versus "nothing
  // matched at all" are different messages and this is what tells them
  // apart. Costs nothing while there is anything to show.
  const { data: viewIgnoringChecked } = useQuery({
    queryKey: ["empties-view-total", projectId, baseParams],
    queryFn: () => labelsApi.empties(projectId, { ...baseParams, limit: 1 }),
    enabled: !isLoading && (data?.total ?? 0) === 0,
  });
  const viewCountIgnoringChecked = viewIgnoringChecked?.total ?? 0;
  const orderedIds = useMemo(() => items.map((i) => i.id), [items]);
  const total = data?.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const floor = data?.floor ?? project?.counting_threshold ?? 0;

  const refresh = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["empties", projectId] });
    queryClient.invalidateQueries({ queryKey: ["labels-progress", projectId] });
  }, [projectId, queryClient]);

  // The viewer changes files while it is open, and some of those
  // changes take the file out of this list: draw a box and it is no
  // longer empty. Refetching there would pull the file out from under
  // the person still working on the box they just drew, so the counts
  // update straight away and the list waits for the viewer to close.
  const listDirtyRef = useRef(false);
  const refreshCountsNow = useCallback(() => {
    listDirtyRef.current = true;
    queryClient.invalidateQueries({ queryKey: ["labels-progress", projectId] });
  }, [projectId, queryClient]);
  const closeViewer = useCallback(() => {
    setOpenIndex(null);
    if (listDirtyRef.current) {
      listDirtyRef.current = false;
      queryClient.invalidateQueries({ queryKey: ["empties", projectId] });
    }
  }, [projectId, queryClient]);

  /**
   * The viewer reached the end of this page. Fetch what comes next and
   * hand it a fresh list instead of shutting it, so a long run of
   * empties is one pass rather than 48 files and a reopen.
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
      const fresh = await labelsApi.empties(projectId, listParams);
      queryClient.setQueryData(["empties", projectId, listParams], fresh);
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

  /** After a bulk action, select the photo sliding into the freed slot,
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
    mutationFn: async (ids: string[]) => {
      // No bulk file endpoint exists and one photo at a time is honest
      // here: a page is 48, not 50,000.
      for (const id of ids) {
        await filesApi.update(id, { verified: true });
      }
    },
    onSuccess: (_r, ids) => {
      // No success toast: this is the repeated action on this page, and
      // the result is already on screen. The photos leave the grid, the
      // selection moves to the next one, and the progress bar ticks up.
      // Matches the crop grid, which stays quiet on a verify too. Errors
      // still toast.
      advanceAfter(ids);
      refresh();
    },
    onError: (err: Error) => toast.error(err.message),
  });

  // Grid keyboard, matching the crop grid: Escape clears the selection,
  // Enter acts on it. Skipped while the viewer is open, which owns the
  // keyboard for the photo it is showing.
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
        // photos the user has not looked at and cannot see.
        e.preventDefault();
        anchorRef.current = orderedIds[0] ?? null;
        updateSelection(new Set(orderedIds));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [openIndex, selected, clearSelection, markChecked, orderedIds, updateSelection]);

  return (
    <div className="space-y-4">
      <VerifyFilterBar
        filters={toFilterBarFilters(filters)}
        onChange={(fp) => setFilters(fromFilterBarFilters(fp, filters))}
        projectId={projectId}
        detectionFloor={project?.counting_threshold ?? 0}
        countBy="file"
        // Every photo here is empty by definition, so it has no label to
        // filter on and no liked / flagged / empty triple to apply.
        showLabels={false}
        showLikedFlaggedEmpty={false}
        confidenceFloorMode="open"
        verificationDefault="unverified"
      />

      <VerifyToolbar>
        {toolbarExtra}
        <SortSelector
          sort={sort}
          seed={seed}
          availableSorts={EMPTIES_SORT_MODES}
          onChange={(next, nextSeed) => {
            setSort(next as EmptiesSort);
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
            shortcuts={EMPTIES_SHORTCUTS}
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

      {/* One sentence, because the setting's own name now says what it
          does and its caption says how far it reaches. Naming where it
          lives is gone too: it is a link, and that is a location the
          note would otherwise have to word per mode. */}
      <Callout variant="info" size="compact">
        {/* The project's threshold, not `floor`. `floor` is the
            effective one, which the confidence slider drags down, so
            quoting it made the sentence claim the setting was 1% when it
            was really 20%. The slider is a lens; this sentence is about
            the setting it names. */}
        A file counts as empty when nothing was found with a detection
        confidence above{" "}
        {formatConfidencePct(project?.counting_threshold ?? floor)}, the
        threshold set with{" "}
        {onEditThreshold ? (
          <button
            type="button"
            onClick={onEditThreshold}
            className="underline underline-offset-2 hover:no-underline"
          >
            Count detections above
          </button>
        ) : (
          "Count detections above"
        )}
        .
      </Callout>

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
          noun="empty files"
          otherNoun="detections"
          otherTabName="Detections"
          onClearFilters={() => setFilters({})}
          onSwitchTab={onSwitchTab}
        />
      ) : (
        <>
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>
              {total.toLocaleString()} file{total === 1 ? "" : "s"} with
              nothing found
            </span>
            {isFetching && <Loader2 className="h-4 w-4 animate-spin" />}
          </div>

          <EmptiesGrid
            // Dimmed while the tiles on screen belong to the previous
            // page, so held-over content never reads as the answer to
            // what you just asked for. Same cue as the Counts grid.
            className={isPlaceholderData ? "opacity-60" : undefined}
            items={items}
            selectedIds={selected}
            onSelect={handleSelect}
            tileSize={tileSize}
            onBackgroundClick={clearSelection}
            onOpen={(item: EmptyFileItem) =>
              setOpenIndex(items.findIndex((i) => i.id === item.id))
            }
          />

          {pageCount > 1 && (
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
                Page {page + 1} of {pageCount}
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
          )}
        </>
      )}

      {selected.size > 0 && (
        <div className="fixed bottom-6 left-1/2 z-40 -translate-x-1/2">
          <div className="flex items-center gap-3 rounded-full border bg-white px-4 py-2 shadow-lg">
            <span className="text-sm font-medium">
              {selected.size} selected
            </span>
            <Button
              size="sm"
              disabled={markChecked.isPending}
              onClick={() => setConfirmBulk([...selected])}
            >
              <CheckCheck className="mr-1.5 h-4 w-4" />
              Verify
            </Button>
            <button
              type="button"
              title="Deselect (Esc)"
              className="text-muted-foreground hover:text-foreground"
              onClick={clearSelection}
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      {/* The work starts only once the dialog has finished closing.
          Running 48 sequential PATCHes while it was still unmounting
          re-rendered this whole subtree through Radix's exit
          transition, which then never completed: the panel stayed
          mounted and its scroll lock left `body` at
          `pointer-events: none`, so the page was unusable until a
          reload. Let the Action close it, then act in `onOpenChange`. */}
      <AlertDialog
        open={confirmBulk !== null}
        onOpenChange={(open) => {
          if (!open) setConfirmBulk(null);
        }}
      >
        {/* No exit animation. Radix keeps the panel mounted until the
            close animation ends, and the 48 sequential PATCHes this
            starts re-render the tree hard enough that the end never
            arrives: it stayed mounted with `body` at
            `pointer-events: none`, locking the page until a reload.
            Nothing to wait for means it unmounts at once. */}
        <AlertDialogContent className="!animate-none">
          <AlertDialogHeader>
            <AlertDialogTitle>
              Verify {confirmBulk?.length ?? 0} file
              {confirmBulk?.length === 1 ? "" : "s"}?
            </AlertDialogTitle>
            <AlertDialogDescription>
              You are saying there is nothing in{" "}
              {confirmBulk?.length === 1 ? "it" : "them"}, so the
              detector's leftover boxes are removed. That cannot be
              undone. Boxes you drew yourself are kept.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => confirmBulk && markChecked.mutate(confirmBulk)}
            >
              Verify
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <EmptyPhotoModal
        projectId={projectId}
        items={items}
        index={openIndex}
        onIndexChange={setOpenIndex}
        onClose={closeViewer}
        onExhausted={continueViewer}
        loadingMore={loadingMore}
        onChanged={refreshCountsNow}
      />
    </div>
  );
}
