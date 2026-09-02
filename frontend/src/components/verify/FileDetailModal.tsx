/**
 * Full-size view of one file, for the Files tab.
 *
 * Same frame as the Detections detail view (`VerifyDetailShell`): picture on
 * the left, what it is on the top right, what you can do on the bottom
 * right. A person is doing the same job in both, so they should not
 * have to learn two screens.
 *
 * Every visible box is drawn: above the threshold, or verified. The
 * sub-threshold boxes are not, and that is load-bearing: Verify says
 * "the boxes you can see are all there is", and the backend deletes the
 * ones you could not see. Drawing them would make that verdict about a
 * threshold instead of about the picture.
 *
 * What you can do. **Verify** signs the file off and moves on; it is one
 * action because it is one decision. Draw a box on an animal the
 * detector missed: arm the crosshair with D or the button, drag, and the
 * species search opens on the new box by itself. Name a species in the
 * Default label card and new boxes take that instead, with nothing to
 * answer, which only pays off while drawing several of one thing.
 *
 * The label actions are the Detections grid's, with one scope rule: R,
 * X, U and the saved labels 1 to 5 act on the selected box, and on every
 * visible box when none is selected. Click a box or Tab through them to
 * select one; the buttons say which scope they are in. M relabels every
 * box to the picture's most common label, whatever is selected. Cmd+Z
 * reverts the last label change, as in the grid; a file verify (boxes
 * deleted) and a drawn box (no original label to go back to) are not
 * undoable. B hides the boxes to see what is under them.
 *
 * The file deliberately stays put after a change. An earlier version
 * refetched the list immediately, so a file that stopped matching the
 * filter left the list and took the modal with it, which meant a box
 * could never be moved, resized or relabelled. The list is refreshed
 * when the modal closes instead.
 *
 * Signing off the last file on a page does not close it either. The tab
 * fetches the next batch and the run carries on, so 500 files is one
 * pass rather than ten passes of 48 with a reopen between each.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Ban,
  Bookmark,
  Check,
  CheckCheck,
  ChevronDown,
  CircleHelp,
  Eye,
  EyeOff,
  Loader2,
  Plus,
  SquareDashed,
  Tag,
  Undo2,
  X,
} from "lucide-react";
import { toast } from "sonner";

import { detectionsApi } from "../../api/detections";
import { filesApi } from "../../api/files";
import { projectsApi } from "../../api/projects";
import { formatCameraDate, formatCameraTime } from "../../lib/datetime";
import { shouldDrawBbox } from "../../lib/detection-utils";
import { basename } from "../../lib/path-utils";
import { useLabelOptions, type LabelOption } from "../../hooks/useLabelOptions";
import { useShortcutLabels } from "../../hooks/useShortcutLabels";
import { Button } from "../ui/button";
import { AnnotationCanvas } from "./AnnotationCanvas";
import { labelMajority } from "./label-majority";
import { LabelPicker, type PinnedOption } from "./LabelPicker";
import { DetailCard, VerifyDetailShell } from "./VerifyDetailShell";
import { FileLocation } from "./FileLocation";
import { UNDO_KBD } from "./shortcuts";
import type { LabelsFileItem } from "../../api/types";

interface FileDetailModalProps {
  projectId: string;
  /** The page currently on screen; navigation stays inside it. */
  items: LabelsFileItem[];
  index: number | null;
  onIndexChange: (next: number) => void;
  onClose: () => void;
  /** Every file on this page has a verdict. The tab fetches what comes
   *  next and either points us at a new file or closes us. */
  onExhausted: () => void;
  /** True while that fetch is in flight. Nothing here reads `items`
   *  meanwhile, which is what keeps the list and the index into it from
   *  disagreeing while the list is being replaced. */
  loadingMore?: boolean;
  /** Something about this file changed. Fires immediately so the
   *  progress bar keeps up; the grid itself waits for the close. */
  onChanged: () => void;
}

/** The keyboard hint on a button. */
function Kbd({ children, onPrimary }: { children: string; onPrimary?: boolean }) {
  return (
    <kbd
      className={
        onPrimary
          ? "ml-1.5 text-[10px] font-sans text-primary-foreground/60 border border-primary-foreground/30 rounded px-1 py-0.5 shadow-[0_1px_0_0_rgba(255,255,255,0.1)] leading-none"
          : "ml-1.5 text-[10px] font-sans text-muted-foreground/60 border border-border/60 rounded px-1 py-0.5 shadow-[0_1px_0_0_rgba(0,0,0,0.08)] leading-none"
      }
    >
      {children}
    </kbd>
  );
}

export function FileDetailModal({
  projectId,
  items,
  index,
  onIndexChange,
  onClose,
  onExhausted,
  loadingMore,
  onChanged,
}: FileDetailModalProps) {
  const queryClient = useQueryClient();
  const [drawMode, setDrawMode] = useState(false);
  const [boxesHidden, setBoxesHidden] = useState(false);
  const [selectedDetectionId, setSelectedDetectionId] = useState<string | null>(
    null,
  );
  // The boxes the relabel search is open for. A list, because with no
  // box selected the search names every box on the picture.
  const [relabelTargets, setRelabelTargets] = useState<string[] | null>(null);
  // The species a newly drawn box gets. Null means an unnamed animal,
  // and the search opens on every new box instead.
  const [activeLabel, setActiveLabel] = useState<LabelOption | null>(null);
  const [speciesPickerOpen, setSpeciesPickerOpen] = useState(false);
  // The quick-label slot whose species is being chosen (the chevron
  // half of a slot's split button, or the "Add a quick label" row).
  const [editSlot, setEditSlot] = useState<number | null>(null);

  // The files this viewer has signed off. The grid deliberately does
  // not refetch while the viewer is open (see `FilesTab`), so `items`
  // still calls them unverified, and "next unverified" would walk
  // straight back onto files that are already done. Cleared on close,
  // which is when the grid refetches anyway.
  const verifiedHere = useRef<Set<string>>(new Set());

  // Undo, the grid's way: a stack of the ids each label action touched,
  // reverted through `bulk-revert-to-original`. Per file: an undo that
  // reached into a file no longer on screen would change something the
  // person cannot see.
  const [undoStack, setUndoStack] = useState<string[][]>([]);

  const item = index === null ? undefined : items[index];

  // Draw mode, the selection and the undo history are per-file, not
  // sticky: leaving draw mode on while paging would put the next Enter
  // on a crosshair. Reset during render rather than in an effect, so the
  // canvas never paints one frame in the wrong mode. Hiding the boxes is
  // a way of looking, so it stays.
  const [stateFor, setStateFor] = useState(item?.id);
  if (item?.id !== stateFor) {
    setStateFor(item?.id);
    setDrawMode(false);
    setSelectedDetectionId(null);
    setRelabelTargets(null);
    setUndoStack([]);
  }
  // Closed: forget what this run signed off. An effect, not part of the
  // render reset above, because a ref must not be touched during render.
  useEffect(() => {
    if (!item) verifiedHere.current.clear();
  }, [item]);

  const { data: project } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => projectsApi.get(projectId),
  });
  const { options: labelOptions, isLoading: labelOptionsLoading } =
    useLabelOptions(project?.classification_model_id ?? null, projectId);

  // The 1 to 5 slots, the same project-wide ones the Detections grid
  // uses; set or changed from either place.
  const { shortcutLabels, updateShortcutLabels } = useShortcutLabels(projectId);
  const pinnedOptions = useMemo<PinnedOption[]>(
    () =>
      Object.entries(shortcutLabels).map(([k, option]) => ({
        key: Number(k),
        option,
      })),
    [shortcutLabels],
  );

  const { data: file } = useQuery({
    queryKey: ["file", item?.id],
    queryFn: () => filesApi.get(item!.id),
    enabled: !!item,
  });

  // The list row is what the grid last fetched; the file query is live.
  const isVerified = file?.verified ?? item?.verified ?? false;

  // The boxes the canvas draws, left to right by box centre (top to
  // bottom for ties), the way a person reads the photo. Storage order
  // was the detector's confidence order, which jumps around the frame.
  // Tab walks this list, and "all boxes" means this list.
  const visibleBoxes = useMemo(() => {
    if (!file) return [];
    const threshold = project?.counting_threshold ?? 0;
    return file.detections
      .filter((d) => shouldDrawBbox(d, file, threshold))
      .sort(
        (a, b) =>
          a.bbox_x + a.bbox_width / 2 - (b.bbox_x + b.bbox_width / 2) ||
          a.bbox_y + a.bbox_height / 2 - (b.bbox_y + b.bbox_height / 2),
      );
  }, [file, project?.counting_threshold]);

  // The one scope rule: the selected box, else every visible box.
  const targetIds = useMemo(
    () =>
      selectedDetectionId ? [selectedDetectionId] : visibleBoxes.map((d) => d.id),
    [selectedDetectionId, visibleBoxes],
  );
  // The wording rule for the box actions: the button says what it does
  // (verb + result), the card's caption says what they act on, and the
  // tooltips spell out the full sentence with the scope. Matches the
  // Detections bar's texts. Only the majority button carries a scope
  // word ("all"): it is the one action that ignores the selection, and
  // without it it would read like a quick label while acting on more.
  const scope =
    selectedDetectionId || visibleBoxes.length === 1
      ? "the box"
      : visibleBoxes.length === 0
        ? "the boxes"
        : `all ${visibleBoxes.length} boxes`;
  const majority = useMemo(() => labelMajority(visibleBoxes), [visibleBoxes]);

  const go = useCallback(
    (delta: number) => {
      if (index === null) return;
      const next = index + delta;
      if (next < 0 || next >= items.length) return;
      onIndexChange(next);
    },
    [index, items.length, onIndexChange],
  );

  /** The next file after this one that still needs a verdict. Stays
   *  inside the page, because that is all this component is given; when
   *  the page runs out it asks the tab for the next batch rather than
   *  shutting, so a long run of files is one uninterrupted pass. */
  const advance = useCallback(() => {
    if (index === null) return;
    for (let i = index + 1; i < items.length; i++) {
      const next = items[i];
      if (!next.verified && !verifiedHere.current.has(next.id)) {
        onIndexChange(i);
        return;
      }
    }
    onExhausted();
  }, [index, items, onIndexChange, onExhausted]);

  const { mutate: setVerified, isPending: verifying } = useMutation({
    mutationFn: ({ id, verified }: { id: string; verified: boolean }) =>
      filesApi.update(id, { verified }),
    onSuccess: (_result, { id, verified }) => {
      if (verified) verifiedHere.current.add(id);
      else verifiedHere.current.delete(id);
      onChanged();
      queryClient.invalidateQueries({ queryKey: ["file", id] });
    },
    onError: (err: Error) => toast.error(err.message),
  });

  /** Enter, and the primary button while the file is unverified: "the
   *  boxes I see are all there is, done with this one". Sent even when
   *  the file already reads verified: the rollup flips that flag once
   *  every visible box is verified, without touching the weak boxes
   *  underneath, and the verify is idempotent, so this is what makes
   *  the sentence true for such a file too. Unverifying stays on the
   *  button, where it cannot be hit by a person leaning on Enter. Same
   *  rule as the Detections viewer's `handleVerifyAndAdvance`. */
  const verifyAndAdvance = useCallback(() => {
    if (!item) return;
    setVerified({ id: item.id, verified: true });
    advance();
  }, [item, setVerified, advance]);

  // Warm the next file while this one is being looked at. Measured on a
  // real file: 1.47 MB for the picture and a 698-byte row, and the row
  // is what the canvas waits on, so without this every step of a run
  // pays for both from cold. The image is a plain <img> rather than a
  // query, but the endpoint sends `max-age=86400, immutable`, so an
  // `Image()` here is all the warming it needs. Same idea as the Counts
  // viewer prefetching the next event and its filmstrip.
  const nextItem = index === null ? undefined : items[index + 1];
  useEffect(() => {
    if (!nextItem) return;
    // Relative, exactly as `AnnotationCanvas` builds it. An absolute
    // form would be a different cache entry and warm nothing.
    new Image().src = `/api/files/${nextItem.id}/image`;
    queryClient.prefetchQuery({
      queryKey: ["file", nextItem.id],
      queryFn: () => filesApi.get(nextItem.id),
    });
  }, [nextItem, queryClient]);

  /** A box was drawn, moved, resized or relabelled. Refresh this file
   *  and the counts, but leave the list alone so the user can keep
   *  working on the box they just made. */
  const handleCanvasChange = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["file", item?.id] });
    onChanged();
  }, [item?.id, onChanged, queryClient]);

  /** A box the user has just drawn. A drawn box is worth little without
   *  a species, and the pill that opens the picker is small and easy to
   *  miss, so the search opens on it by itself. Unless the Default
   *  label card already says what new boxes are: then the answer is
   *  given and asking again would only be in the way. */
  const handleCreated = useCallback(
    (detectionId: string) => {
      if (activeLabel) return;
      setSelectedDetectionId(detectionId);
      setRelabelTargets([detectionId]);
    },
    [activeLabel],
  );

  /** Every label action goes through here: the grid's `bulk-relabel`
   *  (which verifies), then the undo stack, then a refresh. The
   *  selection is cleared because the boxes it pointed at may no longer
   *  be drawn (a box marked false leaves the canvas). */
  const applyLabel = useCallback(
    (ids: string[], label: string | null, category: string | undefined) => {
      if (ids.length === 0) return;
      detectionsApi
        .bulkRelabel(ids, label, category)
        .then(() => {
          setUndoStack((s) => [...s, ids]);
          setSelectedDetectionId(null);
          handleCanvasChange();
        })
        .catch((err: Error) => toast.error(err.message));
    },
    [handleCanvasChange],
  );

  const markTargetsFalse = useCallback(
    () => applyLabel(targetIds, "false detection", undefined),
    [applyLabel, targetIds],
  );
  const markTargetsUnknown = useCallback(
    () => applyLabel(targetIds, "unknown", undefined),
    [applyLabel, targetIds],
  );
  const relabelTargetsNow = useCallback(() => {
    if (targetIds.length) setRelabelTargets(targetIds);
  }, [targetIds]);
  /** M: every visible box takes the picture's most common label. Ties
   *  resolve as in the grid, to the label counted first. */
  const matchMajority = useCallback(() => {
    if (!majority || visibleBoxes.length < 2) return;
    applyLabel(
      visibleBoxes.map((d) => d.id),
      majority.label,
      majority.category,
    );
  }, [applyLabel, majority, visibleBoxes]);
  const applyShortcut = useCallback(
    (slot: number) => {
      const option = shortcutLabels[slot];
      if (!option) return;
      applyLabel(targetIds, option.label, option.category);
    },
    [applyLabel, shortcutLabels, targetIds],
  );

  const handleUndo = useCallback(() => {
    if (undoStack.length === 0) return;
    const ids = undoStack[undoStack.length - 1];
    detectionsApi
      .bulkRevertToOriginal(ids)
      .then(() => {
        setUndoStack((s) => s.slice(0, -1));
        handleCanvasChange();
      })
      .catch((err: Error) => toast.error(err.message));
  }, [undoStack, handleCanvasChange]);

  /** Tab / Shift+Tab: move the selection through `visibleBoxes`, with
   *  "none" as one more stop so a full cycle also clears it. Keyboard-only
   *  review of a photo with several animals needs a way to say which box
   *  you mean; without this that was a mouse click. */
  const cycleSelection = useCallback(
    (delta: 1 | -1) => {
      const ids = visibleBoxes.map((d) => d.id);
      if (ids.length === 0) return;
      // Index -1 is "nothing selected".
      const at = selectedDetectionId ? ids.indexOf(selectedDetectionId) : -1;
      const stops = ids.length + 1;
      const next = (at + 1 + delta + stops) % stops - 1;
      setSelectedDetectionId(next < 0 ? null : ids[next]);
    },
    [visibleBoxes, selectedDetectionId],
  );

  const captured = item?.captured_at_local;

  // Same verbs as the Detections grid, so nothing new to learn. The
  // canvas binds no keys of its own; this is the one place they live,
  // so a key cannot fire twice.
  useEffect(() => {
    // Nothing to act on while the next batch is on its way, and acting
    // on the file we are leaving would ask for that batch twice.
    if (!item || loadingMore) return;
    const onKey = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }
      const key = e.key.toLowerCase();
      if (e.key === "ArrowRight") { e.preventDefault(); go(1); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); go(-1); }
      else if (e.key === "Enter" && !drawMode) {
        e.preventDefault();
        verifyAndAdvance();
      } else if (key === "z" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        handleUndo();
      } else if (key === "d") {
        e.preventDefault();
        setDrawMode((v) => !v);
      } else if (key === "b") {
        e.preventDefault();
        setBoxesHidden((v) => !v);
      } else if (e.key === "Tab") {
        // Taken from the dialog's focus order on purpose: the buttons
        // all have keys of their own, and Tab is the one key a person
        // expects to step through things on screen.
        e.preventDefault();
        cycleSelection(e.shiftKey ? -1 : 1);
      } else if (key === "x") {
        e.preventDefault();
        markTargetsFalse();
      } else if (key === "u") {
        e.preventDefault();
        markTargetsUnknown();
      } else if (key === "r") {
        e.preventDefault();
        relabelTargetsNow();
      } else if (key === "m") {
        e.preventDefault();
        matchMajority();
      } else if (e.key >= "1" && e.key <= "5" && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        applyShortcut(parseInt(e.key));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    item,
    go,
    drawMode,
    verifyAndAdvance,
    loadingMore,
    cycleSelection,
    markTargetsFalse,
    markTargetsUnknown,
    relabelTargetsNow,
    matchMajority,
    applyShortcut,
    handleUndo,
  ]);

  // The page ran out and the next batch is being fetched. Hold the
  // frame rather than letting the dialog blink shut and reopen, and
  // read nothing off `items`, which is about to be replaced.
  if (loadingMore) {
    return (
      <VerifyDetailShell
        open
        onOpenChange={(open) => {
          if (!open) onClose();
        }}
        title="Loading the next files"
        width="95vw"
        height="92vh"
        image={
          <div className="flex items-center gap-2 text-white/60">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-sm">Loading the next files</span>
          </div>
        }
        details={null}
        actions={null}
      />
    );
  }

  if (!item) return null;

  const noBoxes = visibleBoxes.length === 0;
  const freeSlot = [1, 2, 3, 4, 5].find((n) => !shortcutLabels[n]);

  return (
    <VerifyDetailShell
      open
      onOpenChange={(open) => { if (!open) onClose(); }}
      title={basename(item.file_path)}
      width="95vw"
      height="92vh"
      position={`${index! + 1} of ${items.length}`}
      onNavigate={(direction) => {
        if (direction === "prev") go(-1);
        else if (direction === "nextUnverified") advance();
        else go(1);
      }}
      // No tool strip. Everything you can do sits in one column on the
      // right, in words, because an unlabelled icon beside the picture
      // is not found by someone who does not already know it is there.
      image={
        file ? (
          <AnnotationCanvas
            file={file}
            detectionThreshold={project?.counting_threshold ?? 0}
            selectedDetectionId={selectedDetectionId}
            onSelectDetection={setSelectedDetectionId}
            onRequestRelabel={(id) => setRelabelTargets([id])}
            drawMode={drawMode}
            onDrawModeChange={setDrawMode}
            onMutated={handleCanvasChange}
            onCreated={handleCreated}
            boxesHidden={boxesHidden}
            defaultCategory={activeLabel?.category ?? "animal"}
            defaultLabel={activeLabel?.label ?? undefined}
          />
        ) : (
          <div className="text-white/50">Loading...</div>
        )
      }
      details={
        <>
          <DetailCard title={item.file_type === "video" ? "Video" : "Image"}>
            <div className="space-y-0.5 text-xs text-muted-foreground">
              <FileLocation filePath={item.file_path} />
              <div>
                {captured ? (
                  <>
                    {formatCameraDate(captured, { day: "numeric", month: "short", year: "numeric" }, "en-GB")}{" "}
                    {formatCameraTime(captured, { hour: "2-digit", minute: "2-digit" }, "en-GB")}
                  </>
                ) : (
                  "No capture time"
                )}
              </div>
              {/* Say what is on screen, because it is not the whole clip
                  and nothing else here admits that.

                  It leads with the detector on purpose. Saying only that
                  one frame is kept reads as "the AI looked at one frame
                  and missed the rest", which is the opposite of the
                  truth: MegaDetector runs over every sampled frame and
                  the single frame is chosen afterwards, purely because
                  it is the only one written to disk as a JPEG.

                  Deliberately not an offer to watch the video: a box can
                  only be saved on the frame the app kept
                  (`AnnotationCanvas` stamps `best_frame_number` on every
                  box it creates), so an animal spotted on any other
                  second is something the person could see and never
                  record. Worded to hold in every case, including a clip
                  where nothing was found at all and the kept frame is
                  simply the middle one. */}
              {item.file_type === "video" && (
                <div className="pt-1 text-muted-foreground/80">
                  The AI checked the whole clip. This is the one frame
                  AddaxAI kept, so you are judging this frame, and a box
                  you draw is saved on it.
                </div>
              )}
            </div>
          </DetailCard>

          {/* Everything you can do to the picture, in one card so the
              column reads as cards throughout. In the Detections bar's
              order, with its icons and keys. The scope rule in words:
              the selected box, else every box; click a box or Tab
              through them to narrow it. Verify and Undo deliberately
              stay out: the primary action and its escape hatch live at
              the bottom, in a fixed spot. */}
          <DetailCard title="Actions">
            {/* Carries the scope so the buttons do not have to: what
                they act on, and how to narrow it. */}
            <p className="mb-2 text-xs text-muted-foreground">
              These act on the box you select, with a click or Tab.
              None selected means all boxes.
            </p>
            <div className="space-y-1.5">
              <Button
                variant="outline"
                size="sm"
                className="w-full justify-center"
                onClick={() => setDrawMode((v) => !v)}
              >
                <SquareDashed className="h-4 w-4 mr-1" />
                {drawMode ? "Stop drawing" : "Draw a box"}
                <Kbd>D</Kbd>
              </Button>

              <Button
                variant="outline"
                size="sm"
                className="w-full justify-center"
                onClick={() => setBoxesHidden((v) => !v)}
              >
                {boxesHidden ? (
                  <Eye className="h-4 w-4 mr-1" />
                ) : (
                  <EyeOff className="h-4 w-4 mr-1" />
                )}
                {boxesHidden ? "Show boxes" : "Hide boxes"}
                <Kbd>B</Kbd>
              </Button>

              <Button
                variant="outline"
                size="sm"
                className="w-full justify-center"
                disabled={noBoxes}
                onClick={markTargetsFalse}
                title={`Mark ${scope} as false detections and verify`}
              >
                <Ban className="h-4 w-4 mr-1" />
                Mark false
                <Kbd>X</Kbd>
              </Button>

              <Button
                variant="outline"
                size="sm"
                className="w-full justify-center"
                disabled={noBoxes}
                onClick={markTargetsUnknown}
                title={`Mark ${scope} as unidentifiable animals and verify`}
              >
                <CircleHelp className="h-4 w-4 mr-1" />
                Mark unknown
                <Kbd>U</Kbd>
              </Button>

              {/* Whole picture only, whatever is selected: a majority is
                  a statement about the set. Hidden when it cannot mean
                  anything (one box, or no labels). */}
              {majority && visibleBoxes.length >= 2 && (
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full justify-center"
                  onClick={matchMajority}
                  title={`Relabel all ${visibleBoxes.length} boxes to ${majority.common_name ?? majority.label} and verify`}
                >
                  <CheckCheck className="h-4 w-4 mr-1 shrink-0" />
                  <span className="truncate">
                    Set all to {majority.common_name ?? majority.label}
                  </span>
                  <Kbd>M</Kbd>
                </Button>
              )}

              <Button
                variant="outline"
                size="sm"
                className="w-full justify-center"
                disabled={noBoxes}
                onClick={relabelTargetsNow}
                title={`Relabel ${scope} and verify`}
              >
                <Tag className="h-4 w-4 mr-1" />
                Relabel
                <Kbd>R</Kbd>
              </Button>

              {/* The saved labels, the same 1 to 5 slots as the
                  Detections grid, as split buttons: the body applies the
                  slot's label (the key does the same), the chevron
                  segment opens the label search that changes it. Only
                  the set slots show, plus one row to fill the next free
                  slot; changes are saved on the project, so both tabs
                  see them. */}
              {[1, 2, 3, 4, 5]
                .filter((n) => shortcutLabels[n])
                .map((n) => (
                  <div key={n} className="flex w-full">
                    <Button
                      variant="outline"
                      size="sm"
                      className="min-w-0 flex-1 justify-center rounded-r-none border-r-0"
                      disabled={noBoxes}
                      onClick={() => applyShortcut(n)}
                      title={`Relabel ${scope} to ${shortcutLabels[n].displayName} and verify`}
                    >
                      <Bookmark className="h-4 w-4 mr-1" />
                      <span className="truncate">
                        Set to {shortcutLabels[n].displayName}
                      </span>
                      <Kbd>{String(n)}</Kbd>
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-9 shrink-0 rounded-l-none px-0 text-muted-foreground"
                      onClick={() => setEditSlot(n)}
                      title={`Choose the label for key ${n}`}
                    >
                      <ChevronDown className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                ))}
              {freeSlot !== undefined && (
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full justify-center text-muted-foreground"
                  onClick={() => setEditSlot(freeSlot)}
                  title={`Save a label on key ${freeSlot}`}
                >
                  <Plus className="h-4 w-4 mr-1" />
                  Add a quick label
                  <Kbd>{String(freeSlot)}</Kbd>
                </Button>
              )}
            </div>
          </DetailCard>

          {/* Sits with the other cards rather than among the buttons:
              it describes how drawing behaves, it is not a verdict on
              this file. "Ask me each time" is the default and the
              honest one. */}
          <DetailCard title="Default label">
            <p className="mb-2 text-xs text-muted-foreground">
              Pick a species and every box you draw takes it. Saves
              answering the label search each time.
            </p>
            <div className="flex h-9 w-full items-center rounded-md border border-input bg-background">
              <button
                type="button"
                className="flex min-w-0 flex-1 items-center gap-2 px-3 text-sm"
                onClick={() => setSpeciesPickerOpen(true)}
              >
                <Tag className="h-4 w-4 shrink-0 text-muted-foreground" />
                <span
                  className={`truncate ${activeLabel ? "" : "text-muted-foreground"}`}
                >
                  {activeLabel ? activeLabel.displayName : "Ask me each time"}
                </span>
              </button>
              {activeLabel ? (
                <button
                  type="button"
                  className="flex h-full items-center px-3 text-muted-foreground hover:text-foreground"
                  onClick={() => setActiveLabel(null)}
                  title="Go back to being asked each time"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              ) : (
                <ChevronDown className="mr-3 h-3.5 w-3.5 shrink-0 opacity-50" />
              )}
            </div>
          </DetailCard>
        </>
      }
      actions={
        <>
          <Button
            variant="outline"
            size="sm"
            className="w-full justify-center"
            disabled={undoStack.length === 0}
            onClick={handleUndo}
            title="Undo the last label change on this file"
          >
            <Undo2 className="h-4 w-4 mr-1" />
            Undo
            <Kbd>{UNDO_KBD}</Kbd>
          </Button>

          <Button
            className="w-full justify-center"
            size="sm"
            variant={isVerified ? "outline" : "default"}
            onClick={
              isVerified
                ? () => setVerified({ id: item.id, verified: false })
                : verifyAndAdvance
            }
            disabled={verifying}
          >
            <Check className="h-4 w-4 mr-1" />
            {isVerified ? "Unverify" : "Verify"}
            {!isVerified && <Kbd onPrimary>⏎</Kbd>}
          </Button>

          {/* Two searches, no triggers of their own. The first names the
              boxes in `relabelTargets`: opened by drawing one, by R, or
              by clicking a box's label on the canvas. The second sets
              the Default label card. Both use `headless`, which is the
              prop for exactly this. */}
          <LabelPicker
            headless
            value={null}
            onSelect={(option) => {
              const ids = relabelTargets;
              setRelabelTargets(null);
              if (ids) applyLabel(ids, option.label, option.category);
            }}
            options={labelOptions}
            isLoading={labelOptionsLoading}
            pinnedOptions={pinnedOptions}
            forceOpen={!!relabelTargets}
            onOpenChange={(open) => {
              if (!open) setRelabelTargets(null);
            }}
            projectId={projectId}
          />

          <LabelPicker
            headless
            value={activeLabel?.label ?? null}
            displayName={activeLabel?.displayName}
            onSelect={setActiveLabel}
            options={labelOptions}
            isLoading={labelOptionsLoading}
            pinnedOptions={pinnedOptions}
            forceOpen={speciesPickerOpen}
            onOpenChange={(open) => {
              if (!open) setSpeciesPickerOpen(false);
            }}
            projectId={projectId}
          />

          {/* And a third, for the slot being set or changed. */}
          <LabelPicker
            headless
            value={editSlot !== null ? shortcutLabels[editSlot]?.value ?? null : null}
            displayName={editSlot !== null ? shortcutLabels[editSlot]?.displayName : undefined}
            onSelect={(option) => {
              const n = editSlot;
              setEditSlot(null);
              if (n !== null) {
                updateShortcutLabels((prev) => ({ ...prev, [n]: option }));
              }
            }}
            options={labelOptions}
            isLoading={labelOptionsLoading}
            forceOpen={editSlot !== null}
            onOpenChange={(open) => {
              if (!open) setEditSlot(null);
            }}
            projectId={projectId}
          />
        </>
      }
    />
  );
}
