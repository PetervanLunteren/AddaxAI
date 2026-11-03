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
 * "every detection I can see is checked", and the backend rejects the
 * ones you could not see (marked "false detection", like the X key).
 * Drawing them would make that verdict about a threshold instead of
 * about the picture. For a photo that is its boxes; for a clip it is
 * every detection the AI followed through it, one track each, shown as
 * bars under the picture.
 *
 * What you can do. **Verify** (Enter) and **Mark correct** (V) are one
 * verb: they verify what is on screen, or what is selected, with the
 * label it already carries, so a clip the detector handled well is a
 * run of single keys rather than a relabel to the species it already
 * said. Signing a whole clip off is a button of its own, below, behind
 * a confirm that says the numbers (`SignOffDialog`).
 *
 * Boxes are drawn on photos only. Arm the crosshair with D or the
 * button, drag, and the species search opens on the new box by itself;
 * name a species in the Default label card and new boxes take that
 * instead. A clip has no drawing: a hand-drawn box could only land on
 * the frame on screen, which on an hour of footage is one frame in ten
 * thousand, so it could never be how an animal the detector missed gets
 * recorded. The Counts page is, by typing the number.
 *
 * The label actions are the Detections grid's, with one scope rule:
 * actions act on what you see. V, R, X, U and the saved labels 1 to 5
 * act on the selected detection, and with none selected on every
 * detection on screen (`onScreen`): a photo's boxes, or the animals on
 * the clip frame shown, each through all of its frames. A clip's other
 * two thousand tracks are never touched from a frame that shows 17 of
 * them (they were, until 2026-11-03: a V on the MaxN frame signed off
 * the whole drop). One rule decides what happens next (`settleAfter`):
 * when nothing in the file is left unchecked it is signed off and the
 * viewer moves on; otherwise the next animal still waiting, forward
 * in time from the frame on screen, is selected and shown, so Enter
 * after Enter walks a clip. Click a box, Tab through the boxes on screen, or
 * click a bar on a clip's timeline to select one; the buttons say which
 * scope they are in. M relabels every detection on screen to the most
 * common label among them, whatever is selected. Cmd+Z reverts the last
 * label change, as in the grid; a file sign-off (its own scope of
 * boxes) and a drawn box (no original label to go back to) are not
 * undoable.
 *
 * The left rail is the shared `ViewerToolRail`, the same one as the
 * Counts modal: brightness/contrast, hide boxes (B), flag (F), like,
 * download and open in file explorer, each a plain icon. One mental
 * model: the rail is how you look, the right column is what you
 * decide.
 *
 * A clip opens on its cover frame with the timeline of its tracks
 * under it (`ClipTimeline`). A bar click selects that track's card and
 * shows the frame it sits on, decoded on request; a click elsewhere on
 * the lane shows that frame. P or the play button swaps the still for
 * the real clip at the frame on screen, with each frame's boxes and the
 * same timeline (`VideoPlayer`, the Counts modal's player, which
 * honours B); P again pauses and brings the still back on the moment
 * the person stopped at, so what they act on is what they watched. The
 * keys stay live in both modes on a selected animal; with none selected
 * they wait for the still, the only thing a person can judge.
 * Download saves the annotated copy: the recorded boxed MP4 for a
 * playable video, the boxed still otherwise.
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
  Loader2,
  Play,
  Plus,
  SquareDashed,
  Tag,
  Undo2,
  X,
} from "lucide-react";
import { toast } from "sonner";

import { detectionsApi } from "../../api/detections";
import { fileQueryOptions, filesApi } from "../../api/files";
import { projectsApi } from "../../api/projects";
import { formatCameraDate, formatCameraTime } from "../../lib/datetime";
import {
  fileMaxNMarkers,
  frameBoxes,
  nearestSampledFrame,
  steppedFrame,
} from "../../lib/track-utils";
import { basename } from "../../lib/path-utils";
import { useLabelOptions, type LabelOption } from "../../hooks/useLabelOptions";
import { useShortcutLabels } from "../../hooks/useShortcutLabels";
import { Button } from "../ui/button";
import { AnnotationCanvas } from "./AnnotationCanvas";
import { VideoPlayer, isPlayableVideo } from "./VideoPlayer";
import {
  cardBoxes,
  representativeBox,
  uncheckedCards,
} from "../../lib/track-utils";
import { ClipTimeline } from "./TrackTimeline";
import { ViewerToolRail } from "./ViewerToolRail";
import { useFileTriage, useImageAdjust } from "./viewer-tools";
import { labelMajority } from "./label-majority";
import { LabelPicker, type PinnedOption } from "./LabelPicker";
import { DetailCard, VerifyDetailShell } from "./VerifyDetailShell";
import { SignOffDialog } from "./SignOffDialog";
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
  /** Opened from a track's card: show that track's frame with the
   *  track selected, so the detection the person was looking at is the
   *  one on screen. Ignored when the file has no such track. */
  openTrackId?: string | null;
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
  openTrackId = null,
}: FileDetailModalProps) {
  const queryClient = useQueryClient();
  const [drawMode, setDrawMode] = useState(false);
  const [boxesHidden, setBoxesHidden] = useState(false);
  // For a playable video: the still or the real clip. The Counts event
  // modal's pattern.
  const [viewMode, setViewMode] = useState<"frame" | "video">("frame");
  // The frame the still shows: null is the cover frame; a bar click or a
  // lane click sets another, decoded on request. Explicit state rather
  // than derived from the selection, because a lane click shows a frame
  // without selecting anything, and D must return to the cover without
  // clearing the selection (drawing happens on the cover).
  const [shownFrame, setShownFrame] = useState<number | null>(null);
  // One-shot: Download clicked from frame view mounts the player and
  // runs the annotated-video export once the clip is playable.
  const [pendingVideoExport, setPendingVideoExport] = useState(false);
  // Where the player should jump: the track picked on the timeline, or
  // the track this viewer was opened for.
  const [seekRequest, setSeekRequest] = useState<{ frame: number; nonce: number; play?: boolean } | null>(null);
  // The timeline's zoom, kept here so it survives the switch between the
  // still and the player, which mount the timeline afresh.
  const [timelineZoom, setTimelineZoom] = useState(0);
  // The confirm behind "Verify all" on a clip.
  const [signOffOpen, setSignOffOpen] = useState(false);
  // The player fills this: pause and say where. P in video mode calls
  // it, so the still comes back on the moment the person stopped at
  // rather than where play began.
  const pauseFnRef = useRef<(() => number) | null>(null);
  // The mounted surface's export: annotated PNG from the canvas,
  // recorded annotated MP4 from the player. Only one is mounted at a
  // time, so one ref serves both (as in `EventDetailModal`).
  const exportFnRef = useRef<(() => void) | null>(null);
  // The shared rail's state: brightness/contrast and the flag/like
  // writes (the F key below shares the same mutation).
  const { brightness, setBrightness, contrast, setContrast, imageFilter } =
    useImageAdjust();
  const triage = useFileTriage();
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
    setViewMode("frame");
    setShownFrame(null);
    setSeekRequest(null);
    setTimelineZoom(0);
    setPendingVideoExport(false);
    setSignOffOpen(false);
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

  // The full file: every box of every frame, for the canvas at any
  // frame, the player and the timeline. The tile that opened this holds
  // the cards only, so the first open of a file is one request.
  const { data: file } = useQuery({
    ...fileQueryOptions(item?.id ?? ""),
    enabled: !!item,
  });

  // The list row is what the grid last fetched; the file query is live.
  const isVerified = file?.verified ?? item?.verified ?? false;

  // The selected box's track, for the player's highlight. Selecting a
  // bar on the timeline selects that track's representative box, so
  // the usual actions reach the whole animal through the cascade.
  const selectedTrackId = useMemo(
    () => file?.detections.find((d) => d.id === selectedDetectionId)?.track_id ?? null,
    [file, selectedDetectionId],
  );
  const selectTrack = useCallback(
    (trackId: string) => {
      const track = file?.tracks.find((t) => t.id === trackId);
      const box = track && file ? representativeBox(file.detections, track) : undefined;
      setSelectedDetectionId(box?.id ?? null);
      if (track) setShownFrame(track.representative_frame_number);
    },
    [file],
  );
  // A ruler or lane click shows a frame and selects nothing, so the
  // scope caption never names a box that is not on screen. A moment
  // picked anywhere snaps to the nearest sampled frame
  // (`nearestSampledFrame`), the frames the bars make claims about.
  const showFrame = useCallback(
    (frame: number) => {
      setShownFrame(file ? nearestSampledFrame(file, frame) : frame);
      setSelectedDetectionId(null);
    },
    [file],
  );
  // Opened from a card: once the file is in, show that track. Once per
  // file and track, so paging away and back does not jump again.
  const openedFor = useRef<string | null>(null);
  useEffect(() => {
    if (!file || !openTrackId) return;
    const key = `${file.id}:${openTrackId}`;
    if (openedFor.current === key) return;
    openedFor.current = key;
    if (file.tracks.some((t) => t.id === openTrackId)) selectTrack(openTrackId);
  }, [file, openTrackId, selectTrack]);

  const playable = !!file && isPlayableVideo(file);
  // The cover is the stored picture; any other frame is decoded on
  // request, so the canvas asks for it by number only when it differs.
  const frameNumber =
    file && shownFrame != null && shownFrame !== file.best_frame_number ? shownFrame : undefined;
  /** Shift+Arrow and Shift+wheel: the previous or next sampled frame of
   *  a clip (`steppedFrame`, the Counts modal's step too), one per
   *  press, so a person can see what happened just before or after the
   *  moment on screen. The same pairing as the Counts modal, where the
   *  plain arrow is the big unit (events there, files here) and Shift
   *  makes it the small one. A stepped frame is a picked moment like a
   *  ruler click: shown, nothing selected. A photo has one frame, so
   *  this is nothing there. */
  const stepFrame = useCallback(
    (delta: number) => {
      if (!file || file.file_type !== "video") return;
      const from = shownFrame ?? file.best_frame_number ?? 0;
      showFrame(steppedFrame(file, from, delta > 0 ? 1 : -1));
    },
    [file, shownFrame, showFrame],
  );
  // Watch the clip from the frame on screen.
  const watch = useCallback(() => {
    if (shownFrame != null) {
      setSeekRequest({ frame: shownFrame, nonce: Date.now(), play: true });
    }
    setViewMode("video");
  }, [shownFrame]);
  // And back: the still on the paused moment, snapped to the nearest
  // sampled frame (`nearestSampledFrame`), since the detector sampled a
  // few frames per second and a paused frame is rarely one of them.
  // The selection is kept: P is not a lane click.
  const backToStill = useCallback(() => {
    const frame = pauseFnRef.current?.();
    if (frame != null && file) setShownFrame(nearestSampledFrame(file, frame));
    setViewMode("frame");
  }, [file]);
  /** Boxes are drawn on photos only. A clip is reviewed animal by
   *  animal, and a box drawn by hand could only ever land on the one
   *  frame on screen: on an hour of footage at 3 fps that is one frame
   *  in ten thousand, so it can never be the way an animal the detector
   *  missed gets recorded. The Counts page is, by typing the number. */
  const canDraw = file != null && file.file_type !== "video";

  /** Download: for a playable video always the annotated MP4 (mount the
   *  player if needed and record once it can play); for anything else
   *  the annotated still PNG from the canvas. `EventDetailModal`'s
   *  `handleDownload`, verbatim. */
  const handleDownload = useCallback(() => {
    if (playable) {
      setViewMode("video");
      setPendingVideoExport(true);
    } else {
      exportFnRef.current?.();
    }
  }, [playable]);

  // The boxes the canvas draws on the frame on screen, left to right by
  // box centre (top to bottom for ties), the way a person reads the
  // picture. Storage order was the detector's confidence order, which
  // jumps around the frame. Tab walks this list.
  const visibleBoxes = useMemo(() => {
    if (!file) return [];
    const threshold = project?.counting_threshold ?? 0;
    return frameBoxes(file, threshold, frameNumber)
      .sort(
        (a, b) =>
          a.bbox_x + a.bbox_width / 2 - (b.bbox_x + b.bbox_width / 2) ||
          a.bbox_y + a.bbox_height / 2 - (b.bbox_y + b.bbox_height / 2),
      );
  }, [file, project?.counting_threshold, frameNumber]);
  // Every detection in the file: a photo's boxes, a clip's tracks
  // through their cards. What "none selected" acts on.
  const cards = useMemo(
    () => (file ? cardBoxes(file, project?.counting_threshold ?? 0) : []),
    [file, project?.counting_threshold],
  );

  // What you see, the scope with nothing selected: a photo's boxes, or
  // the animals on the clip frame shown (the tracks whose boxes the
  // canvas draws, through their cards, verified or not, like a photo's
  // boxes). After MaxN this is the count's frame, and verifying it is
  // the count check in one press. Empty in video mode: the still is
  // what a person judges, and P brings it back on the paused moment.
  const onScreen = useMemo(() => {
    if (!file) return [];
    if (file.file_type !== "video") return cards;
    if (viewMode === "video") return [];
    const tracks = new Set(visibleBoxes.map((d) => d.track_id));
    return cards.filter((d) => d.track_id && tracks.has(d.track_id));
  }, [file, viewMode, visibleBoxes, cards]);

  // The one scope rule: the selected detection, else what you see.
  const targetIds = useMemo(
    () => (selectedDetectionId ? [selectedDetectionId] : onScreen.map((d) => d.id)),
    [selectedDetectionId, onScreen],
  );
  // The wording rule for the box actions: the button says what it does
  // (verb + result), the card's caption says what they act on, and the
  // tooltips spell out the full sentence with the scope. Matches the
  // Detections bar's texts. Only the majority button carries the scope:
  // it is the one action that ignores the selection, and without it it
  // would read like a quick label while acting on more. A clip's unit
  // is the detection (one track each), a photo's the box.
  const unit = file?.file_type === "video" ? "detection" : "box";
  const units = file?.file_type === "video" ? "detections" : "boxes";
  const noun = file?.file_type === "video" ? "clip" : "photo";
  // What you see, in words: "all 12 boxes", "the 17 detections on this frame".
  const seen =
    file?.file_type === "video"
      ? `the ${onScreen.length} ${units} on this frame`
      : `all ${onScreen.length} ${units}`;
  const scope =
    selectedDetectionId || onScreen.length === 1
      ? `the ${unit}`
      : onScreen.length === 0
        ? `the ${units}`
        : seen;
  // The primary button names what one press commits to, because it
  // sits under a column of actions that read "Verify" too. A photo's is
  // its boxes, the whole picture; a clip's is the frame shown, the one
  // number a person can check. The whole clip is the outline button
  // under it, behind the confirm.
  const verifyLabel =
    cards.length === 0
      ? `Verify empty ${noun}`
      : file?.file_type !== "video"
        ? cards.length === 1
          ? `Verify 1 ${unit}`
          : `Verify all ${cards.length} ${units}`
        : onScreen.length === 0
          ? "Verify this frame"
          : `Verify the ${onScreen.length} on this frame`;
  const majority = useMemo(() => labelMajority(onScreen), [onScreen]);
  // How many detections a whole-clip sign-off would verify unseen.
  const unchecked = useMemo(
    () => (file ? uncheckedCards(file, project?.counting_threshold ?? 0).length : 0),
    [file, project?.counting_threshold],
  );

  // The clip's MaxN frame per species, for the timeline's markers and
  // its MaxN buttons: the file's own number, the one its tile shows.
  const markers = useMemo(
    () => (file && file.file_type === "video"
      ? fileMaxNMarkers(file)
      : []),
    [file],
  );

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

  /** Enter, and the primary button while the file is unverified: "every
   *  detection I see is checked, done with this one". Sent even when
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
    queryClient.prefetchQuery(fileQueryOptions(nextItem.id));
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

  /** The card still waiting after a verdict on `ids`, undefined when
   *  nothing else waits. Read before the verdict lands.
   *
   *  A clip walks forward in time from the frame on screen, by the
   *  frame each animal is shown on (its representative frame): the
   *  unchecked animal whose frame comes first at or after the one on
   *  screen, whatever its species, the ones just acted on left out, and
   *  the earliest remaining one only when nothing lies ahead. So the
   *  picture only ever moves forward until it wraps. Ordering by the
   *  bar's start instead (an hour on 2026-11-03) picked long-lived
   *  animals that had started earlier, and V V V on a shoal walked the
   *  picture backwards. Anchored on the frame, not on a place in a
   *  list, so a verdict on the frame, on a selected animal, or on one
   *  already checked (Backspace on a fish of a frame just verified)
   *  all move on from where the person is. Until 2026-11-03 the walk
   *  was species by species and started over at the front whenever
   *  the acted animal was not in the list, which on a 74 minute drop
   *  threw the picture back to the first fish. A photo walks its boxes
   *  in list order, wrapping round. */
  const nextWaitingAfter = useCallback(
    (ids: string[]) => {
      const threshold = project?.counting_threshold ?? 0;
      const all = file ? uncheckedCards(file, threshold) : [];
      const done = new Set(ids);
      const pending = all.filter((d) => !done.has(d.id));
      if (!file || pending.length === 0) return undefined;
      if (file.file_type !== "video") {
        const at = all.findIndex((d) => done.has(d.id));
        return pending[Math.max(at, 0) % pending.length];
      }
      const frameOf = new Map(file.tracks.map((t) => [t.id, t.representative_frame_number]));
      const shownOn = (d: { track_id: string | null }) =>
        d.track_id ? (frameOf.get(d.track_id) ?? 0) : 0;
      const anchor = shownFrame ?? file.best_frame_number ?? 0;
      const inTime = [...pending].sort(
        (a, b) => shownOn(a) - shownOn(b) || a.id.localeCompare(b.id),
      );
      return inTime.find((d) => shownOn(d) >= anchor) ?? inTime[0];
    },
    [file, project?.counting_threshold, shownFrame],
  );

  /** The frame the next verdict will land on, asked for as soon as the
   *  current picture is on screen, so a run of V V V does not wait at
   *  each step: the walk is deterministic (`nextWaitingAfter`), every
   *  frame is cut from the clip on request, and on an external drive
   *  that took 17 seconds a frame (SLiSN_016, 2026-11-03). The backend
   *  keeps the decoded frame and the browser the JPEG, so the step
   *  itself then costs nothing. Only after the load, never alongside
   *  it: two decodes on the same slow drive slow the one on screen. */
  const warmNextFrame = useCallback(() => {
    if (!file || file.file_type !== "video") return;
    const next = nextWaitingAfter(targetIds);
    const track = next?.track_id ? file.tracks.find((t) => t.id === next.track_id) : undefined;
    if (!track || track.representative_frame_number === file.best_frame_number) return;
    // Relative and in this form, exactly as `AnnotationCanvas` builds
    // it, or the browser cache entry would not be the one it reads.
    new Image().src = `/api/files/${file.id}/image?frame=${track.representative_frame_number}`;
  }, [file, targetIds, nextWaitingAfter]);

  /** Select a card, or nothing. A clip's card goes through `selectTrack`,
   *  so the picture moves to that animal's own frame exactly as a bar
   *  click does. */
  const selectCard = useCallback(
    (card: { id: string; track_id: string | null } | undefined) => {
      if (card?.track_id) selectTrack(card.track_id);
      else setSelectedDetectionId(card?.id ?? null);
    },
    [selectTrack],
  );

  /** What follows a verdict on `ids`, decided before it lands so the
   *  lists still hold the cards it touches. One rule for every verdict:
   *  the viewer moves to the next animal still waiting
   *  (`nextWaitingAfter`) and stays in the mode the verdict came from.
   *  A verdict on a selected animal selects the next one, so V V V
   *  walks the clip animal by animal, the step the grid and the card
   *  modal take. A verdict on the frame, nothing selected, shows the
   *  next animal's frame and selects nothing, so Enter Enter Enter
   *  walks it frame by frame with the outlines on the lane saying what
   *  the next Enter covers; selecting the next animal here (until
   *  2026-11-03) turned a frame pass into a bar pass after one press.
   *  Nothing left: the file is signed off and the viewer moves on,
   *  exactly as Enter on a photo always did. Not a courtesy: the
   *  backend's rollup flips `File.verified` the moment the last card
   *  is verified, without rejecting the weak boxes underneath, so
   *  without the sign-off here a V on the last animal left a file that
   *  read verified, was not, and stayed on screen with an "Unverify"
   *  button nobody had asked for. */
  const settleAfter = useCallback(
    (ids: string[]) => {
      const next = nextWaitingAfter(ids);
      if (!next) return verifyAndAdvance;
      const fromSelection = selectedDetectionId != null;
      return () => {
        if (fromSelection) selectCard(next);
        else if (next.track_id) {
          const track = file?.tracks.find((t) => t.id === next.track_id);
          if (track) showFrame(track.representative_frame_number);
        }
        handleCanvasChange();
      };
    },
    [
      nextWaitingAfter,
      verifyAndAdvance,
      selectedDetectionId,
      selectCard,
      file,
      showFrame,
      handleCanvasChange,
    ],
  );

  /** "The AI got these right": verify `ids` with the labels they carry,
   *  no search and no dialog. Enter sends what is on screen, V what is
   *  targeted. `bulk-verify` expands to the whole track on the way in,
   *  so one press stands for every frame the animal was followed
   *  through, like the other verdicts. Agreeing is the most common
   *  verdict of all on a clip the detector handled well, so a drop is
   *  V V V R V X. No ids and nothing left (an empty file) is the
   *  sign-off; no ids while something waits (video mode, a frame
   *  without a box) is nothing. */
  const verify = useCallback(
    (ids: string[]) => {
      const after = settleAfter(ids);
      if (ids.length === 0) {
        // `settleAfter` hands back `verifyAndAdvance` itself when nothing
        // waits; anything else would only refetch the file for nothing.
        if (after === verifyAndAdvance) verifyAndAdvance();
        return;
      }
      detectionsApi
        .bulkVerify(ids, true)
        .then(() => {
          // Same stack as a relabel: `bulkRevertToOriginal` undoes a
          // human verify as well as a human label.
          setUndoStack((s) => [...s, ids]);
          after();
        })
        .catch((err: Error) => toast.error(err.message));
    },
    [settleAfter, verifyAndAdvance],
  );
  const verifyOnScreen = useCallback(
    () => verify(onScreen.map((d) => d.id)),
    [verify, onScreen],
  );
  const markTargetsCorrect = useCallback(() => verify(targetIds), [verify, targetIds]);

  /** Every label action goes through here: the grid's `bulk-relabel`
   *  (which verifies), then the undo stack, then `settleAfter`'s step. */
  const applyLabel = useCallback(
    (ids: string[], label: string | null, category: string | undefined) => {
      if (ids.length === 0) return;
      const after = settleAfter(ids);
      detectionsApi
        .bulkRelabel(ids, label, category)
        .then(() => {
          setUndoStack((s) => [...s, ids]);
          after();
        })
        .catch((err: Error) => toast.error(err.message));
    },
    [settleAfter],
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
  /** M: everything on screen takes the most common label among them.
   *  Ties resolve as in the grid, to the label counted first. */
  const matchMajority = useCallback(() => {
    if (!majority || onScreen.length < 2) return;
    applyLabel(
      onScreen.map((d) => d.id),
      majority.label,
      majority.category,
    );
  }, [applyLabel, majority, onScreen]);
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
      // The confirm has the keyboard: its Enter must not also verify.
      if (signOffOpen) return;
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }
      // Keys inside a popover (the rail's brightness sliders) belong to
      // the popover: without this, arrow keys on the slider also paged
      // the viewer. Same guard as the grid's key handler.
      if (
        e.target instanceof HTMLElement &&
        e.target.closest("[data-radix-popper-content-wrapper]")
      ) {
        return;
      }
      const key = e.key.toLowerCase();
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        const dir = e.key === "ArrowRight" ? 1 : -1;
        if (e.shiftKey) stepFrame(dir);
        else go(dir);
      }
      else if (e.key === "Enter" && !drawMode) {
        e.preventDefault();
        verifyOnScreen();
      } else if (key === "z" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        handleUndo();
      } else if (key === "v") {
        e.preventDefault();
        markTargetsCorrect();
      } else if (key === "d" && canDraw) {
        e.preventDefault();
        if (drawMode) setDrawMode(false);
        else setDrawMode(true);
      } else if (key === "p") {
        // The Counts modal's key: a playable video toggles between the
        // still and the real clip, at the frame on screen.
        if (playable) {
          e.preventDefault();
          if (viewMode === "video") backToStill();
          else watch();
        }
      } else if (key === "b") {
        e.preventDefault();
        setBoxesHidden((v) => !v);
      } else if (key === "f") {
        // Flag for review, the rail's key everywhere.
        e.preventDefault();
        if (file) triage.toggleFlag(file);
      } else if (e.key === "Tab") {
        // Taken from the dialog's focus order on purpose: the buttons
        // all have keys of their own, and Tab is the one key a person
        // expects to step through things on screen.
        e.preventDefault();
        cycleSelection(e.shiftKey ? -1 : 1);
      } else if (key === "x" || e.key === "Backspace" || e.key === "Delete") {
        // Backspace/Delete alias X: "make this box go away" is what a
        // person means by delete, and mark false is how the app says it.
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
    stepFrame,
    verifyOnScreen,
    backToStill,
    signOffOpen,
    loadingMore,
    cycleSelection,
    markTargetsFalse,
    markTargetsUnknown,
    relabelTargetsNow,
    matchMajority,
    applyShortcut,
    handleUndo,
    viewMode,
    playable,
    file,
    triage,
    watch,
    canDraw,
    markTargetsCorrect,
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

  const noTargets = targetIds.length === 0;
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
      // The shared rail: how you look at the picture (brightness, the
      // box toggle, flag, More), identical to the Counts modal. The
      // verdict actions stay in the right column, in words.
      toolbar={
        <ViewerToolRail
          brightness={brightness}
          onBrightnessChange={setBrightness}
          contrast={contrast}
          onContrastChange={setContrast}
          boxesHidden={boxesHidden}
          onToggleBoxes={() => setBoxesHidden((v) => !v)}
          file={file}
          triage={triage}
          downloadLabel={playable ? "Download video" : "Download image"}
          onDownload={handleDownload}
        />
      }
      image={
        file ? (
          <div className="flex h-full w-full flex-col">
          <div className="relative min-h-0 w-full flex-1">
            {viewMode === "video" && playable ? (
              <VideoPlayer
                file={file}
                detectionThreshold={project?.counting_threshold ?? 0}
                exportFnRef={exportFnRef}
                autoExport={pendingVideoExport}
                onAutoExportConsumed={() => setPendingVideoExport(false)}
                boxesHidden={boxesHidden}
                seekRequest={seekRequest}
                pauseFnRef={pauseFnRef}
                selectedTrackId={selectedTrackId}
                zoomStep={timelineZoom}
                onZoomStep={setTimelineZoom}
                markers={markers}
                onSelectTrack={selectTrack}
              />
            ) : (
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
                onLoaded={warmNextFrame}
                onScrubFrame={stepFrame}
                boxesHidden={boxesHidden}
                defaultCategory={activeLabel?.category ?? "animal"}
                defaultLabel={activeLabel?.label ?? undefined}
                exportFnRef={exportFnRef}
                imageFilter={imageFilter}
                frameNumber={frameNumber}
              />
            )}

            {/* The Counts modal's play affordance, verbatim: a big
                center play button over a video's still, and the
                honest chip when the browser cannot play the format.
                Click-through except the button, so the canvas keeps
                its clicks. No drawing guard: a clip cannot be drawn
                on, so the crosshair is never over this. */}
            {file.file_type === "video" &&
              viewMode !== "video" &&
              (playable ? (
                <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
                  <button
                    type="button"
                    onClick={watch}
                    title="Watch this video (P)"
                    className="pointer-events-auto flex h-16 w-16 items-center justify-center rounded-full bg-black/55 ring-1 ring-white/40 transition hover:scale-105 hover:bg-black/75"
                  >
                    <Play className="h-7 w-7 translate-x-0.5 fill-white text-white" />
                  </button>
                </div>
              ) : (
                <div className="pointer-events-none absolute inset-x-0 bottom-4 z-10 flex justify-center">
                  <span className="rounded-md bg-black/60 px-2.5 py-1 text-xs text-white/80">
                    Video format not supported for playback
                  </span>
                </div>
              ))}
          </div>
          {/* The clip's tracks as bars under the still, the same
              timeline the player carries in video mode. A bar selects
              the track's card and shows its frame; the lane shows any
              frame. */}
          {viewMode !== "video" && (
            <ClipTimeline
              file={file}
              detectionThreshold={project?.counting_threshold ?? 0}
              currentFrame={shownFrame ?? file.best_frame_number ?? 0}
              selectedTrackId={selectedTrackId}
              zoomStep={timelineZoom}
              onZoomStep={setTimelineZoom}
              markers={markers}
              onSelectTrack={selectTrack}
              onSeek={showFrame}
            />
          )}
          </div>
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
              {/* Say what is on screen. The cover is the clip's picture
                  and nothing is judged on it: the detections are the
                  tracks, one bar each under the picture. It leads with
                  the detector on purpose, so nobody reads "one frame"
                  as "the AI looked at one frame". A box can only be
                  drawn on the cover (`AnnotationCanvas` stamps the frame
                  on screen on every box it creates, and D returns
                  there), so that clause stays. Worded to hold for a
                  clip where nothing was found at all. */}
              {item.file_type === "video" && (
                <div className="pt-1 text-muted-foreground/80">
                  The AI checked the whole clip. This is the clip's cover,
                  the picture that stands for it in the grid. A box you
                  draw is saved on it.
                  {(file?.tracks.length ?? 0) > 0 &&
                    " Each detection the AI followed is a bar below; click one to see it on its own frame."}
                  {playable && " Press P or the play button to watch the clip."}
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
              {file?.file_type !== "video"
                ? "These act on the box you select, with a click or Tab. None selected means all boxes."
                : viewMode === "video"
                  ? "These act on the detection you select by clicking a bar. Press P to come back to the still on the moment you paused; with nothing selected they act on the detections that frame shows."
                  : "These act on the detection you select: click a box, Tab through them, or click a bar. None selected means every detection on this frame, each through all of its frames."}
            </p>
            <div className="space-y-1.5">
              {/* First, because agreeing is the commonest verdict on a
                  clip the detector handled well. Same tick as the
                  Verify button on purpose: one verb, two scopes, told
                  apart by where they sit and by the primary styling. */}
              <Button
                variant="outline"
                size="sm"
                className="w-full justify-center"
                disabled={noTargets}
                onClick={markTargetsCorrect}
                title={
                  selectedDetectionId
                    ? `Verify ${scope} with the label it already has and go to the next one`
                    : `Verify ${scope} as labelled`
                }
              >
                <Check className="h-4 w-4 mr-1" />
                Mark correct
                <Kbd>V</Kbd>
              </Button>

              {canDraw && (
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full justify-center"
                  onClick={() => setDrawMode(!drawMode)}
                >
                  <SquareDashed className="h-4 w-4 mr-1" />
                  {drawMode ? "Stop drawing" : "Draw a box"}
                  <Kbd>D</Kbd>
                </Button>
              )}

              <Button
                variant="outline"
                size="sm"
                className="w-full justify-center"
                disabled={noTargets}
                onClick={markTargetsFalse}
                title={`Mark ${scope} as false and verify`}
              >
                <Ban className="h-4 w-4 mr-1" />
                Mark false
                <Kbd>X</Kbd>
              </Button>

              <Button
                variant="outline"
                size="sm"
                className="w-full justify-center"
                disabled={noTargets}
                onClick={markTargetsUnknown}
                title={`Mark ${scope} as unidentifiable animals and verify`}
              >
                <CircleHelp className="h-4 w-4 mr-1" />
                Mark unknown
                <Kbd>U</Kbd>
              </Button>

              {/* What is on screen, whatever is selected: a majority is
                  a statement about the set. Hidden when it cannot mean
                  anything (one box, or no labels). */}
              {majority && onScreen.length >= 2 && (
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full justify-center"
                  onClick={matchMajority}
                  title={`Relabel ${seen} to ${majority.common_name ?? majority.label} and verify`}
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
                disabled={noTargets}
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
                      disabled={noTargets}
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
              honest one. Photos only, with the drawing it serves. */}
          {canDraw && (
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
          )}
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
                : verifyOnScreen
            }
            disabled={verifying || (!isVerified && cards.length > 0 && onScreen.length === 0)}
            title={
              isVerified
                ? "Unverify: hand every detection in this file back to the AI"
                : file?.file_type === "video"
                  ? "Verify every detection on this frame, each through all of its frames, and go on to the next animal still unchecked. The clip is signed off once nothing in it is left unchecked."
                  : `Verify every ${unit} in this ${noun} as it is and go to the next file`
            }
          >
            <Check className="h-4 w-4 mr-1" />
            {isVerified ? "Unverify" : verifyLabel}
            {!isVerified && <Kbd onPrimary>⏎</Kbd>}
          </Button>

          {/* The whole clip, the one scope a person cannot see: a button
              without a key, behind a confirm that says how many would
              count as verified unseen. No number on the button: the
              dialog carries them, and "all 1 detections" reads wrong. */}
          {file?.file_type === "video" && !isVerified && cards.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              className="w-full justify-center"
              disabled={verifying}
              onClick={() => setSignOffOpen(true)}
              title="Verify every detection in this clip, the ones you have not looked at too, and go to the next file. Asks first."
            >
              <CheckCheck className="h-4 w-4 mr-1" />
              Verify the whole clip
            </Button>
          )}
          <SignOffDialog
            open={signOffOpen}
            onOpenChange={setSignOffOpen}
            clips={1}
            detections={cards.length}
            unchecked={unchecked}
            onConfirm={verifyAndAdvance}
          />

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
              // Same rule as the keys: a selected box (including a
              // freshly drawn one, which arrives selected) hands the
              // selection to the next waiting card; the last verdict on
              // a file signs it off.
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
