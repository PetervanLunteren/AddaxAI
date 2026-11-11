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
 * The picture is a canvas, a photo and a clip's frame alike. Arm the
 * crosshair with D or the button, drag, and the species search opens
 * on the new box by itself; name a species in the Default label card
 * and new boxes take that instead; drag a box or its corners to fix
 * it. On a clip the box is saved on the frame on screen and becomes a
 * one-frame track there (a bar one step wide, a card, a count on that
 * frame), never while the clip plays. Drawing on clips was off from
 * 2026-09-09 to 2026-11-03 on the argument that one frame in ten
 * thousand is no place to record a missed animal; MaxN is a per-frame
 * count, so the frame you count on is exactly the place, which is what
 * BRUV analysts do on the MaxN frame.
 *
 * A verdict waits for the frame it is about. While a frame of a clip is
 * on its way the canvas dims the old picture and takes no clicks, and
 * `canJudge` holds Enter, V, X and the rest until the file and its frame
 * have landed (`verify` and `applyLabel`, the two doors every verdict
 * goes through), so nothing is judged or drawn on a picture that is not
 * the one asked for, and a clip still loading is never signed off as
 * empty. Navigation keys stay live.
 *
 * The label actions are the Detections grid's, with one scope rule:
 * actions act on what you see. V, R, X, U and the saved number-key labels
 * act on the selected detection, and with none selected on every
 * detection on screen (`onScreen`): a photo's boxes, or the animals on
 * the clip frame shown, each through all of its frames. A clip's other
 * two thousand tracks are never touched from a frame that shows 17 of
 * them (they were, until 2026-11-03: a V on the MaxN frame signed off
 * the whole drop). One rule decides what happens next (`settleAfter`):
 * the picture moves on only when agreeing (Enter, V) leaves every box
 * on it checked; a correction (X, U, R, M, a number key, naming a drawn box)
 * stays where the person is working. A photo is one frame, so the
 * rule is the same there. Click a box, Tab through the boxes on screen, or
 * click a bar on a clip's timeline to select one; the buttons say which
 * scope they are in. M relabels every detection on screen to the most
 * common label among them, whatever is selected. Cmd+Z is the tab's
 * undo, shared with the grid: it puts back exactly what the last action
 * changed, a label, a verify or an unverify, on whichever file that was,
 * and shows that file. Drawing or moving a box cannot be undone, so it
 * starts a fresh history.
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
  CircleMinus,
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
  playStartFrame,
  steppedFrame,
} from "../../lib/track-utils";
import { canvasImage, frameImageUrl, warmClipFrames } from "../../lib/clip-frames";
import { basename } from "../../lib/path-utils";
import { useLabelOptions, type LabelOption } from "../../hooks/useLabelOptions";
import {
  SHORTCUT_SLOTS,
  shortcutSlotFromKey,
  useShortcutLabels,
} from "../../hooks/useShortcutLabels";
import { Button } from "../ui/button";
import { Kbd } from "../ui/kbd";
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
import type { UndoStack } from "./useUndoStack";
import { mergeSnapshots } from "../../lib/undo-snapshot";
import type { LabelsFileItem, UndoSnapshot } from "../../api/types";

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
  /** The tab's undo history, shared so an action here and Cmd+Z in the
   *  grid are one sequence. The tab refetches the tiles on undo; the
   *  viewer adds what only it knows (`handleUndo`). */
  undo: UndoStack;
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
  undo,
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
  // without selecting anything. A box drawn on a clip lands on this
  // frame (the canvas stamps it; the cover when null).
  const [shownFrame, setShownFrame] = useState<number | null>(null);
  // False from the moment a picture is asked for until it is on screen
  // (`onLoadingChange`); the frame gate described in the header.
  const [frameReady, setFrameReady] = useState(false);
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
  const { imageFilter } = useImageAdjust();
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

  const item = index === null ? undefined : items[index];

  // Draw mode and the selection are per-file, not sticky: leaving draw
  // mode on while paging would put the next Enter on a crosshair. Reset
  // during render rather than in an effect, so the canvas never paints
  // one frame in the wrong mode. Hiding the boxes is a way of looking,
  // so it stays.
  const [stateFor, setStateFor] = useState(item?.id);
  if (item?.id !== stateFor) {
    setStateFor(item?.id);
    setDrawMode(false);
    setSelectedDetectionId(null);
    setRelabelTargets(null);
    setViewMode("frame");
    setShownFrame(null);
    setFrameReady(false);
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

  // The number-key slots, the same project-wide ones the Detections grid
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
  // The frame gate (header): a verdict needs the file, and on the still
  // the frame it is about. Playback has no frame to wait for, and the
  // canvas is unmounted there, so it never reports one; the still
  // reports again when it comes back.
  const canJudge = file != null && (viewMode === "video" || frameReady);

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
  // Watch the clip from the frame on screen, where the playhead is,
  // the cover included, and from the start once it is at the end
  // (`playStartFrame`, the Counts modal's rule too). Until 2026-11-05
  // the cover sent no seek, so P on a fresh open played from 0:00 while
  // P after a step played from the step. An armed crosshair is
  // dropped: nothing is drawn on a playing clip.
  const watch = useCallback(() => {
    const frame = file ? playStartFrame(file, shownFrame) : null;
    if (frame != null) {
      setSeekRequest({ frame, nonce: Date.now(), play: true });
    }
    setDrawMode(false);
    setViewMode("video");
  }, [file, shownFrame]);
  // And back: the still on the paused moment, snapped to the nearest
  // sampled frame (`nearestSampledFrame`), since the detector sampled a
  // few frames per second and a paused frame is rarely one of them.
  // The selection is kept: P is not a lane click.
  const backToStill = useCallback(() => {
    const frame = pauseFnRef.current?.();
    if (frame != null && file) setShownFrame(nearestSampledFrame(file, frame));
    setViewMode("frame");
  }, [file]);
  /** A still is a canvas, a photo's and a clip's frame alike; a playing
   *  clip is not. The box is saved on the frame on screen. */
  const canDraw = file != null && viewMode !== "video";

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
  // would read like a quick label while acting on more. One word for
  // photo and clip alike, the box: a clip's frame is a picture with
  // boxes, one per animal on it (until 2026-11-05 a clip said
  // "detection", and the two viewers read like two tools).
  const unit = "box";
  const units = "boxes";
  // The wording reads the grid's item, not the loaded file: the item
  // already knows it is a clip, so nothing says "photo" while it loads.
  const noun = item?.file_type === "video" ? "clip" : "photo";
  // What you see, in words: "all 12 boxes", "the 17 boxes on this frame".
  const isClip = item?.file_type === "video";
  const seen = isClip
    ? onScreen.length === 1
      ? `the 1 ${unit} on this frame`
      : `the ${onScreen.length} ${units} on this frame`
    : onScreen.length === 1
      ? `the 1 ${unit}`
      : `all ${onScreen.length} ${units}`;
  // What the action buttons will hit right now, and how to change it,
  // in one live line above them. The primary button cannot say it: it
  // counts what Enter verifies and ignores the selection, so with one
  // box picked it still says 33 while Mark false would hit 1. A live
  // line teaches the rule by changing with it; the fixed sentence it
  // replaced described the rule and was hard to follow (2026-11-05).
  // Playback first: a click on the playing clip pauses it rather than
  // clearing the selection, so the still's hint would be wrong there.
  // In playback it also says why buttons are grey: the verdicts wait
  // for a picked bar, drawing for a still.
  const scopeLine = viewMode === "video"
    ? selectedDetectionId
      ? "Acting on the box you picked. Pause (P) to draw."
      : "Pick a bar to act on its box. Pause (P) to draw."
    : selectedDetectionId
      ? "Acting on the selected box. Click the picture outside the boxes to go back to all."
      : onScreen.length === 0
        ? isClip && cards.length > 0
          ? "No boxes on this frame. Click a bar to pick one."
          : isClip
            ? "No boxes on this frame."
            : "No boxes to act on."
        : `Acting on ${seen}. Click a box${isClip ? " or a bar" : ""} to pick one.`;
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
      : item?.file_type !== "video"
        ? cards.length === 1
          ? `Verify 1 ${unit}`
          : `Verify all ${cards.length} ${units}`
        : onScreen.length === 0
          ? "Verify this frame"
          : onScreen.length === 1
            ? `Verify 1 ${unit} on this frame`
            : `Verify the ${onScreen.length} ${units} on this frame`;
  const majority = useMemo(() => labelMajority(onScreen), [onScreen]);
  // A majority is a statement about a set: two boxes or more, with labels.
  const canMatchMajority = majority != null && onScreen.length >= 2;
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

  // Through the bulk route, the one path every file verify in the app
  // takes, which answers `before` for the undo history. `alongside` is
  // the snapshot of a label change made by the same keypress (a
  // whole-picture X, U or number key relabels and then signs off): the
  // two merge into one entry, so one press is one undo.
  const { mutate: setVerified, isPending: verifying } = useMutation({
    mutationFn: ({
      id,
      verified,
    }: {
      id: string;
      verified: boolean;
      alongside?: UndoSnapshot;
    }) => filesApi.bulkVerify([id], verified),
    onSuccess: (r, { id, verified, alongside }) => {
      undo.push(alongside ? mergeSnapshots([alongside, r.before]) : r.before);
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
  const verifyAndAdvance = useCallback(
    (alongside?: UndoSnapshot) => {
      if (!item) return;
      setVerified({ id: item.id, verified: true, alongside });
      advance();
    },
    [item, setVerified, advance],
  );

  // Warm the next file while this one is being looked at. Measured on a
  // real file: 1.47 MB for the picture and a 698-byte row, and the row
  // is what the canvas waits on, so without this every step of a run
  // pays for both from cold. The image is not a query, but the endpoint
  // sends `max-age=86400, immutable`, so loading it the way the canvas
  // does (`canvasImage`, `frameImageUrl`) is all the warming it needs.
  // Same idea as the Counts viewer prefetching the next event and its
  // filmstrip.
  const nextItem = index === null ? undefined : items[index + 1];
  useEffect(() => {
    if (!nextItem) return;
    canvasImage().src = frameImageUrl(nextItem.id);
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

  /** The cards still unchecked once a verdict on `ids` lands, read
   *  before it does. By animal, not by box: on a clip the box picked on
   *  the picture is that frame's box while the list holds each animal's
   *  card, and a verdict on either reaches the whole track. */
  const waitingAfter = useCallback(
    (ids: string[]) => {
      if (!file) return [];
      const acted = new Set(ids);
      const tracks = new Set(
        file.detections.filter((d) => acted.has(d.id) && d.track_id).map((d) => d.track_id),
      );
      return uncheckedCards(file, project?.counting_threshold ?? 0).filter(
        (c) => !acted.has(c.id) && !(c.track_id && tracks.has(c.track_id)),
      );
    },
    [file, project?.counting_threshold],
  );

  /** The clip's next animal still waiting after a verdict on `ids`,
   *  undefined when none waits. Forward in time from the frame on
   *  screen, by the frame each animal is shown on (its representative
   *  frame): the first unchecked animal at or after the one on screen,
   *  whatever its species, and the earliest only when nothing lies
   *  ahead. Ordering by the bar's start instead (an hour on 2026-11-03)
   *  picked long-lived animals that had started earlier, and V V V on a
   *  shoal walked the picture backwards. A photo never walks: a photo
   *  finished is a file finished. */
  const nextWaitingAfter = useCallback(
    (ids: string[]) => {
      const pending = waitingAfter(ids);
      if (!file || file.file_type !== "video" || pending.length === 0) return undefined;
      const frameOf = new Map(file.tracks.map((t) => [t.id, t.representative_frame_number]));
      const shownOn = (d: { track_id: string | null }) =>
        d.track_id ? (frameOf.get(d.track_id) ?? 0) : 0;
      const anchor = shownFrame ?? file.best_frame_number ?? 0;
      const inTime = [...pending].sort(
        (a, b) => shownOn(a) - shownOn(b) || a.id.localeCompare(b.id),
      );
      return inTime.find((d) => shownOn(d) >= anchor) ?? inTime[0];
    },
    [file, waitingAfter, shownFrame],
  );

  /** The frames a person goes to next, asked for as soon as the current
   *  picture is on screen (`warmClipFrames`): the frame the next verdict
   *  lands on, so a run of V V V does not wait at each step, and the
   *  sampled frames either side, so Shift+Arrow steps around a moment
   *  (the MaxN frame, a fish half out of view) without a wait. Every
   *  frame is cut from the clip on request, which took 17 seconds a
   *  frame on an external drive (SLiSN_016, 2026-11-03). Only after the
   *  load, never alongside it: a decode next to the one on screen slows
   *  it down. */
  const warmNextFrames = useCallback(() => {
    if (!file || file.file_type !== "video") return;
    const next = nextWaitingAfter(targetIds);
    const track = next?.track_id ? file.tracks.find((t) => t.id === next.track_id) : undefined;
    const here = shownFrame ?? file.best_frame_number ?? 0;
    warmClipFrames(file, [
      track?.representative_frame_number,
      steppedFrame(file, here, 1),
      steppedFrame(file, here, -1),
    ]);
  }, [file, targetIds, nextWaitingAfter, shownFrame]);
  /** The canvas's word on whether the picture asked for is on screen:
   *  the frame gate (`frameReady`), and the moment to warm the next
   *  frame. */
  const handleLoading = useCallback(
    (loading: boolean) => {
      setFrameReady(!loading);
      if (!loading) warmNextFrames();
    },
    [warmNextFrames],
  );

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
   *  lists still hold the cards it touches (2026-11-06, with Peter):
   *
   *  - The picture moves on only when agreeing (`agreeing`: Enter, V)
   *    leaves every box on it checked. Then the clip shows the next
   *    animal still waiting (`nextWaitingAfter`), selected when the
   *    verdict was on a selected one, so V V V walks animal by animal
   *    and Enter Enter walks frame by frame.
   *  - Agreeing with boxes left on the picture selects the next of them,
   *    so V V V clears a frame and the last V moves on.
   *  - A correction (X, U, R, M, 1 to 5, naming a drawn box) stays on
   *    the picture with nothing selected: the person is working there,
   *    often on several boxes in a row. Until 2026-11-06 every verdict
   *    walked, and an X at the end of a drop wrapped back to its first
   *    unchecked fish.
   *  - Nothing left in the file: it is signed off, and agreeing also
   *    moves to the next file, as Enter on a photo always did. The
   *    sign-off is not a courtesy: the backend's rollup flips
   *    `File.verified` the moment the last card is verified, without
   *    rejecting the weak boxes underneath. */
  const settleAfter = useCallback(
    (ids: string[], agreeing: boolean) => {
      // Each returned step takes the verdict's undo snapshot: it goes on
      // the history as one entry, merged with the sign-off when the
      // verdict finishes the file, so one press is one undo.
      const stay = (before: UndoSnapshot) => {
        undo.push(before);
        setSelectedDetectionId(null);
        handleCanvasChange();
      };
      const waiting = waitingAfter(ids);
      if (waiting.length === 0) {
        if (agreeing) return verifyAndAdvance;
        return (before: UndoSnapshot) => {
          if (item) setVerified({ id: item.id, verified: true, alongside: before });
          else undo.push(before);
          setSelectedDetectionId(null);
          handleCanvasChange();
        };
      }
      if (!agreeing) return stay;
      // The next box still waiting on this picture, in the order Tab
      // walks (`visibleBoxes`, left to right), from the selected one on
      // and round: a photo and a clip frame alike. A clip's box waits
      // when its animal does. Storage order made the highlight hop
      // across a busy frame (bug hunt, 2026-11-06).
      const waitingTracks = new Set(waiting.map((c) => c.track_id));
      const waitingIds = new Set(waiting.map((c) => c.id));
      const at = visibleBoxes.findIndex((b) => b.id === selectedDetectionId);
      const nextBox = [...visibleBoxes.slice(at + 1), ...visibleBoxes.slice(0, at + 1)].find(
        (b) => (b.track_id ? waitingTracks.has(b.track_id) : waitingIds.has(b.id)),
      );
      if (nextBox) {
        return (before: UndoSnapshot) => {
          undo.push(before);
          setSelectedDetectionId(nextBox.id);
          handleCanvasChange();
        };
      }
      const next = nextWaitingAfter(ids);
      const fromSelection = selectedDetectionId != null;
      return (before: UndoSnapshot) => {
        undo.push(before);
        if (fromSelection) selectCard(next);
        else if (next?.track_id) {
          const track = file?.tracks.find((t) => t.id === next.track_id);
          if (track) showFrame(track.representative_frame_number);
        }
        handleCanvasChange();
      };
    },
    [
      waitingAfter,
      verifyAndAdvance,
      item,
      setVerified,
      visibleBoxes,
      nextWaitingAfter,
      selectedDetectionId,
      selectCard,
      file,
      showFrame,
      handleCanvasChange,
      undo,
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
      if (!canJudge) return;
      const after = settleAfter(ids, true);
      if (ids.length === 0) {
        // `settleAfter` hands back `verifyAndAdvance` itself when nothing
        // waits; anything else would only refetch the file for nothing.
        if (after === verifyAndAdvance) verifyAndAdvance();
        return;
      }
      detectionsApi
        .bulkVerify(ids, true)
        .then((r) => {
          after(r.before);
        })
        .catch((err: Error) => toast.error(err.message));
    },
    [canJudge, settleAfter, verifyAndAdvance],
  );
  const verifyOnScreen = useCallback(
    () => verify(onScreen.map((d) => d.id)),
    [verify, onScreen],
  );
  const markTargetsCorrect = useCallback(() => verify(targetIds), [verify, targetIds]);

  /** Every label action goes through here: the grid's `bulk-relabel`
   *  (which verifies), its `before` onto the undo history, then
   *  `settleAfter`'s rule
   *  for a correction: stay on the picture. Like `verify`, nothing while
   *  the frame is on its way.
   *
   *  On a clip every action here is about the animal, so each box goes
   *  with its track's card. `bulk-relabel` verifies only the boxes it is
   *  sent and labels the rest of the track, and the card is what says an
   *  animal is checked: a label on a frame box left the animal waiting,
   *  so the next V picked it again and the clip was never signed off
   *  (bug hunt, 2026-11-06). `bulk-verify` reaches the card by itself.
   *  Not in the backend: a card's Frames view relabels a few frames of a
   *  track without passing judgement on the animal. */
  const applyLabel = useCallback(
    (ids: string[], label: string | null, category: string | undefined) => {
      if (ids.length === 0 || !canJudge) return;
      const targets = new Set(ids);
      if (file?.file_type === "video") {
        for (const d of file.detections) {
          if (!targets.has(d.id) || !d.track_id) continue;
          const track = file.tracks.find((t) => t.id === d.track_id);
          const card = track && representativeBox(file.detections, track);
          if (card) targets.add(card.id);
        }
      }
      const sent = [...targets];
      const after = settleAfter(ids, false);
      detectionsApi
        .bulkRelabel(sent, label, category)
        .then((r) => after(r.before))
        .catch((err: Error) => toast.error(err.message));
    },
    [canJudge, file, settleAfter],
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
    if (!canMatchMajority) return;
    applyLabel(
      onScreen.map((d) => d.id),
      majority.label,
      majority.category,
    );
  }, [applyLabel, canMatchMajority, majority, onScreen]);
  const applyShortcut = useCallback(
    (slot: number) => {
      const option = shortcutLabels[slot];
      if (!option) return;
      applyLabel(targetIds, option.label, option.category);
    },
    [applyLabel, shortcutLabels, targetIds],
  );

  /** Cmd+Z and the Undo button. The tab's shared history does the
   *  restore and refetches the tiles; here is what only the viewer
   *  knows. `verifiedHere` follows the restored sign-offs, or "next
   *  unverified" would skip a file that undo just reopened. And the
   *  restored file is shown: after Enter advanced, the file undo put
   *  back is the previous one, and what undo did has to be visible. */
  const handleUndo = useCallback(async () => {
    const result = await undo.undo();
    if (!result) return;
    for (const f of result.snapshot.files) {
      if (f.verified) verifiedHere.current.add(f.file_id);
      else verifiedHere.current.delete(f.file_id);
    }
    const restored = new Set(result.snapshot.files.map((f) => f.file_id));
    for (const d of result.detections) restored.add(d.file_id);
    if (item && !restored.has(item.id)) {
      const idx = items.findIndex((i) => restored.has(i.id));
      if (idx !== -1) onIndexChange(idx);
    }
  }, [undo, item, items, onIndexChange]);

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
      } else if (shortcutSlotFromKey(e) !== null) {
        e.preventDefault();
        applyShortcut(shortcutSlotFromKey(e)!);
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
  // A clip with detections but none on the frame shown (video mode, a
  // frame between animals): Enter has nothing to do, so the button is
  // off and shows no key.
  const nothingOnScreen = cards.length > 0 && onScreen.length === 0;
  const freeSlot = SHORTCUT_SLOTS.find((n) => !shortcutLabels[n]);

  return (
    <VerifyDetailShell
      open
      onOpenChange={(open) => { if (!open) onClose(); }}
      // While drawing, Escape stops drawing and the viewer stays; the
      // next Escape closes it (the Draw button's tooltip says so).
      onEscapeKeyDown={(e) => {
        if (!drawMode) return;
        e.preventDefault();
        setDrawMode(false);
      }}
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
          // Flush with the panel, cancelling the shell's padding: the
          // picture sits in black either way, and the timeline then runs
          // edge to edge as it does in the Counts modal, not inside a
          // black frame. The card modal keeps the padding for its crop.
          <div className="-m-2 flex h-[calc(100%+1rem)] w-[calc(100%+1rem)] flex-col">
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
                onEnded={backToStill}
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
                // A drawn or moved box cannot be undone, so it starts a
                // fresh history: Cmd+Z must never reach past it to an
                // older action.
                onMutated={() => {
                  undo.clear();
                  handleCanvasChange();
                }}
                onCreated={handleCreated}
                onLoadingChange={handleLoading}
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
                its clicks; while drawing the button goes too, so a
                drag that starts in the middle draws a box instead of
                starting the clip. */}
            {file.file_type === "video" &&
              viewMode !== "video" &&
              !drawMode &&
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
            <p className="mb-2 text-xs text-muted-foreground" data-testid="action-scope">
              {scopeLine}
            </p>
            {/* A fieldset so one flag disables every verdict button
                while the frame is on its way (the header's frame gate);
                `min-w-0` undoes the element's own min-content width. */}
            <fieldset disabled={!canJudge} className="min-w-0 space-y-1.5">
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

              {/* Greyed out while the clip plays, not hidden: the action
                  stays where it is, and the scope line says why. */}
              <Button
                variant="outline"
                size="sm"
                className="w-full justify-center"
                disabled={!canDraw}
                onClick={() => setDrawMode(!drawMode)}
                // The mode's one hint, here rather than over the picture,
                // where a banner covered a corner someone may draw in.
                title={drawMode ? "Stop drawing (D or Esc)" : "Drag over the picture to draw a box. Esc stops drawing."}
              >
                <SquareDashed className="h-4 w-4 mr-1" />
                {drawMode ? "Stop drawing" : "Draw a box"}
                <Kbd>D</Kbd>
              </Button>

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
                  a statement about the set. Greyed out, not hidden, when
                  it cannot mean anything (one box, no labels, playback),
                  so the column does not jump between frames. */}
              <Button
                variant="outline"
                size="sm"
                className="w-full justify-center"
                disabled={!canMatchMajority}
                onClick={matchMajority}
                title={
                  majority
                    ? `Relabel ${seen} to ${majority.common_name ?? majority.label} and verify`
                    : undefined
                }
              >
                <CheckCheck className="h-4 w-4 mr-1 shrink-0" />
                <span className="truncate">
                  Set all to {canMatchMajority
                    ? (majority.common_name ?? majority.label)
                    : "the majority"}
                </span>
                <Kbd>M</Kbd>
              </Button>

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

              {/* The saved labels, the same number-key slots as the
                  Detections grid, as split buttons: the body applies the
                  slot's label (the key does the same), the chevron
                  segment opens the label search that changes it. Only
                  the set slots show, plus one row to fill the next free
                  slot; changes are saved on the project, so both tabs
                  see them. */}
              {SHORTCUT_SLOTS
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
            </fieldset>
          </DetailCard>

          {/* Sits with the other cards rather than among the buttons:
              it describes how drawing behaves, it is not a verdict on
              this file. "Ask me each time" is the default and the
              honest one. Stays through playback, unlike the Draw
              button: it is a setting, not an action. */}
          {file != null && (
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
            disabled={!undo.canUndo}
            onClick={handleUndo}
            title="Undo the last action"
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
            disabled={verifying || !canJudge || (!isVerified && nothingOnScreen)}
            title={
              isVerified
                ? "Unverify: hand every detection in this file back to the AI"
                : item?.file_type === "video"
                  ? "Verify the boxes on this frame and go to the next animal still unchecked"
                  : `Verify every ${unit} in this ${noun} as it is and go to the next file`
            }
          >
            {isVerified ? (
              <CircleMinus className="h-4 w-4 mr-1" />
            ) : (
              <Check className="h-4 w-4 mr-1" />
            )}
            {isVerified ? "Unverify" : verifyLabel}
            {!isVerified && !nothingOnScreen && canJudge && <Kbd onPrimary>⏎</Kbd>}
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
            // Called bare: the dialog's button would hand its click event
            // to `verifyAndAdvance` as the undo snapshot to merge with.
            onConfirm={() => verifyAndAdvance()}
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
              // A label is a correction: it stays on the picture
              // (`settleAfter`), a box just drawn included.
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
