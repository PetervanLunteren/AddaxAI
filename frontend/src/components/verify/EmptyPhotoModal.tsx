/**
 * Full-size view of one empty file, for the Empties tab.
 *
 * Same frame as the Detections detail view (`VerifyDetailShell`): picture on
 * the left, what it is on the top right, what you can do on the bottom
 * right. A person is doing the same job in both, so they should not
 * have to learn two screens.
 *
 * A tile is not enough to tell an empty scene from a half-hidden
 * animal, so every check goes through here. The detector's
 * sub-threshold boxes are NOT drawn: the page promises "empty", and
 * machine boxes on a page called Empties read as a contradiction, not
 * as a hint. Only boxes the user draws show. (An earlier version drew
 * the weak boxes as "where the detector nearly fired" and it confused
 * more than it helped.)
 *
 * Two things can happen. **Verify** signs the file off, whether that
 * means "nothing here" or "what I drew is right"; it is one action
 * because it is one decision. Or draw a box on an animal the detector
 * missed: arm the crosshair with D or the button, drag, and the species
 * search opens on the new box by itself. Name a species in the Default
 * label card and new boxes take that instead, with nothing to answer,
 * which only pays off while drawing several of one thing.
 *
 * The file deliberately stays put after a box is drawn. An earlier
 * version refetched immediately, so the file stopped being empty, left
 * the list and took the modal with it, which meant a box could never be
 * moved, resized, relabelled or deleted. The list is refreshed when the
 * modal closes instead.
 *
 * Signing off the last file on a page does not close it either. The tab
 * fetches the next batch and the run carries on, so 500 empties is one
 * pass rather than ten passes of 48 with a reopen between each.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  ChevronDown,
  Loader2,
  SquareDashed,
  Tag,
  X,
} from "lucide-react";
import { toast } from "sonner";

import { filesApi } from "../../api/files";
import { projectsApi } from "../../api/projects";
import { formatCameraDate, formatCameraTime } from "../../lib/datetime";
import { isHumanDrawnBox, isNonLabel } from "../../lib/detection-utils";
import { basename } from "../../lib/path-utils";
import { useLabelOptions, type LabelOption } from "../../hooks/useLabelOptions";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
import { AnnotationCanvas } from "./AnnotationCanvas";
import { LabelPicker } from "./LabelPicker";
import { DetailCard, VerifyDetailShell } from "./VerifyDetailShell";
import type { EmptyFileItem } from "../../api/types";

interface EmptyPhotoModalProps {
  projectId: string;
  /** The page currently on screen; navigation stays inside it. */
  items: EmptyFileItem[];
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

export function EmptyPhotoModal({
  projectId,
  items,
  index,
  onIndexChange,
  onClose,
  onExhausted,
  loadingMore,
  onChanged,
}: EmptyPhotoModalProps) {
  const queryClient = useQueryClient();
  const [drawMode, setDrawMode] = useState(false);
  const [selectedDetectionId, setSelectedDetectionId] = useState<string | null>(
    null,
  );
  const [relabelDetectionId, setRelabelDetectionId] = useState<string | null>(
    null,
  );
  // The species a newly drawn box gets. Null means an unnamed animal,
  // which is the honest default here: the detector found nothing, so
  // there is no majority label to borrow from.
  const [activeLabel, setActiveLabel] = useState<LabelOption | null>(null);
  const [speciesPickerOpen, setSpeciesPickerOpen] = useState(false);

  // The files this viewer has signed off. The grid deliberately does
  // not refetch while the viewer is open (see `EmptiesTab`), so `items`
  // still calls them unverified, and "next unverified" would walk
  // straight back onto files that are already done. Cleared on close,
  // which is when the grid refetches anyway.
  const verifiedHere = useRef<Set<string>>(new Set());

  const item = index === null ? undefined : items[index];

  // Draw mode is a per-file decision, not a sticky tool: leaving it on
  // while paging would put the next Enter on a crosshair. Reset during
  // render rather than in an effect, so the canvas never paints one
  // frame in the wrong mode.
  const [stateFor, setStateFor] = useState(item?.id);
  if (item?.id !== stateFor) {
    setStateFor(item?.id);
    setDrawMode(false);
    setSelectedDetectionId(null);
    setRelabelDetectionId(null);
    if (!item) verifiedHere.current.clear();
  }

  const { data: project } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => projectsApi.get(projectId),
  });
  const { options: labelOptions, isLoading: labelOptionsLoading } =
    useLabelOptions(project?.classification_model_id ?? null, projectId);

  const { data: file } = useQuery({
    queryKey: ["file", item?.id],
    queryFn: () => filesApi.get(item!.id),
    enabled: !!item,
  });

  // The list row is what the grid last fetched; the file query is live.
  const isVerified = file?.verified ?? item?.verified ?? false;

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
   *  shutting, so a long run of empties is one uninterrupted pass. */
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

  /** Enter, and the primary button while the file is unverified: "I am
   *  done with this one". Drawing a box already verifies the file, so
   *  after that this is only the move on, which is the whole reason it
   *  is not a toggle. Unverifying stays on the button, where it cannot
   *  be hit by a person leaning on Enter. Same rule as the Detections
   *  viewer's `handleVerifyAndAdvance`. */
  const verifyAndAdvance = useCallback(() => {
    if (!item) return;
    if (!isVerified) setVerified({ id: item.id, verified: true });
    advance();
  }, [item, isVerified, setVerified, advance]);

  // Warm the next file while this one is being looked at. Measured on a
  // real empty: 1.47 MB for the picture and a 698-byte row, and the row
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

  /** A box was drawn, moved, resized, relabelled or deleted. Refresh
   *  this file and the counts, but leave the list alone so the user can
   *  keep working on the box they just made. */
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
      setRelabelDetectionId(detectionId);
    },
    [activeLabel],
  );

  const captured = item?.captured_at_local;
  // A box the user drew that still claims something is in the picture.
  //
  // Both halves matter. `isHumanDrawnBox` replaces
  // `classification_method === "human"`, which a *relabelled* machine
  // box also carries and which therefore fired on boxes the user never
  // drew. The `!isNonLabel` half excludes a drawn box the user has
  // since marked as "nothing here": that box still renders on the
  // canvas, deliberately, so it can be deleted, but it does not stop
  // the file being empty and must not claim otherwise.
  const hasHumanBox = (file?.detections ?? []).some(
    (d) => isHumanDrawnBox(d) && !isNonLabel(d.label),
  );

  // Same verbs as the Detections modal, so nothing new to learn.
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
      if (e.key === "ArrowRight") { e.preventDefault(); go(1); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); go(-1); }
      else if (e.key === "Enter" && !drawMode) {
        e.preventDefault();
        verifyAndAdvance();
      } else if (e.key === "d" || e.key === "D") {
        e.preventDefault();
        setDrawMode((v) => !v);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [item, go, drawMode, verifyAndAdvance, loadingMore]);

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
            // Show only the boxes the user drew. The detector's
            // sub-threshold boxes stay hidden: this page says "empty",
            // so drawing machine boxes here contradicts it. If there
            // is an animal, the user draws it.
            //
            // Said with a flag rather than by passing a threshold of 1
            // and leaning on human boxes carrying confidence 1.0. That
            // worked, but it stated none of the intent, and it stopped
            // being enough once `shouldDrawBbox` learned to let
            // verified boxes through at any confidence.
            detectionThreshold={project?.counting_threshold ?? 0}
            humanDrawnOnly
            selectedDetectionId={selectedDetectionId}
            onSelectDetection={setSelectedDetectionId}
            onRequestRelabel={setRelabelDetectionId}
            drawMode={drawMode}
            onDrawModeChange={setDrawMode}
            onMutated={handleCanvasChange}
            onCreated={handleCreated}
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
              <div className="truncate">{basename(item.file_path)}</div>
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

          {/* Sits with the other cards rather than among the buttons:
              it describes how drawing behaves, it is not a verdict on
              this file. "Ask me each time" is the default and the
              honest one, since nothing was found here and there is no
              majority label to borrow from. */}
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

          {/* Not a card, and last: the cards describe the file, this
              describes something you did to it. It answers the question
              that follows drawing a box, "so what happens now?", and it
              is not there until you have drawn one. */}
          {hasHumanBox && (
            <Callout variant="info" size="compact" className="mx-3 mt-3">
              Your box means this file is not empty any more. It has
              moved to Detections, where it counts like any other
              detection.
            </Callout>
          )}
        </>
      }
      actions={
        <>
          <Button
            variant="outline"
            size="sm"
            className="w-full justify-center"
            onClick={() => setDrawMode((v) => !v)}
          >
            <SquareDashed className="h-4 w-4 mr-1" />
            {drawMode ? "Stop drawing" : "Draw a box"}
            <kbd className="ml-1.5 text-[10px] font-sans text-muted-foreground/60 border border-border/60 rounded px-1 py-0.5 shadow-[0_1px_0_0_rgba(0,0,0,0.08)] leading-none">
              D
            </kbd>
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
            {!isVerified && (
              <kbd className="ml-1.5 text-[10px] font-sans text-primary-foreground/60 border border-primary-foreground/30 rounded px-1 py-0.5 shadow-[0_1px_0_0_rgba(255,255,255,0.1)] leading-none">
                ⏎
              </kbd>
            )}
          </Button>

          {/* Two searches, no triggers of their own. The first names a
              box: opened by drawing one, or by clicking a box's label
              on the canvas. The second sets the row above. Both use
              `headless`, which is the prop for exactly this. */}
          <LabelPicker
            headless
            value={null}
            onSelect={(option) => {
              const id = relabelDetectionId;
              setRelabelDetectionId(null);
              if (!id) return;
              import("../../api/detections").then(({ detectionsApi }) =>
                detectionsApi
                  .bulkRelabel([id], option.label, option.category)
                  .then(handleCanvasChange)
                  .catch((err: Error) => toast.error(err.message)),
              );
            }}
            options={labelOptions}
            isLoading={labelOptionsLoading}
            forceOpen={!!relabelDetectionId}
            onOpenChange={(open) => {
              if (!open) setRelabelDetectionId(null);
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
            forceOpen={speciesPickerOpen}
            onOpenChange={(open) => {
              if (!open) setSpeciesPickerOpen(false);
            }}
            projectId={projectId}
          />
        </>
      }
    />
  );
}
