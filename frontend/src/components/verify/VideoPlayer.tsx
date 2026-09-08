/**
 * Video player with SVG bounding box overlays synced to the current frame.
 *
 * Renders an HTML5 <video> element with an SVG overlay that shows
 * detection bounding boxes for the current frame. Used in the
 * verification modal as an alternative to the best-frame AnnotationCanvas.
 *
 * Bbox / label styling is driven by the shared detection-overlay constants
 * so changes in AnnotationCanvas are automatically reflected here.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { API_BASE_URL } from "../../lib/api-client";
import { splitPath } from "../../lib/path-utils";
import { passesDrawFilter } from "../../lib/detection-utils";
import {
  computePillLayout,
  placePill,
  roundedRectPath,
  PILL_PAD_Y,
  LINE_GAP,
  FONT,
  TEXT_START_X,
  BBOX_STROKE_WIDTH,
  BBOX_OPACITY,
  BBOX_CORNER_RADIUS,
  DIM_FILL,
  PILL_BG,
} from "../../lib/detection-overlay";
import { SpotlightDim } from "./SpotlightDim";
import type { FileWithDetections, DetectionResponse } from "../../api/types";
import { useSpeciesColorsVersion } from "../../utils/species-colors";
import { clipDurationFrames } from "../../lib/track-utils";
import { ClipTimeline, type TimelineMarker } from "./TrackTimeline";
interface VideoPlayerProps {
  file: FileWithDetections;
  detectionThreshold: number;
  /** For frame files: the source video's file ID (used for the video URL). */
  sourceVideoId?: string;
  /** For frame files: aggregated detections from all sibling frames. */
  allDetections?: DetectionResponse[];
  exportFnRef?: React.MutableRefObject<(() => void) | null>;
  /** When set true, run the annotated-video export once the video is
   *  playable. Lets the modal's Download button produce the boxed video
   *  even when it was clicked from frame view (it switches here first). */
  autoExport?: boolean;
  /** Called as soon as an autoExport request has been picked up, so the
   *  parent can clear the one-shot flag. */
  onAutoExportConsumed?: () => void;
  /** The modal's B toggle. Hides the live overlay only; the export
   *  always records the boxes, because an annotated video is the one
   *  thing it produces that the file on disk is not. */
  boxesHidden?: boolean;
  /** Jump to a frame and pause there (the Counts page's "Show" for the
   *  frame a MaxN was counted on). The nonce lets the same frame be
   *  asked for twice in a row. */
  seekRequest?: { frame: number; nonce: number } | null;
  /** The animal the person is looking at: its boxes keep the normal
   *  style, every other box dims. Null dims nothing. */
  selectedTrackId?: string | null;
  /** A bar on the timeline was clicked. The player has already jumped to
   *  the track's representative frame; the owner records the choice. */
  onSelectTrack?: (trackId: string) => void;
  /** Per species, the frame its MaxN was counted on (the Counts modal). */
  markers?: TimelineMarker[];
}

/** Browser-playable video formats. */
const PLAYABLE_FORMATS = new Set(["mp4", "m4v", "mov", "webm"]);

/** Frames at full opacity before fading starts. */
const HOLD_FRAMES = 5;
/** Frames over which the overlay fades from full to zero (after the hold). */
const FADE_FRAMES = 25;
/** Opacity factor for the boxes of every other track while one is selected. */
const OTHER_TRACK_DIM = 0.25;

/** Check whether a file's video format is browser-playable. */
export function isPlayableVideo(file: FileWithDetections): boolean {
  if (file.file_type !== "video") return false;
  return (
    file.frame_rate != null &&
    PLAYABLE_FORMATS.has((file.file_format || "").toLowerCase())
  );
}

// Detections that paint onto the video canvas always carry a bbox;
// event-level observations (null bbox) are filtered out upstream and
// never reach the renderer. This alias narrows the bbox fields so the
// canvas math doesn't need null guards at every coordinate.
type BboxedDetection = DetectionResponse & {
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
};

// ── Canvas overlay drawing (for video export) ─────────────────────
// Mirrors the SVG overlay rendering using the shared detection-overlay
// constants, so exported videos match the on-screen appearance.

function drawOverlaysOnCanvas(
  ctx: CanvasRenderingContext2D,
  dets: BboxedDetection[],
  w: number,
  h: number,
  opacity: number,
  scale: number,
) {
  if (dets.length === 0 || opacity <= 0) return;

  ctx.save();
  ctx.globalAlpha = opacity;

  // Spotlight dim: dim everything outside the UNION of the boxes. Built on an
  // offscreen canvas so punching the box holes (destination-out) clears only
  // the dim, never the underlying video frame. Overlapping boxes union, so the
  // overlap stays bright (an evenodd outer-rect-minus-holes path re-dims it).
  const dimLayer = document.createElement("canvas");
  dimLayer.width = w;
  dimLayer.height = h;
  const dctx = dimLayer.getContext("2d");
  if (dctx) {
    dctx.fillStyle = DIM_FILL;
    dctx.fillRect(0, 0, w, h);
    dctx.globalCompositeOperation = "destination-out";
    for (const det of dets) {
      dctx.beginPath();
      roundedRectPath(
        dctx,
        det.bbox_x * w, det.bbox_y * h,
        det.bbox_width * w, det.bbox_height * h,
        BBOX_CORNER_RADIUS * scale,
      );
      dctx.fill();
    }
    ctx.drawImage(dimLayer, 0, 0);
  }

  // Bounding boxes
  for (const det of dets) {
    const pill = computePillLayout(det);
    ctx.beginPath();
    roundedRectPath(
      ctx,
      det.bbox_x * w, det.bbox_y * h,
      det.bbox_width * w, det.bbox_height * h,
      BBOX_CORNER_RADIUS * scale,
    );
    ctx.strokeStyle = pill.color;
    ctx.lineWidth = BBOX_STROKE_WIDTH * scale;
    ctx.globalAlpha = opacity * BBOX_OPACITY;
    ctx.stroke();
    ctx.globalAlpha = opacity;
  }

  // Label pills
  ctx.textBaseline = "top";
  for (const det of dets) {
    const pill = computePillLayout(det);
    const pw = pill.pillWidth * scale;
    const ph = pill.pillHeight * scale;
    const { x, y: pillY } = placePill(
      {
        x: det.bbox_x * w,
        y: det.bbox_y * h,
        width: det.bbox_width * w,
        height: det.bbox_height * h,
      },
      { width: pw, height: ph },
      { width: w, height: h },
    );

    // Pill background
    ctx.beginPath();
    roundedRectPath(ctx, x, pillY, pw, ph, BBOX_CORNER_RADIUS * scale);
    ctx.fillStyle = PILL_BG;
    ctx.fill();

    // Text — both lines share one font, regular white.
    ctx.font = `${FONT * scale}px Arial, sans-serif`;
    ctx.fillStyle = "white";
    ctx.fillText(pill.categoryText, x + TEXT_START_X * scale, pillY + PILL_PAD_Y * scale);
    if (pill.hasLabel) {
      ctx.fillText(pill.labelText, x + TEXT_START_X * scale, pillY + (PILL_PAD_Y + FONT + LINE_GAP) * scale);
    }
  }

  ctx.restore();
}

// ── Component ─────────────────────────────────────────────────────

export function VideoPlayer({
  file,
  detectionThreshold,
  sourceVideoId,
  allDetections,
  exportFnRef,
  autoExport,
  onAutoExportConsumed,
  boxesHidden,
  seekRequest,
  selectedTrackId = null,
  onSelectTrack,
  markers,
}: VideoPlayerProps) {
  // Repaint when the project's colour map lands or changes.
  useSpeciesColorsVersion();
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [currentFrame, setCurrentFrame] = useState<number>(0);
  // The clip's length in frames: what the timeline draws its bars over.
  // The stored length (ingest) is there at once; the element's own
  // duration replaces it once the metadata is in.
  const [metadataFrames, setMetadataFrames] = useState(0);
  const [displayWidth, setDisplayWidth] = useState(0);
  const [isExporting, setIsExporting] = useState(false);
  // Export progress, driven by playback position (recording runs at 1x, so
  // these are exact, not estimates): fraction done and wall-clock seconds left.
  const [exportProgress, setExportProgress] = useState(0);
  const [exportRemaining, setExportRemaining] = useState(0);
  const animFrameRef = useRef<number>(0);
  const exportAbortRef = useRef(false);

  const videoFileId = sourceVideoId ?? file.id;
  const videoUrl = `${API_BASE_URL}/api/files/${videoFileId}/video`;
  const frameRate = file.frame_rate || 30;
  const durationFrames = metadataFrames || clipDurationFrames(file);
  const imgW = file.width_px || 1;
  const imgH = file.height_px || 1;

  // Track container display size so we can scale labels to screen pixels.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      setDisplayWidth(entries[0].contentRect.width);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Scale factor: image pixels per display pixel
  const s = displayWidth > 0 ? imgW / displayWidth : 1;

  const detections = allDetections ?? file.detections;

  // Group detections by frame_number, through the same filter every
  // other drawing surface uses. `passesDrawFilter` rather than
  // `shouldDrawBbox` because this player draws EVERY frame's boxes over
  // the real video on purpose, so the best-frame gate must not apply
  // here — that is the one rule this surface legitimately differs on.
  //
  // It used to inline `d.confidence < detectionThreshold` instead, and
  // so missed both the verified override and the rejected-box rule.
  // The result was one event modal disagreeing with itself: a box a
  // human confirmed below the threshold drew on the still and vanished
  // on play, and a box they rejected did the opposite.
  const detectionsByFrame = useMemo(() => {
    const map = new Map<number, BboxedDetection[]>();
    for (const d of detections) {
      if (!passesDrawFilter(d, detectionThreshold)) continue;
      if (d.frame_number == null) continue;
      const bb = d as BboxedDetection;
      const existing = map.get(d.frame_number);
      if (existing) {
        existing.push(bb);
      } else {
        map.set(d.frame_number, [bb]);
      }
    }
    return map;
  }, [detections, detectionThreshold]);

  // Find detections for the current video frame, persisting the last-seen
  // detections through frames that have none so boxes don't disappear between
  // analyzed frames.
  const lastDetectionsRef = useRef<BboxedDetection[]>([]);
  const lastMatchFrameRef = useRef<number>(0);

  const currentDetections = useMemo(() => {
    if (detectionsByFrame.size === 0) return [];
    let found = detectionsByFrame.get(currentFrame);
    if (!found) {
      for (const offset of [1, -1]) {
        found = detectionsByFrame.get(currentFrame + offset);
        if (found) break;
      }
    }
    if (found) {
      lastDetectionsRef.current = found;
      lastMatchFrameRef.current = currentFrame;
      return found;
    }
    return lastDetectionsRef.current;
  }, [currentFrame, detectionsByFrame]);

  const dimFor = (det: DetectionResponse) =>
    selectedTrackId != null && det.track_id !== selectedTrackId ? OTHER_TRACK_DIM : 1;

  // Full opacity for HOLD_FRAMES, then linearly fade to 0 over FADE_FRAMES
  const framesSinceMatch = Math.max(0, currentFrame - lastMatchFrameRef.current);
  const overlayOpacity =
    framesSinceMatch <= HOLD_FRAMES
      ? 1
      : Math.max(0, 1 - (framesSinceMatch - HOLD_FRAMES) / FADE_FRAMES);

  // Sync frame number from video time using requestAnimationFrame for smooth updates
  const syncFrame = useCallback(() => {
    const video = videoRef.current;
    if (video && !video.paused) {
      const frame = Math.round(video.currentTime * frameRate);
      setCurrentFrame(frame);
      animFrameRef.current = requestAnimationFrame(syncFrame);
    }
  }, [frameRate]);

  const handlePlay = useCallback(() => {
    animFrameRef.current = requestAnimationFrame(syncFrame);
  }, [syncFrame]);

  const handlePause = useCallback(() => {
    cancelAnimationFrame(animFrameRef.current);
    const video = videoRef.current;
    if (video) {
      setCurrentFrame(Math.round(video.currentTime * frameRate));
    }
  }, [frameRate]);

  const handleSeeked = useCallback(() => {
    const video = videoRef.current;
    if (video) {
      setCurrentFrame(Math.round(video.currentTime * frameRate));
    }
  }, [frameRate]);

  /** Jump to a frame and pause there, so that frame's boxes stay on
   *  screen. The `seekRequest` prop and the timeline both come here. */
  const seekTo = useCallback(
    (frame: number) => {
      const video = videoRef.current;
      if (!video) return;
      video.pause();
      video.currentTime = frame / frameRate;
      setCurrentFrame(frame);
    },
    [frameRate],
  );

  const handleSelectTrack = useCallback(
    (trackId: string) => {
      const track = file.tracks?.find((t) => t.id === trackId);
      if (track) seekTo(track.representative_frame_number);
      onSelectTrack?.(trackId);
    },
    [file.tracks, seekTo, onSelectTrack],
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      exportAbortRef.current = true;
      cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  // Auto-play when the component mounts or the video source changes
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.play().catch(() => {});
  }, [videoUrl]);

  // Jump to the requested frame and pause there, so the boxes of that
  // frame stay on screen. Waits for the metadata when the video is not
  // ready yet (a fresh mount, which is the usual case: the modal switches
  // to the player and asks for the frame in the same render).
  useEffect(() => {
    if (!seekRequest) return;
    const video = videoRef.current;
    if (!video) return;
    const seek = () => seekTo(seekRequest.frame);
    if (video.readyState >= 1) {
      seek();
      return;
    }
    video.addEventListener("loadedmetadata", seek, { once: true });
    return () => video.removeEventListener("loadedmetadata", seek);
  }, [seekRequest, seekTo, videoUrl]);

  // ── Video export ────────────────────────────────────────────────
  // Records the video with canvas-rendered overlays to an MP4 (or WebM
  // fallback). Plays the video from the start at normal speed and
  // captures each frame with overlays via the shared constants.

  const startExport = useCallback(() => {
    const video = videoRef.current;
    if (!video || isExporting) return;

    const canvas = document.createElement("canvas");
    canvas.width = imgW;
    canvas.height = imgH;
    const ctx = canvas.getContext("2d")!;

    // Label scale for native-resolution canvas (same ratio as live overlay)
    const exportScale = s;

    // Independent detection lookup state for the export
    let lastDets: BboxedDetection[] = [];
    let lastMatch = 0;

    const findDets = (frame: number) => {
      let found = detectionsByFrame.get(frame);
      if (!found) {
        for (const offset of [1, -1]) {
          found = detectionsByFrame.get(frame + offset);
          if (found) break;
        }
      }
      if (found) {
        lastDets = found;
        lastMatch = frame;
      }
      const elapsed = frame - lastMatch;
      const opacity =
        elapsed <= HOLD_FRAMES
          ? 1
          : Math.max(0, 1 - (elapsed - HOLD_FRAMES) / FADE_FRAMES);
      return { dets: lastDets, opacity };
    };

    const stream = canvas.captureStream(frameRate);
    const mimeType = "video/mp4;codecs=avc1";
    const recorder = new MediaRecorder(stream, {
      mimeType,
      videoBitsPerSecond: 8_000_000,
    });
    const chunks: Blob[] = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: mimeType });
      const url = URL.createObjectURL(blob);
      // For frame files, derive the video name from the parent directory
      // (frames are stored as .addaxai/video_frames/{video_name}/frame000000.jpg)
      const parts = splitPath(file.file_path);
      const fileName =
        sourceVideoId
          ? (parts[parts.length - 2]?.replace(/\.[^.]+$/, "") || "video")
          : (parts[parts.length - 1]?.replace(/\.[^.]+$/, "") || "video");
      const a = document.createElement("a");
      a.href = url;
      a.download = `${fileName}_annotated.mp4`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setIsExporting(false);
      setExportProgress(0);
      setExportRemaining(0);
    };

    exportAbortRef.current = false;
    setIsExporting(true);
    setExportProgress(0);
    setExportRemaining(0);

    const beginRecording = () => {
      recorder.start(1000);
      video.play().catch(() => {});

      const drawFrame = () => {
        if (exportAbortRef.current || video.ended) {
          video.pause();
          recorder.stop();
          return;
        }

        const frame = Math.round(video.currentTime * frameRate);

        // Progress from playback position. Guard NaN/Infinity duration.
        const dur = video.duration;
        if (dur && Number.isFinite(dur)) {
          setExportProgress(Math.min(1, video.currentTime / dur));
          setExportRemaining(Math.max(0, dur - video.currentTime));
        }

        // Draw video frame
        ctx.drawImage(video, 0, 0, imgW, imgH);

        // Draw overlays
        const { dets, opacity } = findDets(frame);
        drawOverlaysOnCanvas(ctx, dets, imgW, imgH, opacity, exportScale);

        requestAnimationFrame(drawFrame);
      };

      requestAnimationFrame(drawFrame);
    };

    // Seek to start before recording. If already at 0, start directly
    // since the seeked event won't fire.
    if (video.currentTime === 0) {
      beginRecording();
    } else {
      const onSeeked = () => {
        video.removeEventListener("seeked", onSeeked);
        beginRecording();
      };
      video.addEventListener("seeked", onSeeked);
      video.currentTime = 0;
    }
  }, [imgW, imgH, s, frameRate, detectionsByFrame, file.file_path, isExporting]);

  // Register export function for the download button
  useEffect(() => {
    if (!exportFnRef) return;
    exportFnRef.current = startExport;
  }, [exportFnRef, startExport]);

  // Auto-run the export when the parent requests it (Download clicked from
  // frame view, which switches here just to record). Wait for the video to
  // be playable so the first recorded frames aren't blank. Consume the
  // one-shot request immediately so it can't fire twice.
  useEffect(() => {
    if (!autoExport) return;
    const video = videoRef.current;
    if (!video) return;
    const run = () => {
      onAutoExportConsumed?.();
      startExport();
    };
    if (video.readyState >= 2) {
      run();
      return;
    }
    video.addEventListener("canplay", run, { once: true });
    return () => video.removeEventListener("canplay", run);
  }, [autoExport, startExport, onAutoExportConsumed]);

  return (
    <div className="relative w-full h-full flex flex-col items-center justify-center">
      {/* Video + SVG overlay container */}
      <div
        ref={containerRef}
        className="relative max-w-full min-h-0 flex-1"
        style={{ aspectRatio: imgW / imgH }}
      >
        <video
          ref={videoRef}
          src={videoUrl}
          crossOrigin="anonymous"
          controls
          controlsList="nodownload"
          className="w-full h-full object-contain"
          onLoadedMetadata={(e) =>
            setMetadataFrames(Math.round(e.currentTarget.duration * frameRate))
          }
          onPlay={handlePlay}
          onPause={handlePause}
          onSeeked={handleSeeked}
          onEnded={handlePause}
        />

        {/* Recording indicator — centered over the focus while the
            annotated video records. Progress and time-left are exact since
            recording is locked to 1x playback. Click-through so it never
            blocks the video underneath. */}
        {isExporting && (
          <div className="absolute inset-0 z-10 flex items-center justify-center pointer-events-none">
            <div className="flex flex-col items-center gap-3 rounded-lg bg-black/70 px-6 py-5 text-white">
              <div className="flex items-center gap-2 text-base font-medium">
                <span className="h-3 w-3 rounded-full bg-red-500 animate-pulse" />
                Recording…
              </div>
              <div className="h-2 w-56 overflow-hidden rounded-full bg-white/20">
                <div
                  className="h-full rounded-full bg-primary transition-[width] duration-150"
                  style={{ width: `${Math.round(exportProgress * 100)}%` }}
                />
              </div>
              <div className="text-xs text-white/80 tabular-nums">
                {Math.round(exportProgress * 100)}% ·{" "}
                {Math.ceil(exportRemaining)}s left
              </div>
            </div>
          </div>
        )}

        {/* SVG overlay: spotlight + bboxes + labels */}
        {!boxesHidden && currentDetections.length > 0 && overlayOpacity > 0 && (
          <svg
            style={{ opacity: overlayOpacity, transition: "opacity 0.05s linear" }}
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox={`0 0 ${imgW} ${imgH}`}
            preserveAspectRatio="xMidYMid meet"
          >
            {/* Spotlight dim overlay (union of boxes stays bright). */}
            <SpotlightDim
              width={imgW}
              height={imgH}
              rx={BBOX_CORNER_RADIUS * s}
              fill={DIM_FILL}
              boxes={currentDetections.map((det) => ({
                x: det.bbox_x * imgW,
                y: det.bbox_y * imgH,
                width: det.bbox_width * imgW,
                height: det.bbox_height * imgH,
              }))}
            />

            {/* Bounding boxes. With a track selected, the others dim. */}
            {currentDetections.map((det) => {
              const pill = computePillLayout(det);
              return (
                <rect
                  key={det.id}
                  x={det.bbox_x * imgW}
                  y={det.bbox_y * imgH}
                  width={det.bbox_width * imgW}
                  height={det.bbox_height * imgH}
                  rx={BBOX_CORNER_RADIUS * s}
                  fill="none"
                  stroke={pill.color}
                  strokeWidth={BBOX_STROKE_WIDTH * s}
                  opacity={BBOX_OPACITY * dimFor(det)}
                />
              );
            })}

            {/* Label pills — rendered at screen-pixel sizes via scale(s) */}
            {currentDetections.map((det) => {
              const pill = computePillLayout(det);
              const { x, y: pillY } = placePill(
                {
                  x: det.bbox_x * imgW,
                  y: det.bbox_y * imgH,
                  width: det.bbox_width * imgW,
                  height: det.bbox_height * imgH,
                },
                { width: pill.pillWidth * s, height: pill.pillHeight * s },
                { width: imgW, height: imgH },
              );

              return (
                <g
                  key={`label-${det.id}`}
                  transform={`translate(${x}, ${pillY}) scale(${s})`}
                  opacity={dimFor(det)}
                >
                  <rect
                    x={0}
                    y={0}
                    width={pill.pillWidth}
                    height={pill.pillHeight}
                    rx={BBOX_CORNER_RADIUS}
                    fill={PILL_BG}
                  />
                  <text
                    x={TEXT_START_X}
                    y={PILL_PAD_Y}
                    fill="white"
                    fontSize={FONT}
                    fontFamily="Arial, sans-serif"
                    dominantBaseline="hanging"
                  >
                    {pill.categoryText}
                  </text>
                  {pill.hasLabel && (
                    <text
                      x={TEXT_START_X}
                      y={PILL_PAD_Y + FONT + LINE_GAP}
                      fill="white"
                      fontSize={FONT}
                      fontFamily="Arial, sans-serif"
                      dominantBaseline="hanging"
                    >
                      {pill.labelText}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        )}
      </div>
      <ClipTimeline
        file={file}
        durationFrames={durationFrames}
        currentFrame={currentFrame}
        selectedTrackId={selectedTrackId}
        markers={markers}
        onSelectTrack={handleSelectTrack}
        onSeek={seekTo}
      />
    </div>
  );
}
