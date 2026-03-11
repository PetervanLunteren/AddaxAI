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
import {
  computePillLayout,
  roundedRectPath,
  svgRoundedRectPath,
  PILL_PAD_X,
  PILL_PAD_Y,
  DOT_R,
  LINE_GAP,
  FONT_SM,
  FONT_LG,
  TEXT_START_X,
  BBOX_STROKE_WIDTH,
  BBOX_OPACITY,
  BBOX_CORNER_RADIUS,
  DIM_FILL,
  PILL_BG,
} from "../../lib/detection-overlay";
import type { FileWithDetections, DetectionResponse } from "../../api/types";

interface VideoPlayerProps {
  file: FileWithDetections;
  detectionThreshold: number;
  /** For frame files: the source video's file ID (used for the video URL). */
  sourceVideoId?: string;
  /** For frame files: aggregated detections from all sibling frames. */
  allDetections?: DetectionResponse[];
  exportFnRef?: React.MutableRefObject<(() => void) | null>;
}

/** Browser-playable video formats. */
const PLAYABLE_FORMATS = new Set(["mp4", "m4v", "mov", "webm"]);

/** Frames at full opacity before fading starts. */
const HOLD_FRAMES = 5;
/** Frames over which the overlay fades from full to zero (after the hold). */
const FADE_FRAMES = 25;

/** Check whether a file's video format is browser-playable. */
export function isPlayableVideo(file: FileWithDetections): boolean {
  if (file.file_type === "video") {
    return (
      file.frame_rate != null &&
      PLAYABLE_FORMATS.has((file.file_format || "").toLowerCase())
    );
  }
  if (file.file_type === "frame" && file.source_video_id != null && file.frame_rate != null) {
    return true;
  }
  return false;
}

// ── Canvas overlay drawing (for video export) ─────────────────────
// Mirrors the SVG overlay rendering using the shared detection-overlay
// constants, so exported videos match the on-screen appearance.

function drawOverlaysOnCanvas(
  ctx: CanvasRenderingContext2D,
  dets: DetectionResponse[],
  w: number,
  h: number,
  opacity: number,
  scale: number,
) {
  if (dets.length === 0 || opacity <= 0) return;

  ctx.save();
  ctx.globalAlpha = opacity;

  // Spotlight dim overlay (evenodd: outer rect minus detection holes)
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(w, 0);
  ctx.lineTo(w, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  for (const det of dets) {
    roundedRectPath(
      ctx,
      det.bbox_x * w, det.bbox_y * h,
      det.bbox_width * w, det.bbox_height * h,
      BBOX_CORNER_RADIUS * scale,
    );
  }
  ctx.fillStyle = DIM_FILL;
  ctx.fill("evenodd");

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
    const x = det.bbox_x * w;
    const y = det.bbox_y * h;
    const pw = pill.pillWidth * scale;
    const ph = pill.pillHeight * scale;
    const pillY = y - ph > 0 ? y - ph : y;

    // Pill background
    ctx.beginPath();
    roundedRectPath(ctx, x, pillY, pw, ph, BBOX_CORNER_RADIUS * scale);
    ctx.fillStyle = PILL_BG;
    ctx.fill();

    // Color dot
    ctx.beginPath();
    ctx.arc(
      x + (PILL_PAD_X + DOT_R) * scale,
      pillY + ph / 2,
      DOT_R * scale,
      0, Math.PI * 2,
    );
    ctx.fillStyle = pill.color;
    ctx.fill();

    // Text
    if (pill.hasLabel) {
      ctx.font = `${FONT_SM * scale}px Arial, sans-serif`;
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.fillText(pill.categoryText, x + TEXT_START_X * scale, pillY + PILL_PAD_Y * scale);

      ctx.font = `bold ${FONT_LG * scale}px Arial, sans-serif`;
      ctx.fillStyle = "white";
      ctx.fillText(pill.labelText, x + TEXT_START_X * scale, pillY + (PILL_PAD_Y + FONT_SM + LINE_GAP) * scale);
    } else {
      ctx.font = `bold ${FONT_LG * scale}px Arial, sans-serif`;
      ctx.fillStyle = "white";
      ctx.fillText(pill.categoryText, x + TEXT_START_X * scale, pillY + PILL_PAD_Y * scale);
    }
  }

  ctx.restore();
}

// ── Component ─────────────────────────────────────────────────────

export function VideoPlayer({ file, detectionThreshold, sourceVideoId, allDetections, exportFnRef }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [currentFrame, setCurrentFrame] = useState<number>(0);
  const [displayWidth, setDisplayWidth] = useState(0);
  const [isExporting, setIsExporting] = useState(false);
  const animFrameRef = useRef<number>(0);
  const exportAbortRef = useRef(false);

  const videoFileId = sourceVideoId ?? file.id;
  const videoUrl = `${API_BASE_URL}/api/files/${videoFileId}/video`;
  const frameRate = file.frame_rate || 30;
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

  // Group detections by frame_number, filtered by threshold
  const detectionsByFrame = useMemo(() => {
    const map = new Map<number, DetectionResponse[]>();
    for (const d of detections) {
      if (d.confidence < detectionThreshold || d.frame_number == null) continue;
      const existing = map.get(d.frame_number);
      if (existing) {
        existing.push(d);
      } else {
        map.set(d.frame_number, [d]);
      }
    }
    return map;
  }, [detections, detectionThreshold]);

  // Find detections for the current video frame, persisting the last-seen
  // detections through frames that have none so boxes don't disappear between
  // analyzed frames.
  const lastDetectionsRef = useRef<DetectionResponse[]>([]);
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
    let lastDets: DetectionResponse[] = [];
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
      const parts = file.file_path.split("/");
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
    };

    exportAbortRef.current = false;
    setIsExporting(true);

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

  // Build spotlight SVG path: outer rect with rounded-rect holes for each detection
  const spotlightPath = useMemo(() => {
    if (currentDetections.length === 0) return "";
    let d = `M0,0H${imgW}V${imgH}H0Z`;
    for (const det of currentDetections) {
      d += svgRoundedRectPath(
        det.bbox_x * imgW,
        det.bbox_y * imgH,
        det.bbox_width * imgW,
        det.bbox_height * imgH,
        BBOX_CORNER_RADIUS * s,
      );
    }
    return d;
  }, [currentDetections, imgW, imgH, s]);

  return (
    <div className="relative w-full h-full flex items-center justify-center">
      {/* Video + SVG overlay container */}
      <div
        ref={containerRef}
        className="relative max-w-full max-h-full"
        style={{ aspectRatio: imgW / imgH }}
      >
        <video
          ref={videoRef}
          src={videoUrl}
          crossOrigin="anonymous"
          controls
          controlsList="nodownload"
          className="w-full h-full object-contain"
          onPlay={handlePlay}
          onPause={handlePause}
          onSeeked={handleSeeked}
          onEnded={handlePause}
        />

        {/* Recording indicator */}
        {isExporting && (
          <div className="absolute top-2 left-2 z-10 flex items-center gap-1.5 bg-black/70 text-white text-xs px-2 py-1 rounded">
            <span className="h-2 w-2 rounded-full bg-red-500 animate-pulse" />
            Recording…
          </div>
        )}

        {/* SVG overlay: spotlight + bboxes + labels */}
        {currentDetections.length > 0 && overlayOpacity > 0 && (
          <svg
            style={{ opacity: overlayOpacity, transition: "opacity 0.05s linear" }}
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox={`0 0 ${imgW} ${imgH}`}
            preserveAspectRatio="xMidYMid meet"
          >
            {/* Spotlight dim overlay */}
            <path fillRule="evenodd" d={spotlightPath} fill={DIM_FILL} />

            {/* Bounding boxes */}
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
                  opacity={BBOX_OPACITY}
                />
              );
            })}

            {/* Label pills — rendered at screen-pixel sizes via scale(s) */}
            {currentDetections.map((det) => {
              const pill = computePillLayout(det);
              const x = det.bbox_x * imgW;
              const y = det.bbox_y * imgH;
              const pillH = pill.pillHeight * s;
              const pillY = y - pillH > 0 ? y - pillH : y;

              return (
                <g key={`label-${det.id}`} transform={`translate(${x}, ${pillY}) scale(${s})`}>
                  <rect
                    x={0}
                    y={0}
                    width={pill.pillWidth}
                    height={pill.pillHeight}
                    rx={BBOX_CORNER_RADIUS}
                    fill={PILL_BG}
                  />
                  <circle
                    cx={PILL_PAD_X + DOT_R}
                    cy={pill.pillHeight / 2}
                    r={DOT_R}
                    fill={pill.color}
                  />
                  {pill.hasLabel ? (
                    <>
                      <text
                        x={TEXT_START_X}
                        y={PILL_PAD_Y}
                        fill="rgba(255,255,255,0.7)"
                        fontSize={FONT_SM}
                        fontFamily="Arial, sans-serif"
                        dominantBaseline="hanging"
                      >
                        {pill.categoryText}
                      </text>
                      <text
                        x={TEXT_START_X}
                        y={PILL_PAD_Y + FONT_SM + LINE_GAP}
                        fill="white"
                        fontSize={FONT_LG}
                        fontWeight="bold"
                        fontFamily="Arial, sans-serif"
                        dominantBaseline="hanging"
                      >
                        {pill.labelText}
                      </text>
                    </>
                  ) : (
                    <text
                      x={TEXT_START_X}
                      y={PILL_PAD_Y}
                      fill="white"
                      fontSize={FONT_LG}
                      fontWeight="bold"
                      fontFamily="Arial, sans-serif"
                      dominantBaseline="hanging"
                    >
                      {pill.categoryText}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        )}
      </div>
    </div>
  );
}
