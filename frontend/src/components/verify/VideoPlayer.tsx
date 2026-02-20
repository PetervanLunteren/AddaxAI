/**
 * Video player with SVG bounding box overlays synced to the current frame.
 *
 * Renders an HTML5 <video> element with an SVG overlay that shows
 * detection bounding boxes for the current frame. Used in the
 * verification modal as an alternative to the best-frame AnnotationCanvas.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { SkipBack, SkipForward } from "lucide-react";
import { API_BASE_URL } from "../../lib/api-client";
import { getCategoryColor } from "../../lib/detection-utils";
import type { FileWithDetections, DetectionResponse } from "../../api/types";

interface VideoPlayerProps {
  file: FileWithDetections;
  detectionThreshold: number;
}

/** Browser-playable video formats. */
const PLAYABLE_FORMATS = new Set(["mp4", "m4v", "mov", "webm"]);

/** Check whether a file's video format is browser-playable. */
export function isPlayableVideo(file: FileWithDetections): boolean {
  return (
    file.file_type === "video" &&
    file.frame_rate != null &&
    PLAYABLE_FORMATS.has((file.file_format || "").toLowerCase())
  );
}

export function VideoPlayer({ file, detectionThreshold }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentFrame, setCurrentFrame] = useState<number>(0);
  const animFrameRef = useRef<number>(0);

  const videoUrl = `${API_BASE_URL}/api/files/${file.id}/video`;
  const frameRate = file.frame_rate || 30;
  const aspectRatio = (file.width_px || 16) / (file.height_px || 9);

  // Group detections by frame_number, filtered by threshold
  const detectionsByFrame = useMemo(() => {
    const map = new Map<number, DetectionResponse[]>();
    for (const d of file.detections) {
      if (d.confidence < detectionThreshold || d.frame_number == null) continue;
      const existing = map.get(d.frame_number);
      if (existing) {
        existing.push(d);
      } else {
        map.set(d.frame_number, [d]);
      }
    }
    return map;
  }, [file.detections, detectionThreshold]);

  // Sorted list of frame numbers that have detections (for prev/next navigation)
  const detectionFrames = useMemo(
    () => [...detectionsByFrame.keys()].sort((a, b) => a - b),
    [detectionsByFrame]
  );

  // Find nearest detection frame for the current video time
  const currentDetections = useMemo(() => {
    if (detectionsByFrame.size === 0) return [];
    // Exact match first
    const exact = detectionsByFrame.get(currentFrame);
    if (exact) return exact;
    // Find closest frame (within ±1 frame tolerance for rounding)
    for (const offset of [1, -1]) {
      const nearby = detectionsByFrame.get(currentFrame + offset);
      if (nearby) return nearby;
    }
    return [];
  }, [currentFrame, detectionsByFrame]);

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
    // Final sync on pause
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

  // Cleanup animation frame on unmount
  useEffect(() => {
    return () => cancelAnimationFrame(animFrameRef.current);
  }, []);

  // Jump to previous detection frame
  const jumpPrev = useCallback(() => {
    const video = videoRef.current;
    if (!video || detectionFrames.length === 0) return;
    // Find the last detection frame before currentFrame
    let target = detectionFrames[detectionFrames.length - 1];
    for (let i = detectionFrames.length - 1; i >= 0; i--) {
      if (detectionFrames[i] < currentFrame - 1) {
        target = detectionFrames[i];
        break;
      }
    }
    video.currentTime = target / frameRate;
    setCurrentFrame(target);
  }, [currentFrame, detectionFrames, frameRate]);

  // Jump to next detection frame
  const jumpNext = useCallback(() => {
    const video = videoRef.current;
    if (!video || detectionFrames.length === 0) return;
    // Find the first detection frame after currentFrame
    let target = detectionFrames[0];
    for (const frame of detectionFrames) {
      if (frame > currentFrame + 1) {
        target = frame;
        break;
      }
    }
    video.currentTime = target / frameRate;
    setCurrentFrame(target);
  }, [currentFrame, detectionFrames, frameRate]);

  return (
    <div className="relative w-full h-full flex flex-col items-center justify-center">
      {/* Video + SVG overlay container */}
      <div
        className="relative max-w-full max-h-full"
        style={{ aspectRatio }}
      >
        <video
          ref={videoRef}
          src={videoUrl}
          controls
          className="w-full h-full object-contain"
          onPlay={handlePlay}
          onPause={handlePause}
          onSeeked={handleSeeked}
          onEnded={handlePause}
        />

        {/* SVG detection overlay */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
        >
          {currentDetections.map((det) => {
            const color = getCategoryColor(det.category);
            return (
              <rect
                key={det.id}
                x={det.bbox_x}
                y={det.bbox_y}
                width={det.bbox_width}
                height={det.bbox_height}
                fill="none"
                stroke={color}
                strokeWidth={0.003}
                opacity={0.7}
              />
            );
          })}
        </svg>

        {/* Label pills - rendered in a separate SVG with proper aspect ratio
            so text isn't stretched by preserveAspectRatio="none" */}
        {currentDetections.length > 0 && (
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox={`0 0 ${file.width_px || 1} ${file.height_px || 1}`}
            preserveAspectRatio="xMidYMid meet"
          >
            {currentDetections.map((det) => {
              const color = getCategoryColor(det.category);
              const imgW = file.width_px || 1;
              const imgH = file.height_px || 1;
              const x = det.bbox_x * imgW;
              const y = det.bbox_y * imgH;
              const label = det.species
                ? `${det.species} ${((det.species_confidence ?? det.confidence) * 100).toFixed(0)}%`
                : `${det.category} ${(det.confidence * 100).toFixed(0)}%`;
              const fontSize = Math.max(12, Math.min(16, imgH * 0.015));
              const pillH = fontSize + 8;
              const pillW = label.length * fontSize * 0.6 + 16;
              const pillY = y - pillH > 0 ? y - pillH : y;

              return (
                <g key={`label-${det.id}`}>
                  <rect
                    x={x}
                    y={pillY}
                    width={pillW}
                    height={pillH}
                    rx={4}
                    fill="rgba(0,0,0,0.6)"
                  />
                  <circle
                    cx={x + 8 + 4}
                    cy={pillY + pillH / 2}
                    r={4}
                    fill={color}
                  />
                  <text
                    x={x + 8 + 4 + 4 + 5}
                    y={pillY + pillH / 2}
                    fill="white"
                    fontSize={fontSize}
                    dominantBaseline="central"
                  >
                    {label}
                  </text>
                </g>
              );
            })}
          </svg>
        )}
      </div>

      {/* Frame navigation bar */}
      {detectionFrames.length > 0 && (
        <div className="flex items-center gap-2 mt-2 text-xs text-white/80">
          <button
            onClick={jumpPrev}
            className="p-1 hover:text-white rounded transition-colors"
            title="Previous detection frame"
          >
            <SkipBack className="h-3.5 w-3.5" />
          </button>
          <span className="tabular-nums">
            Frame {currentFrame} &middot; {currentDetections.length} detection{currentDetections.length !== 1 ? "s" : ""}
          </span>
          <button
            onClick={jumpNext}
            className="p-1 hover:text-white rounded transition-colors"
            title="Next detection frame"
          >
            <SkipForward className="h-3.5 w-3.5" />
          </button>
        </div>
      )}
    </div>
  );
}
