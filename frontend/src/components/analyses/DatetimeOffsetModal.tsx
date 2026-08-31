/**
 * Datetime Offset Modal
 *
 * Lets users correct camera firmware clock errors before analysis.
 * Shows sample images with extracted EXIF dates so users can visually
 * compare with the burned-in pixel date, then apply a bulk offset.
 *
 * Two modes:
 * - "Set correct date": pick a reference image, type the real date,
 *   system computes the offset automatically.
 * - "Manual offset": enter days/hours/minutes directly with quick
 *   buttons for common corrections (AM/PM flip, timezone shift).
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { CalendarDays, ChevronLeft, ChevronRight, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { API_BASE_URL } from "@/lib/api-client";
import { msToNaiveString, naiveDateMs } from "@/lib/datetime";
import { formatOffset } from "@/lib/utils";

interface SampleFile {
  path: string;
  file_datetime: string | null;
}

interface DatetimeOffsetModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sampleFiles: SampleFile[];
  folderPath: string;
  currentOffsetSeconds: number;
  /** The whole-deployment offset, or one camera's own offset when
   *  `camera` is set (on top of the whole-deployment one). */
  onApply: (offsetSeconds: number, camera: string | null) => void;
  /** The folder's camera-style subfolders (from the scan). With two or
   *  more and `pairedCameras` on, a dropdown lets one camera be corrected
   *  on its own; with `pairedCameras` off, a one-line hint says where the
   *  tick is. */
  cameras?: string[];
  /** Whether the deployment is (being) marked as paired cameras. */
  pairedCameras?: boolean;
  /** Current per-camera offsets, keyed like `cameras`. */
  cameraOffsets?: Record<string, number>;
  /** Mirror of the folder-scan opt-in. Without it this modal would show
   *  "unknown" for every file in a folder that has no capture dates, so
   *  the offset could never be worked out for exactly the folders that
   *  need it most. */
  useFileMtimeFallback?: boolean;
}

/** Format a datetime string for display (shorter than ISO). */
function fmtDate(iso: string | null): string {
  if (!iso) return "unknown";
  const d = new Date(iso);
  return d.toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/** Reference datetime plus offset, as a datetime-local input value.
 *
 * All arithmetic runs in naive wall-clock space (naiveDateMs), matching
 * how the backend applies the offset. Browser-local epoch math here used
 * to shift the result an hour when the offset crossed a DST transition
 * in the viewer's timezone.
 */
function shiftedDateStr(iso: string, offsetSeconds: number): string {
  const ms = naiveDateMs(iso);
  return ms === null ? "" : msToNaiveString(ms + offsetSeconds * 1000);
}

export function DatetimeOffsetModal({
  open,
  onOpenChange,
  sampleFiles,
  folderPath,
  currentOffsetSeconds,
  onApply,
  useFileMtimeFallback = false,
  cameras,
  pairedCameras = false,
  cameraOffsets = {},
}: DatetimeOffsetModalProps) {
  const [referenceIndex, setReferenceIndex] = useState(0);
  // The value being edited: the whole-deployment offset, or the selected
  // camera's own extra offset. One rule keeps the two from double
  // counting: the shown date is raw + fixedSeconds + offsetSeconds, where
  // fixedSeconds is the whole-deployment offset while a camera is
  // selected and zero otherwise.
  const [offsetSeconds, setOffsetSeconds] = useState(currentOffsetSeconds);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);
  const fixedSeconds = selectedCamera ? currentOffsetSeconds : 0;
  // Stale keys (a camera folder that disappeared) stay listed so their
  // value can still be cleared.
  const cameraChoices = [...new Set([...(cameras ?? []), ...Object.keys(cameraOffsets)])].sort();
  const showCameras = pairedCameras && cameraChoices.length >= 2;
  // The Adjust dates link sits above the Options section, so someone with
  // dependent cameras can land here before ticking Paired cameras. One
  // muted line, only when the folder looks like it could be a pair.
  const showPairedHint = !pairedCameras && (cameras?.length ?? 0) >= 2;
  const visibleFiles = selectedCamera
    ? sampleFiles.filter((f) => f.path.split(/[\\/]/)[0] === selectedCamera)
    : sampleFiles;

  // Image zoom + pan state
  const [imageZoom, setImageZoom] = useState(1);
  const [panX, setPanX] = useState(0);
  const [panY, setPanY] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const dragStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 });
  const zoomContainerRef = useRef<HTMLDivElement>(null);

  // The user's corrected datetime string (from the date picker)
  const [correctedDateStr, setCorrectedDateStr] = useState("");

  // Attach wheel handler with { passive: false } so preventDefault works
  // (React registers wheel listeners as passive by default). Re-runs when
  // the modal opens because Radix Dialog remounts content on open, so the
  // ref points to a fresh DOM element that needs a new listener.
  useEffect(() => {
    if (!open) return;
    // Small delay to let Radix Dialog finish mounting the portal content
    const timer = setTimeout(() => {
      const el = zoomContainerRef.current;
      if (!el) return;

      const handleWheel = (e: WheelEvent) => {
        e.preventDefault();
        const rect = el.getBoundingClientRect();
        const w = el.clientWidth;
        const h = el.clientHeight;
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        setImageZoom((oldZoom) => {
          const newZoom = Math.min(5, Math.max(1, oldZoom + (e.deltaY < 0 ? 0.5 : -0.5)));
          if (newZoom === 1) {
            setPanX(0);
            setPanY(0);
          } else {
            const ratio = newZoom / oldZoom;
            setPanX((px) => Math.min(0, Math.max(w * (1 - newZoom), mx - (mx - px) * ratio)));
            setPanY((py) => Math.min(0, Math.max(h * (1 - newZoom), my - (my - py) * ratio)));
          }
          return newZoom;
        });
      };

      el.addEventListener("wheel", handleWheel, { passive: false });
      // Store cleanup on the element so the outer cleanup can find it
      (el as any)._wheelCleanup = () => el.removeEventListener("wheel", handleWheel);
    }, 50);

    return () => {
      clearTimeout(timer);
      const el = zoomContainerRef.current;
      if (el && (el as any)._wheelCleanup) {
        (el as any)._wheelCleanup();
        delete (el as any)._wheelCleanup;
      }
    };
  }, [open, referenceIndex]); // eslint-disable-line react-hooks/exhaustive-deps

  // Lazily fetched datetimes (index → ISO string or null).
  // Datetimes are NOT included in the scan response (too slow for 10k+
  // files). Instead we fetch on demand as the user navigates.
  const [dateCache, setDateCache] = useState<Record<number, string | null>>({});
  const [dateFetching, setDateFetching] = useState(false);

  const currentFile = visibleFiles[referenceIndex] ?? null;
  const currentDatetime = dateCache[referenceIndex] ?? null;

  // Fetch datetime for the current image on navigation
  useEffect(() => {
    if (!open || !currentFile) return;
    // Already cached
    if (referenceIndex in dateCache) return;

    let cancelled = false;
    setDateFetching(true);

    fetch(
      `${API_BASE_URL}/api/deployments/file-datetime` +
        `?folder=${encodeURIComponent(folderPath)}` +
        `&file=${encodeURIComponent(currentFile.path)}` +
        (useFileMtimeFallback ? "&use_file_mtime_fallback=true" : ""),
    )
      .then((r) => r.json())
      .then((data) => {
        if (cancelled) return;
        setDateCache((prev) => ({ ...prev, [referenceIndex]: data.file_datetime }));
      })
      .catch(() => {
        if (!cancelled) {
          setDateCache((prev) => ({ ...prev, [referenceIndex]: null }));
        }
      })
      .finally(() => {
        if (!cancelled) setDateFetching(false);
      });

    return () => { cancelled = true; };
  }, [open, referenceIndex, currentFile, useFileMtimeFallback]); // eslint-disable-line react-hooks/exhaustive-deps

  // Sync the corrected date display whenever the current image's datetime
  // becomes available (from cache or fresh fetch) or when navigating to a
  // different image. NOT triggered by offsetSeconds changes — that would
  // create a feedback loop with user edits to the date picker.
  useEffect(() => {
    if (currentDatetime) {
      setCorrectedDateStr(shiftedDateStr(currentDatetime, fixedSeconds + offsetSeconds));
    } else {
      setCorrectedDateStr("");
    }
  }, [currentDatetime]); // eslint-disable-line react-hooks/exhaustive-deps

  // Re-initialize working state when the modal opens
  useEffect(() => {
    if (open) {
      setSelectedCamera(null);
      setOffsetSeconds(currentOffsetSeconds);
      setReferenceIndex(0);
      setImageZoom(1);
      setPanX(0);
      setPanY(0);
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  // Clear datetime cache when the folder changes (different files at same
  // indices), or when the file-date fallback is toggled: the cached values
  // were fetched under the other rule and would all read as unknown.
  useEffect(() => {
    setDateCache({});
  }, [folderPath, useFileMtimeFallback]);

  const handleCameraChange = useCallback(
    (value: string) => {
      const cam = value === "__all__" ? null : value;
      setSelectedCamera(cam);
      setOffsetSeconds(cam ? (cameraOffsets[cam] ?? 0) : currentOffsetSeconds);
      setReferenceIndex(0);
      // Clear the cache here, not in an effect keyed on selectedCamera:
      // the fetch effect runs before such an effect and would see the old
      // camera's entry at index 0, skip the fetch, and then lose it to
      // the wipe, leaving the new camera's date unknown forever.
      setDateCache({});
      setImageZoom(1);
      setPanX(0);
      setPanY(0);
    },
    [cameraOffsets, currentOffsetSeconds],
  );

  // Sync offset from corrected date. Diffed in naive wall-clock space:
  // the backend applies the offset to naive datetimes, so an epoch diff
  // taken across a DST boundary would be an hour off what the user typed.
  const handleCorrectedDateChange = useCallback(
    (dateStr: string) => {
      setCorrectedDateStr(dateStr);
      if (!currentDatetime || !dateStr) return;
      const originalMs = naiveDateMs(currentDatetime);
      const correctedMs = naiveDateMs(dateStr);
      if (originalMs === null || correctedMs === null) return;
      setOffsetSeconds(Math.round((correctedMs - originalMs) / 1000) - fixedSeconds);
    },
    [currentDatetime, fixedSeconds],
  );

  // Quick offset buttons
  const applyQuickOffset = useCallback(
    (deltaSeconds: number) => {
      const newOffset = offsetSeconds + deltaSeconds;
      setOffsetSeconds(newOffset);
      if (currentDatetime) {
        setCorrectedDateStr(shiftedDateStr(currentDatetime, fixedSeconds + newOffset));
      }
    },
    [offsetSeconds, currentDatetime, fixedSeconds],
  );

  const handleApply = useCallback(() => {
    onApply(offsetSeconds, selectedCamera);
    onOpenChange(false);
  }, [offsetSeconds, selectedCamera, onApply, onOpenChange]);

  const handleReset = useCallback(() => {
    setOffsetSeconds(0);
    if (currentDatetime) {
      setCorrectedDateStr(shiftedDateStr(currentDatetime, fixedSeconds));
    } else {
      setCorrectedDateStr("");
    }
  }, [currentDatetime, fixedSeconds]);

  if (sampleFiles.length === 0) {
    return (
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Adjust dates</DialogTitle>
            <DialogDescription>
              No files with datetime metadata found in this folder.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Adjust dates</DialogTitle>
          <DialogDescription>
            A camera clock error affects all files equally. Scroll to zoom in
            and read the burned-in date. If it doesn't match the extracted
            date, set the correct date for one image, browse a few more to
            verify the offset looks right, then click apply.
            {showCameras
              ? " Paired cameras drift apart: pick one camera to correct it on its own, on top of the shift for the whole deployment."
              : " The same correction will be applied to all files in the deployment."}
          </DialogDescription>
        </DialogHeader>

        {showPairedHint && (
          <p className="text-xs text-muted-foreground">
            Does this deployment contain multiple dependent cameras? Tick
            Paired cameras under Options first, then each camera can get its
            own time shift.
          </p>
        )}

        {showCameras && (
          <div className="flex items-center gap-3">
            <Label className="text-xs shrink-0">Camera</Label>
            <Select value={selectedCamera ?? "__all__"} onValueChange={handleCameraChange}>
              <SelectTrigger className="h-8 w-64 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__all__">All cameras</SelectItem>
                {cameraChoices.map((cam) => (
                  <SelectItem key={cam} value={cam}>
                    Only {cam}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        <div className="flex-1 min-h-0 overflow-y-auto">
          <div className="grid grid-cols-[1fr_auto] gap-6">
            {/* Left: Single image with navigation */}
            <div className="space-y-3">
              {currentFile && (() => {
                const imgUrl = `${API_BASE_URL}/api/deployments/preview-image?folder=${encodeURIComponent(folderPath)}&file=${encodeURIComponent(currentFile.path)}`;
                return (
                  <>
                    <div className="rounded-lg overflow-hidden border border-border">
                      {/* Scroll to zoom, drag to pan. Double-click resets. */}
                      <div
                        ref={zoomContainerRef}
                        className="overflow-hidden bg-muted select-none"
                        onMouseDown={(e) => {
                          if (imageZoom <= 1) return;
                          e.preventDefault();
                          setIsDragging(true);
                          dragStart.current = { x: e.clientX, y: e.clientY, panX, panY };
                        }}
                        onMouseMove={(e) => {
                          if (!isDragging) return;
                          const el = zoomContainerRef.current;
                          if (!el) return;
                          const w = el.clientWidth;
                          const h = el.clientHeight;
                          const rawX = dragStart.current.panX + (e.clientX - dragStart.current.x);
                          const rawY = dragStart.current.panY + (e.clientY - dragStart.current.y);
                          setPanX(Math.min(0, Math.max(w * (1 - imageZoom), rawX)));
                          setPanY(Math.min(0, Math.max(h * (1 - imageZoom), rawY)));
                        }}
                        onMouseUp={() => setIsDragging(false)}
                        onMouseLeave={() => setIsDragging(false)}
                        onDoubleClick={() => {
                          setImageZoom(1);
                          setPanX(0);
                          setPanY(0);
                        }}
                      >
                        <img
                          src={imgUrl}
                          alt={currentFile.path}
                          className="w-full"
                          draggable={false}
                          style={{
                            transformOrigin: "0 0",
                            transform: `translate(${panX}px, ${panY}px) scale(${imageZoom})`,
                            transition: isDragging ? "none" : "transform 150ms",
                            cursor: imageZoom <= 1
                              ? "zoom-in"
                              : isDragging
                                ? "grabbing"
                                : "grab",
                          }}
                        />
                      </div>
                      <div className="px-2 py-1.5 text-xs text-muted-foreground truncate text-center">
                        {currentFile.path}
                      </div>
                    </div>

                    {/* Navigation */}
                    {visibleFiles.length > 1 && (
                      <div className="flex items-center justify-center gap-3">
                        <Button
                          type="button"
                          variant="outline"
                          size="icon"
                          className="h-8 w-8"
                          disabled={referenceIndex === 0}
                          onClick={() => {
                            const next = referenceIndex - 1;
                            setReferenceIndex(next);
                            const dt = dateCache[next];
                            setCorrectedDateStr(dt ? shiftedDateStr(dt, fixedSeconds + offsetSeconds) : "");
                          }}
                        >
                          <ChevronLeft className="h-4 w-4" />
                        </Button>
                        <span className="text-xs text-muted-foreground">
                          {referenceIndex + 1} of {visibleFiles.length}
                        </span>
                        <Button
                          type="button"
                          variant="outline"
                          size="icon"
                          className="h-8 w-8"
                          disabled={referenceIndex === visibleFiles.length - 1}
                          onClick={() => {
                            const next = referenceIndex + 1;
                            setReferenceIndex(next);
                            const dt = dateCache[next];
                            setCorrectedDateStr(dt ? shiftedDateStr(dt, fixedSeconds + offsetSeconds) : "");
                          }}
                        >
                          <ChevronRight className="h-4 w-4" />
                        </Button>
                      </div>
                    )}
                  </>
                );
              })()}
            </div>

            {/* Right: Controls */}
            <div className="w-72 flex flex-col">
              <div className="space-y-5">
              {/* Extracted date */}
              <div className="space-y-1.5">
                <Label className="text-xs">Date extracted from file</Label>
                <Input
                  type="text"
                  readOnly
                  value={
                    dateFetching
                      ? "Loading..."
                      : currentDatetime
                        ? fmtDate(currentDatetime)
                        : "Date not available for this file"
                  }
                  className="bg-muted text-sm"
                />
              </div>

              {/* Common fixes */}
              <div className="space-y-1.5">
                <Label className="text-xs">Common fixes</Label>
                <div className="grid grid-cols-2 gap-1.5">
                  <Button type="button" variant="outline" size="sm" onClick={() => applyQuickOffset(-12 * 3600)}>
                    -12 hours
                  </Button>
                  <Button type="button" variant="outline" size="sm" onClick={() => applyQuickOffset(12 * 3600)}>
                    +12 hours
                  </Button>
                  <Button type="button" variant="outline" size="sm" onClick={() => applyQuickOffset(-3600)}>
                    -1 hour
                  </Button>
                  <Button type="button" variant="outline" size="sm" onClick={() => applyQuickOffset(3600)}>
                    +1 hour
                  </Button>
                  {/* Seconds matter for paired cameras, whose clocks drift
                      by seconds; offered everywhere so the modal is one. */}
                  <Button type="button" variant="outline" size="sm" onClick={() => applyQuickOffset(-1)}>
                    -1 second
                  </Button>
                  <Button type="button" variant="outline" size="sm" onClick={() => applyQuickOffset(1)}>
                    +1 second
                  </Button>
                </div>
              </div>

              {/* Set correct date */}
              <div className="space-y-3">
                <div className="space-y-1.5">
                  <Label className="text-xs">
                    What should the date actually be?
                  </Label>
                  {/* Formatted display with a hidden datetime-local input.
                      Clicking the display opens the native browser picker
                      via showPicker(). The formatted text matches the
                      "Date extracted from file" field above. */}
                  {/* Without an extracted date there is nothing to compute
                      the offset against, so the picker is disabled rather
                      than silently ignoring the picked date. */}
                  <button
                    type="button"
                    disabled={!currentDatetime}
                    className="flex h-9 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-1 text-sm text-left hover:bg-muted/50 transition-colors disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-background"
                    onClick={() => {
                      const input = document.getElementById("datetime-offset-picker") as HTMLInputElement;
                      input?.showPicker();
                    }}
                  >
                    <span>
                      {correctedDateStr
                        ? fmtDate(correctedDateStr)
                        : "Click to set date"}
                    </span>
                    <CalendarDays className="h-4 w-4 text-muted-foreground shrink-0" />
                  </button>
                  <input
                    id="datetime-offset-picker"
                    type="datetime-local"
                    step="1"
                    value={correctedDateStr}
                    onChange={(e) => handleCorrectedDateChange(e.target.value)}
                    className="sr-only"
                    tabIndex={-1}
                  />
                  {!currentDatetime && !dateFetching && (
                    <p className="text-xs text-muted-foreground">
                      No date could be read from this file. Browse to a file
                      with a date, or use the buttons above.
                    </p>
                  )}
                </div>
              </div>

              {/* Offset summary */}
              <div className={`rounded-md border p-3 ${
                offsetSeconds !== 0 ? "border-primary bg-primary/5" : ""
              }`}>
                <div className="flex items-start justify-between gap-2">
                  <div className="space-y-1">
                    <div className="text-xs text-muted-foreground">
                      {selectedCamera
                        ? `Files of ${selectedCamera} get an extra shift of`
                        : "All dates will be shifted by"}
                    </div>
                    <div className="text-sm font-medium">
                      {offsetSeconds === 0
                        ? "No change"
                        : formatOffset(offsetSeconds)}
                    </div>
                  </div>
                  {offsetSeconds !== 0 && (
                    <button
                      type="button"
                      onClick={handleReset}
                      className="text-muted-foreground hover:text-foreground p-1 rounded-md hover:bg-muted transition-colors shrink-0"
                      title="Clear offset"
                    >
                      <RotateCcw className="h-4 w-4" />
                    </button>
                  )}
                </div>
              </div>

              </div>

              {/* Spacer pushes buttons to bottom */}
              <div className="flex-1" />

              {/* Actions — stuck to bottom */}
              <div className="mt-5 grid grid-cols-2 gap-2">
                <Button variant="outline" onClick={() => onOpenChange(false)}>
                  Cancel
                </Button>
                <Button onClick={handleApply}>Apply offset</Button>
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
