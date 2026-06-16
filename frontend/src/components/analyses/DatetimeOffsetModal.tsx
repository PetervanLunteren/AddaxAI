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
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { API_BASE_URL } from "@/lib/api-client";
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
  onApply: (offsetSeconds: number) => void;
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

/** Convert a Date to a local datetime-local input value (YYYY-MM-DDTHH:MM:SS). */
function toDatetimeLocal(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export function DatetimeOffsetModal({
  open,
  onOpenChange,
  sampleFiles,
  folderPath,
  currentOffsetSeconds,
  onApply,
}: DatetimeOffsetModalProps) {
  const [referenceIndex, setReferenceIndex] = useState(0);
  const [offsetSeconds, setOffsetSeconds] = useState(currentOffsetSeconds);

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

  const currentFile = sampleFiles[referenceIndex] ?? null;
  const currentDatetime = dateCache[referenceIndex] ?? null;

  // Fetch datetime for the current image on navigation
  useEffect(() => {
    if (!open || !currentFile) return;
    // Already cached
    if (referenceIndex in dateCache) return;

    let cancelled = false;
    setDateFetching(true);

    fetch(
      `${API_BASE_URL}/api/deployments/file-datetime?folder=${encodeURIComponent(folderPath)}&file=${encodeURIComponent(currentFile.path)}`,
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
  }, [open, referenceIndex, currentFile]); // eslint-disable-line react-hooks/exhaustive-deps

  // Sync the corrected date display whenever the current image's datetime
  // becomes available (from cache or fresh fetch) or when navigating to a
  // different image. NOT triggered by offsetSeconds changes — that would
  // create a feedback loop with user edits to the date picker.
  useEffect(() => {
    if (currentDatetime) {
      setCorrectedDateStr(
        toDatetimeLocal(
          new Date(new Date(currentDatetime).getTime() + offsetSeconds * 1000),
        ),
      );
    } else {
      setCorrectedDateStr("");
    }
  }, [currentDatetime]); // eslint-disable-line react-hooks/exhaustive-deps

  // Re-initialize working state when the modal opens
  useEffect(() => {
    if (open) {
      setOffsetSeconds(currentOffsetSeconds);
      setReferenceIndex(0);
      setImageZoom(1);
      setPanX(0);
      setPanY(0);
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  // Clear datetime cache when the folder changes (different files at same indices)
  useEffect(() => {
    setDateCache({});
  }, [folderPath]);

  // Sync offset from corrected date
  const handleCorrectedDateChange = useCallback(
    (dateStr: string) => {
      setCorrectedDateStr(dateStr);
      if (!currentDatetime || !dateStr) return;
      const original = new Date(currentDatetime);
      const corrected = new Date(dateStr);
      if (Number.isNaN(corrected.getTime())) return;
      setOffsetSeconds(
        Math.round((corrected.getTime() - original.getTime()) / 1000),
      );
    },
    [currentDatetime],
  );

  // Quick offset buttons
  const applyQuickOffset = useCallback(
    (deltaSeconds: number) => {
      const newOffset = offsetSeconds + deltaSeconds;
      setOffsetSeconds(newOffset);
      if (currentDatetime) {
        const ref = new Date(currentDatetime);
        setCorrectedDateStr(toDatetimeLocal(new Date(ref.getTime() + newOffset * 1000)));
      }
    },
    [offsetSeconds, currentDatetime],
  );

  const handleApply = useCallback(() => {
    onApply(offsetSeconds);
    onOpenChange(false);
  }, [offsetSeconds, onApply, onOpenChange]);

  const handleReset = useCallback(() => {
    setOffsetSeconds(0);
    if (currentDatetime) {
      setCorrectedDateStr(toDatetimeLocal(new Date(currentDatetime)));
    } else {
      setCorrectedDateStr("");
    }
  }, [currentDatetime]);

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
            verify the offset looks right, then click apply. The same
            correction will be applied to all files in the deployment.
          </DialogDescription>
        </DialogHeader>

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
                    {sampleFiles.length > 1 && (
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
                            if (dt) {
                              setCorrectedDateStr(toDatetimeLocal(new Date(new Date(dt).getTime() + offsetSeconds * 1000)));
                            } else {
                              setCorrectedDateStr("");
                            }
                          }}
                        >
                          <ChevronLeft className="h-4 w-4" />
                        </Button>
                        <span className="text-xs text-muted-foreground">
                          {referenceIndex + 1} of {sampleFiles.length}
                        </span>
                        <Button
                          type="button"
                          variant="outline"
                          size="icon"
                          className="h-8 w-8"
                          disabled={referenceIndex === sampleFiles.length - 1}
                          onClick={() => {
                            const next = referenceIndex + 1;
                            setReferenceIndex(next);
                            const dt = dateCache[next];
                            if (dt) {
                              setCorrectedDateStr(toDatetimeLocal(new Date(new Date(dt).getTime() + offsetSeconds * 1000)));
                            } else {
                              setCorrectedDateStr("");
                            }
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
                  <button
                    type="button"
                    className="flex h-9 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-1 text-sm text-left hover:bg-muted/50 transition-colors"
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
                </div>
              </div>

              {/* Offset summary */}
              <div className={`rounded-md border p-3 ${
                offsetSeconds !== 0 ? "border-primary bg-primary/5" : ""
              }`}>
                <div className="flex items-start justify-between gap-2">
                  <div className="space-y-1">
                    <div className="text-xs text-muted-foreground">
                      All dates will be shifted by
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
