/**
 * The one line the app says when a frame of a clip is slow to arrive:
 * under the open track's spinner, and inside `FrameWait`, the chip the
 * stills and the player show from the first moment. Mounted with the
 * wait and unmounted with it, so every new wait starts its clock at
 * zero and the line only shows once the wait has lasted SLOW_WAIT_MS.
 *
 * Why a timer and not a drive check: the drive type is not the signal,
 * the time spent reading is. The same external SSD was useless on one
 * port and fine on another (2026-11-02), and a network share can go
 * either way. A measured wait is honest in every case.
 */

import { type ReactNode, useEffect, useState } from "react";

import { cn } from "@/lib/utils";

export const SLOW_WAIT_MS = 5000;

export const SLOW_DRIVE_HINT =
  "Still working. The frames are read from the video file. A faster drive or a direct connection speeds this up.";

export function SlowHint({
  afterMs = SLOW_WAIT_MS,
  className,
  children = SLOW_DRIVE_HINT,
}: {
  afterMs?: number;
  className?: string;
  children?: ReactNode;
}) {
  const [show, setShow] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => setShow(true), afterMs);
    return () => clearTimeout(timer);
  }, [afterMs]);
  return show ? (
    <p className={cn("text-xs text-muted-foreground text-center", className)}>{children}</p>
  ) : null;
}
