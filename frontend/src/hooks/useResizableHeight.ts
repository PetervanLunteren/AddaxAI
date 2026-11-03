/**
 * A panel docked at the bottom of its column that a person drags taller
 * or shorter, with the height remembered. The Counts modal's filmstrip
 * and the track timeline under a video both use it, so one drag feels
 * like the other and the listener cleanup lives in one place.
 *
 * The height is state plus the cursor's travel since mousedown, never
 * measured off the panel: a lane whose content is shorter than the
 * stored height renders shorter, and measuring it turned a 5 px drag
 * on a four-row clip into a 59 px height for every clip after
 * (2026-11-03). Dragging up grows a bottom-docked panel.
 */

import {
  type MouseEvent as ReactMouseEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import { persistSetting, readSetting } from "../lib/local-settings";

export function useResizableHeight({
  storageKey,
  field,
  defaultHeight,
  minHeight,
  maxHeight,
}: {
  /** The localStorage key; with `field`, the key holds an object. */
  storageKey: string;
  field?: string;
  defaultHeight: number;
  minHeight: number;
  /** Read at drag time, so it can follow the window or a column. */
  maxHeight: () => number;
}): {
  height: number;
  startDrag: (e: ReactMouseEvent) => void;
} {
  const [height, setHeight] = useState<number>(() => {
    const stored = readSetting(storageKey, field);
    return typeof stored === "number" && Number.isFinite(stored) ? stored : defaultHeight;
  });
  // The listeners of a drag in progress, so an unmount mid-drag (the
  // modal closes under the cursor) does not leave them on the window.
  const stopRef = useRef<(() => void) | null>(null);
  useEffect(() => () => stopRef.current?.(), []);

  const startDrag = useCallback(
    (e: ReactMouseEvent) => {
      e.preventDefault();
      const startY = e.clientY;
      const startHeight = height;
      let latest = height;
      const onMove = (ev: MouseEvent) => {
        latest = Math.max(minHeight, Math.min(maxHeight(), startHeight + (startY - ev.clientY)));
        setHeight(latest);
      };
      const stop = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", stop);
        stopRef.current = null;
        persistSetting(storageKey, latest, field);
      };
      stopRef.current?.();
      stopRef.current = stop;
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", stop);
    },
    [height, minHeight, maxHeight, storageKey, field],
  );

  return { height, startDrag };
}
