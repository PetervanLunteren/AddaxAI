/**
 * The one undo history behind the Labels page.
 *
 * Two rules, no exceptions. Every undoable action (verify, unverify,
 * relabel, mark false, mark unknown, on boxes or on files) returns the
 * before-state of what it changed; that snapshot is the undo entry.
 * Undo sends the top entry to `POST /api/undo`, which writes it back
 * verbatim, so undo puts back exactly what was there. An action that
 * cannot be undone (drawing or moving a box) calls `clear()`, so Cmd+Z
 * never reverses an older, unrelated change.
 *
 * One stack per tab, shared with that tab's viewer. Session-scoped, in
 * memory: a tab switch unmounts it, like before.
 */

import { useCallback, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { undoApi } from "../../api/undo";
import type { DetectionResponse, UndoSnapshot } from "../../api/types";
import { isEmptySnapshot } from "../../lib/undo-snapshot";

export interface UndoResult {
  /** The entry that was undone. */
  snapshot: UndoSnapshot;
  /** The boxes the server wrote back, for patching a grid in place. */
  detections: DetectionResponse[];
}

export interface UndoStack {
  /** Record an action's before-state. Empty snapshots are ignored. */
  push: (snapshot: UndoSnapshot) => void;
  /** Undo the last action. Resolves null when there was nothing to undo,
   *  the previous undo is still running, or the restore failed (toasted). */
  undo: () => Promise<UndoResult | null>;
  /** Forget the history, after an action that cannot be undone. */
  clear: () => void;
  canUndo: boolean;
}

/**
 * @param onRestored runs after every successful undo, before `undo()`
 *   resolves: the owner refreshes or patches its view here. The latest
 *   callback is used, so the caller need not memoise it.
 */
export function useUndoStack(onRestored: (result: UndoResult) => void): UndoStack {
  const stackRef = useRef<UndoSnapshot[]>([]);
  const busyRef = useRef(false);
  const [canUndo, setCanUndo] = useState(false);
  const onRestoredRef = useRef(onRestored);
  onRestoredRef.current = onRestored;

  const push = useCallback((snapshot: UndoSnapshot) => {
    if (isEmptySnapshot(snapshot)) return;
    stackRef.current.push(snapshot);
    setCanUndo(true);
  }, []);

  const clear = useCallback(() => {
    stackRef.current = [];
    setCanUndo(false);
  }, []);

  const undo = useCallback(async (): Promise<UndoResult | null> => {
    const snapshot = stackRef.current[stackRef.current.length - 1];
    if (!snapshot || busyRef.current) return null;
    busyRef.current = true;
    try {
      const { detections } = await undoApi.restore(snapshot);
      // Popped only now: a failed restore leaves the entry for a retry.
      stackRef.current.pop();
      setCanUndo(stackRef.current.length > 0);
      if (detections.length < snapshot.detections.length) {
        toast.info("Some of those boxes no longer exist. The rest were put back.");
      }
      const result = { snapshot, detections };
      onRestoredRef.current(result);
      return result;
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Undo failed");
      return null;
    } finally {
      busyRef.current = false;
    }
  }, []);

  return useMemo(() => ({ push, undo, clear, canUndo }), [push, undo, clear, canUndo]);
}
