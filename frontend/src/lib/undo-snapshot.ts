/**
 * Pure helpers for the before-state snapshots that undo is built on.
 *
 * A bulk action over more than 500 rows runs as several sequential
 * requests, each answering its own `before`. Merged, they are one undo
 * entry. Detections are disjoint across chunks; files are not (two
 * chunks can touch boxes of the same file), and only the first chunk
 * saw that file untouched, so the first state wins.
 */

import type { UndoSnapshot } from "../api/types";

export function emptySnapshot(): UndoSnapshot {
  return { detections: [], files: [] };
}

export function isEmptySnapshot(s: UndoSnapshot): boolean {
  return s.detections.length === 0 && s.files.length === 0;
}

export function mergeSnapshots(parts: UndoSnapshot[]): UndoSnapshot {
  const out = emptySnapshot();
  const seenDetections = new Set<string>();
  const seenFiles = new Set<string>();
  for (const part of parts) {
    for (const d of part.detections) {
      if (seenDetections.has(d.detection_id)) continue;
      seenDetections.add(d.detection_id);
      out.detections.push(d);
    }
    for (const f of part.files) {
      if (seenFiles.has(f.file_id)) continue;
      seenFiles.add(f.file_id);
      out.files.push(f);
    }
  }
  return out;
}
