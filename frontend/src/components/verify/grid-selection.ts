/**
 * Selection behaviour shared by the two Labels grids.
 *
 * Detections selects boxes and Empties selects photos, but the gestures
 * are the same and a user should only have to learn them once:
 *
 *   click            select just this one, and put the anchor here
 *   shift + click    add the range between the anchor and this one,
 *                    leaving the anchor put so the range can be redrawn
 *   cmd / ctrl click toggle this one, and move the anchor here
 *   escape           clear
 *   click background clear
 *
 * Plus: after a bulk action consumes the selection, the card that slides
 * into the freed slot becomes selected, so a repeated action never needs
 * the mouse again.
 *
 * Both are pure functions over an ordered list of ids. They own the rules
 * and nothing else, so the two grids cannot drift apart, and neither one
 * has to know how the other stores its rows.
 *
 * Note what is deliberately absent: nothing is selected on load. The grid
 * opens with an empty selection and the first thing a user clicks decides
 * the anchor.
 */

export interface SelectionResult {
  ids: Set<string>;
  /** The new anchor, which the caller stores for the next shift-click. */
  anchor: string;
}

/**
 * The selection after a click on `targetId`.
 *
 * `orderedIds` must be the grid's current visual order; the range for a
 * shift-click is read from it. An anchor that is no longer in the list
 * (filtered away since it was set) falls back to a plain click, which is
 * the honest outcome: there is no range to draw from a card that is gone.
 */
export function selectOnClick(
  orderedIds: string[],
  anchor: string | null,
  targetId: string,
  event: { shiftKey: boolean; metaKey: boolean; ctrlKey: boolean },
  current: Set<string>,
): SelectionResult {
  if (event.shiftKey && anchor) {
    const start = orderedIds.indexOf(anchor);
    const end = orderedIds.indexOf(targetId);
    if (start !== -1 && end !== -1) {
      const [lo, hi] = start < end ? [start, end] : [end, start];
      const ids = new Set(current);
      for (let i = lo; i <= hi; i++) ids.add(orderedIds[i]);
      // Anchor stays put so repeated shift-clicks redraw the range.
      return { ids, anchor };
    }
    return { ids: new Set(current), anchor };
  }

  if (event.metaKey || event.ctrlKey) {
    const ids = new Set(current);
    if (ids.has(targetId)) ids.delete(targetId);
    else ids.add(targetId);
    return { ids, anchor: targetId };
  }

  return { ids: new Set([targetId]), anchor: targetId };
}

/**
 * Which single id to select once `actedIds` have been dealt with.
 *
 * Takes the first untouched id after the acted block, so the selection
 * lands on the card sliding into the freed slot. At the tail it falls
 * back to the card just before the block. Returns null when everything
 * was acted on, and the caller should clear instead.
 *
 * Pass the order as it was *before* the action: the rows are still being
 * removed when this is called.
 */
export function nextAfterActed(
  orderedIds: string[],
  actedIds: string[],
): string | null {
  const acted = new Set(actedIds);
  let first = -1;
  let last = -1;
  for (let i = 0; i < orderedIds.length; i++) {
    if (acted.has(orderedIds[i])) {
      if (first === -1) first = i;
      last = i;
    }
  }
  if (last === -1) return null;

  for (let i = last + 1; i < orderedIds.length; i++) {
    if (!acted.has(orderedIds[i])) return orderedIds[i];
  }
  for (let i = first - 1; i >= 0; i--) {
    if (!acted.has(orderedIds[i])) return orderedIds[i];
  }
  return null;
}
