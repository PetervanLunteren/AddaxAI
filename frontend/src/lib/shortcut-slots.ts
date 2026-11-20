/**
 * Pure helpers for the quick-label slots (`hooks/useShortcutLabels`).
 * Kept apart from the hook so they can be tested without the API client.
 */

/** The slots without one key. The other keys keep their numbers: a
 *  person's fingers know them. */
export function withoutSlot<T>(labels: Record<number, T>, slot: number): Record<number, T> {
  const next = { ...labels };
  delete next[slot];
  return next;
}
