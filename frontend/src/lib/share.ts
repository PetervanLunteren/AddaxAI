/**
 * A share as a whole percentage for people to read, used wherever the
 * dashboard splits a total (demographics bars, the Files tile).
 *
 * A real but tiny share prints as "<1%", never as "0%": "0% people" next
 * to six photos of people reads as none.
 */

export function formatShare(share: number): string {
  if (share > 0 && share < 0.005) return "<1%";
  return `${Math.round(share * 100)}%`;
}

/**
 * "98% animals · 2% vehicles · <1% people": the parts of a total that
 * exist, largest first. Parts with nothing in them are left out, because
 * a list of zeros is noise and pushes the line onto a second row.
 */
export function shareNote(parts: { label: string; count: number }[]): string {
  const total = parts.reduce((sum, p) => sum + p.count, 0);
  if (total === 0) return "";
  return parts
    .filter((p) => p.count > 0)
    .sort((a, b) => b.count - a.count)
    .map((p) => `${formatShare(p.count / total)} ${p.label}`)
    .join(" · ");
}
