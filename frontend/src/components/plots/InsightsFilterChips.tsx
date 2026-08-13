/**
 * Active filter chips for the Insights pages.
 *
 * Mirrors the chip row used by the verify tabs, but driven by a generic
 * list of `{ key, label, onRemove }` entries because each insights page
 * has its own filter shape. Pages compose their chips from the helpers
 * exported below, then pass them in alongside an `onClearAll` callback
 * that resets data filters while leaving display-mode controls (sort,
 * density, view mode, base layer, etc.) untouched.
 */

import { X } from "lucide-react";
import { Badge } from "../ui/badge";
import { NO_SITE_SENTINEL } from "../../lib/filter-url";

export interface FilterChip {
  key: string;
  label: string;
  onRemove: () => void;
}

interface InsightsFilterChipsProps {
  chips: FilterChip[];
  onClearAll: () => void;
}

export function InsightsFilterChips({
  chips,
  onClearAll,
}: InsightsFilterChipsProps) {
  if (chips.length === 0) return null;

  return (
    <div className="flex items-center gap-2 flex-wrap">
      {chips.map((chip) => (
        <Badge
          key={chip.key}
          variant="secondary"
          className="text-xs gap-1 pr-1"
        >
          {chip.label}
          <button
            type="button"
            onClick={chip.onRemove}
            className="ml-0.5 rounded-full hover:bg-black/10 p-0.5"
            aria-label={`Remove filter: ${chip.label}`}
          >
            <X className="h-3 w-3" />
          </button>
        </Badge>
      ))}
      <button
        type="button"
        onClick={onClearAll}
        className="text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        Clear all
      </button>
    </div>
  );
}

/**
 * Card every Insights filter bar wears.
 *
 * Owns the card styling and, more importantly, where the chip row sits:
 * inside the card, under the controls, the way the verify tabs already
 * do it. Left to each page, the chips ended up as a sibling floating
 * below the bar on five pages and inside it everywhere else.
 *
 * Bars pass their controls as `children` and the page's chips through.
 */
export function InsightsFilterBarShell({
  chips,
  onClearAll,
  children,
}: {
  chips: FilterChip[];
  onClearAll: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border bg-card pt-2 pb-3 px-3 space-y-4">
      {children}
      <InsightsFilterChips chips={chips} onClearAll={onClearAll} />
    </div>
  );
}

/** Build a {site_id -> name} map. The NO_SITE sentinel is always
 *  mapped so chips display "(no site)" the same way the multi-select
 *  does, even if the user is sharing a URL across projects. */
export function buildSiteNameMap(
  sites: { id: string; name: string }[] | undefined,
): Record<string, string> {
  const map: Record<string, string> = { [NO_SITE_SENTINEL]: "(no site)" };
  for (const s of sites ?? []) map[s.id] = s.name;
  return map;
}

/** One chip per site when ≤ 2 are selected, else a single "N sites" chip. */
export function siteChips(
  siteIds: string[] | undefined,
  siteNames: Record<string, string>,
  setSiteIds: (next: string[]) => void,
): FilterChip[] {
  if (!siteIds?.length) return [];
  if (siteIds.length <= 2) {
    return siteIds.map((id) => ({
      key: `site-${id}`,
      label: `Site: ${siteNames[id] ?? id}`,
      onRemove: () => setSiteIds(siteIds.filter((s) => s !== id)),
    }));
  }
  return [
    {
      key: "sites",
      label: `${siteIds.length} sites`,
      onRemove: () => setSiteIds([]),
    },
  ];
}

/** From / To chips for a date range. Either or both may be set. */
export function dateChips(
  dateFrom: string | null | undefined,
  dateTo: string | null | undefined,
  onClearFrom: () => void,
  onClearTo: () => void,
): FilterChip[] {
  const out: FilterChip[] = [];
  if (dateFrom) {
    out.push({
      key: "date-from",
      label: `From: ${dateFrom}`,
      onRemove: onClearFrom,
    });
  }
  if (dateTo) {
    out.push({
      key: "date-to",
      label: `To: ${dateTo}`,
      onRemove: onClearTo,
    });
  }
  return out;
}

/** One chip per label when ≤ 2 are selected, else a single "N labels" chip. */
export function labelChips(
  labels: string[] | undefined,
  displayLabels: Record<string, string> | undefined,
  setLabels: (next: string[]) => void,
): FilterChip[] {
  if (!labels?.length) return [];
  if (labels.length <= 2) {
    return labels.map((id) => ({
      key: `label-${id}`,
      label: displayLabels?.[id] ?? id,
      onRemove: () => setLabels(labels.filter((s) => s !== id)),
    }));
  }
  return [
    {
      key: "labels",
      label: `${labels.length} labels`,
      onRemove: () => setLabels([]),
    },
  ];
}
