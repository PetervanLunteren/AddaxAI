/**
 * Active filter chips displayed below the filter panel.
 *
 * Shows a summary of active filters with dismiss buttons and result count.
 */

import { X } from "lucide-react";
import type { EmptyFilter, EventFilterParams } from "../../api/types";
import { Badge } from "../ui/badge";

/** True when any user filter is set on these params. Shared by every
 *  verify tab so the chip row and "no results" copy stay in sync with
 *  the chips this component actually renders. `emptyDefault` is the
 *  page's resting empty value (Counts "hide", Files "all"). */
export function hasAnyActiveFilter(
  filters: EventFilterParams,
  emptyDefault: EmptyFilter = "hide",
): boolean {
  return (
    (filters.site_ids?.length ?? 0) > 0 ||
    !!filters.date_from ||
    !!filters.date_to ||
    (filters.labels?.length ?? 0) > 0 ||
    (!!filters.verification && filters.verification !== "all") ||
    (!!filters.flagged && filters.flagged !== "all") ||
    (!!filters.favorited && filters.favorited !== "all") ||
    (!!filters.empty && filters.empty !== emptyDefault) ||
    filters.min_confidence !== undefined ||
    filters.max_confidence !== undefined ||
    filters.min_label_confidence !== undefined ||
    filters.max_label_confidence !== undefined
  );
}

interface FilterChipsProps {
  filters: EventFilterParams;
  onChange: (filters: EventFilterParams) => void;
  siteNames: Record<string, string>;
  displayLabels?: Record<string, string>;
  /** Project counting_threshold — used to detect when det range is "default". */
  detectionFloor?: number;
  /** The page's default verification value: no chip when the filter
   *  equals it (a default is not a filter). Counts defaults to "all",
   *  the Labels page to "unverified". */
  verificationDefault?: string;
  /** The page's default empty value, same rule: no chip when the filter
   *  equals it. Counts rests on "hide", the Files tab on "all". */
  emptyDefault?: EmptyFilter;
  /** Whether this view acts on the empty filter at all. The Detections
   *  tab shares the Files tab's URL state but filters per box, where
   *  "empty" means nothing, so a chip there would claim a filtering
   *  that is not happening. */
  showEmpty?: boolean;
  /** Override the verification chip wording (the Counts page uses
   *  "Confirmed" / "Unconfirmed"; defaults to "Verified" / "Unverified"). */
  verificationLabels?: Record<string, string>;
}

/** Format a raw label ID for display (e.g. "artiodactyla:unspecified" -> "Artiodactyla"). */
function formatLabel(raw: string, displayLabels?: Record<string, string>): string {
  if (displayLabels?.[raw]) return displayLabels[raw];
  const name = raw.replace(/:unspecified$/, "").replace(/_/g, " ");
  return name.charAt(0).toUpperCase() + name.slice(1);
}

const VERIFICATION_LABELS: Record<string, string> = {
  verified: "Verified",
  unverified: "Unverified",
  // Only ever a chip on the Labels page, where "unverified" is the
  // default and showing everything is the explicit deviation.
  all: "All",
};

const FLAGGED_LABELS: Record<string, string> = {
  flagged: "Flagged",
  not_flagged: "Not flagged",
};

const FAVORITED_LABELS: Record<string, string> = {
  favorited: "Liked",
  not_favorited: "Not liked",
};

const EMPTY_LABELS: Record<string, string> = {
  show_only: "Empty only",
  all: "Including empty",
  // Only ever a chip on the Files tab, where "all" is the default.
  hide: "Hiding empty",
};


export function FilterChips({
  filters,
  onChange,
  siteNames,
  displayLabels,
  detectionFloor = 0,
  verificationDefault = "all",
  emptyDefault = "hide",
  showEmpty = true,
  verificationLabels = VERIFICATION_LABELS,
}: FilterChipsProps) {
  const chips: { key: string; label: string; onRemove: () => void }[] = [];

  // Site chips
  if (filters.site_ids?.length) {
    if (filters.site_ids.length <= 2) {
      for (const id of filters.site_ids) {
        chips.push({
          key: `site-${id}`,
          label: `Site: ${siteNames[id] ?? id}`,
          onRemove: () => {
            const next = filters.site_ids!.filter((s) => s !== id);
            onChange({ ...filters, site_ids: next.length ? next : undefined });
          },
        });
      }
    } else {
      chips.push({
        key: "sites",
        label: `${filters.site_ids.length} sites`,
        onRemove: () => onChange({ ...filters, site_ids: undefined }),
      });
    }
  }

  // Date chips
  if (filters.date_from) {
    chips.push({
      key: "date-from",
      label: `From: ${filters.date_from}`,
      onRemove: () => onChange({ ...filters, date_from: undefined }),
    });
  }
  if (filters.date_to) {
    chips.push({
      key: "date-to",
      label: `To: ${filters.date_to}`,
      onRemove: () => onChange({ ...filters, date_to: undefined }),
    });
  }

  // Label chips
  if (filters.labels?.length) {
    if (filters.labels.length <= 2) {
      for (const lbl of filters.labels) {
        chips.push({
          key: `label-${lbl}`,
          label: formatLabel(lbl, displayLabels),
          onRemove: () => {
            const next = filters.labels!.filter((s) => s !== lbl);
            onChange({ ...filters, labels: next.length ? next : undefined });
          },
        });
      }
    } else {
      chips.push({
        key: "labels",
        label: `${filters.labels.length} labels`,
        onRemove: () => onChange({ ...filters, labels: undefined }),
      });
    }
  }

  // Verification chip
  if (filters.verification && filters.verification !== verificationDefault) {
    chips.push({
      key: "verification",
      label: verificationLabels[filters.verification] ?? filters.verification,
      onRemove: () => onChange({ ...filters, verification: undefined }),
    });
  }

  // Favorited chip
  if (filters.favorited && filters.favorited !== "all") {
    chips.push({
      key: "favorited",
      label: FAVORITED_LABELS[filters.favorited] ?? filters.favorited,
      onRemove: () => onChange({ ...filters, favorited: undefined }),
    });
  }

  // Flagged chip
  if (filters.flagged && filters.flagged !== "all") {
    chips.push({
      key: "flagged",
      label: FLAGGED_LABELS[filters.flagged] ?? filters.flagged,
      onRemove: () => onChange({ ...filters, flagged: undefined }),
    });
  }

  // Empty chip. The page default renders no chip — a default is not a
  // filter. Only a deviation shows, and removing it clears the filter,
  // which the page resolves back to its default. A view the filter is
  // inert on (`showEmpty` false) shows no chip either, so a chip never
  // claims a filtering that is not happening.
  if (showEmpty && filters.empty && filters.empty !== emptyDefault) {
    chips.push({
      key: "empty",
      label: EMPTY_LABELS[filters.empty] ?? filters.empty,
      onRemove: () => onChange({ ...filters, empty: undefined }),
    });
  }

  // Detection-confidence range chip
  const detMin = filters.min_confidence;
  const detMax = filters.max_confidence;
  if (detMin !== undefined || detMax !== undefined) {
    const lo = Math.round((detMin ?? detectionFloor) * 100);
    const hi = Math.round((detMax ?? 1) * 100);
    // Say what is actually filtered: a floor alone reads "≥", not a
    // range whose top is a 100% nobody set (the Files tab only has the
    // floor, and on Detections an untouched high handle is no ceiling).
    const label =
      detMax === undefined
        ? `Det: ≥ ${lo}%`
        : detMin === undefined
          ? `Det: ≤ ${hi}%`
          : `Det: ${lo} – ${hi}%`;
    chips.push({
      key: "det-confidence",
      label,
      onRemove: () =>
        onChange({
          ...filters,
          min_confidence: undefined,
          max_confidence: undefined,
        }),
    });
  }

  // Classification-confidence range chip
  const clsMin = filters.min_label_confidence;
  const clsMax = filters.max_label_confidence;
  if (clsMin !== undefined || clsMax !== undefined) {
    const lo = Math.round((clsMin ?? 0) * 100);
    const hi = Math.round((clsMax ?? 1) * 100);
    const label =
      clsMax === undefined
        ? `Cls: ≥ ${lo}%`
        : clsMin === undefined
          ? `Cls: ≤ ${hi}%`
          : `Cls: ${lo} – ${hi}%`;
    chips.push({
      key: "cls-confidence",
      label,
      onRemove: () =>
        onChange({
          ...filters,
          min_label_confidence: undefined,
          max_label_confidence: undefined,
        }),
    });
  }

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
            onClick={chip.onRemove}
            className="ml-0.5 rounded-full hover:bg-black/10 p-0.5"
          >
            <X className="h-3 w-3" />
          </button>
        </Badge>
      ))}
      <button
        onClick={() => onChange({})}
        className="text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        Clear all
      </button>
    </div>
  );
}
