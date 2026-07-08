/**
 * Active filter chips displayed below the filter panel.
 *
 * Shows a summary of active filters with dismiss buttons and result count.
 */

import { X } from "lucide-react";
import type { EventFilterParams } from "../../api/types";
import { Badge } from "../ui/badge";

/** True when any user filter is set on these params. Shared by every
 *  verify tab so the chip row and "no results" copy stay in sync with
 *  the chips this component actually renders. */
export function hasAnyActiveFilter(filters: EventFilterParams): boolean {
  return (
    (filters.site_ids?.length ?? 0) > 0 ||
    !!filters.date_from ||
    !!filters.date_to ||
    (filters.labels?.length ?? 0) > 0 ||
    (!!filters.verification && filters.verification !== "all") ||
    (!!filters.flagged && filters.flagged !== "all") ||
    (!!filters.favorited && filters.favorited !== "all") ||
    (!!filters.empty && filters.empty !== "hide") ||
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
  /** Project detection_threshold — used to detect when det range is "default". */
  detectionFloor?: number;
  /** The page's default verification value: no chip when the filter
   *  equals it (a default is not a filter). Counts defaults to "all",
   *  the Labels page to "unverified". */
  verificationDefault?: string;
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
};


export function FilterChips({
  filters,
  onChange,
  siteNames,
  displayLabels,
  detectionFloor = 0,
  verificationDefault = "all",
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

  // Empty chip. "hide" is the implicit default (hide blank captures),
  // so it renders no chip — a default is not a filter. Only a
  // deviation ("Including empty" / "Empty only") shows, and removing it
  // resets to the "hide" default (set explicitly because undefined also
  // falls back to "hide").
  if (filters.empty && filters.empty !== "hide") {
    chips.push({
      key: "empty",
      label: EMPTY_LABELS[filters.empty] ?? filters.empty,
      onRemove: () => onChange({ ...filters, empty: "hide" }),
    });
  }

  // Detection-confidence range chip
  const detMin = filters.min_confidence;
  const detMax = filters.max_confidence;
  if (detMin !== undefined || detMax !== undefined) {
    const lo = Math.round((detMin ?? detectionFloor) * 100);
    const hi = Math.round((detMax ?? 1) * 100);
    chips.push({
      key: "det-confidence",
      label: `Det: ${lo} – ${hi}%`,
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
    chips.push({
      key: "cls-confidence",
      label: `Cls: ${lo} – ${hi}%`,
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
