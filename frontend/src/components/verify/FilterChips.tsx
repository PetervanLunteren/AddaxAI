/**
 * Active filter chips displayed below the filter panel.
 *
 * Shows a summary of active filters with dismiss buttons and result count.
 */

import { X } from "lucide-react";
import type { EventFilterParams } from "../../api/types";
import { Badge } from "../ui/badge";

interface FilterChipsProps {
  filters: EventFilterParams;
  onChange: (filters: EventFilterParams) => void;
  filteredCount: number;
  totalCount: number;
  siteNames: Record<string, string>;
}

const VERIFICATION_LABELS: Record<string, string> = {
  none_verified: "None verified",
  not_fully_verified: "Partially verified",
  unverified_representative: "Representative not verified",
  verified_representative: "Representative verified",
  fully_verified: "Fully verified",
};


export function FilterChips({
  filters,
  onChange,
  filteredCount,
  totalCount,
  siteNames,
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

  // Species chips
  if (filters.species?.length) {
    if (filters.species.length <= 2) {
      for (const sp of filters.species) {
        chips.push({
          key: `species-${sp}`,
          label: sp,
          onRemove: () => {
            const next = filters.species!.filter((s) => s !== sp);
            onChange({ ...filters, species: next.length ? next : undefined });
          },
        });
      }
    } else {
      chips.push({
        key: "species",
        label: `${filters.species.length} species`,
        onRemove: () => onChange({ ...filters, species: undefined }),
      });
    }
  }

  // Verification chip
  if (filters.verification && filters.verification !== "all") {
    chips.push({
      key: "verification",
      label: VERIFICATION_LABELS[filters.verification] ?? filters.verification,
      onRemove: () => onChange({ ...filters, verification: undefined }),
    });
  }

  // Confidence chip
  if (filters.min_confidence !== undefined || filters.max_confidence !== undefined) {
    const lo = ((filters.min_confidence ?? 0) * 100).toFixed(0);
    const hi = ((filters.max_confidence ?? 1) * 100).toFixed(0);
    chips.push({
      key: "confidence",
      label: `Confidence ${lo}–${hi}%`,
      onRemove: () => onChange({ ...filters, min_confidence: undefined, max_confidence: undefined }),
    });
  }

  if (chips.length === 0) return null;

  const isFiltered = filteredCount !== totalCount;

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-sm text-muted-foreground shrink-0">
        {isFiltered
          ? `${filteredCount} of ${totalCount} events`
          : `${totalCount} events`}
      </span>
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
