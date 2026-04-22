/**
 * Shared taxonomic-rank dropdown used by the dashboard and every
 * insights plot. Thin wrapper around the existing Radix Select; options
 * and default come from lib/taxonomic-rank.ts so the dashboard and the
 * matrix / report can't drift.
 */

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./select";
import {
  RANK_OPTIONS,
  type TaxonomicRank,
} from "../../lib/taxonomic-rank";

interface TaxonomicRankPickerProps {
  value: TaxonomicRank;
  onChange: (value: TaxonomicRank) => void;
  disabled?: boolean;
  className?: string;
}

export function TaxonomicRankPicker({
  value,
  onChange,
  disabled,
  className,
}: TaxonomicRankPickerProps) {
  return (
    <Select
      value={value}
      onValueChange={(v) => onChange(v as TaxonomicRank)}
      disabled={disabled}
    >
      <SelectTrigger
        className={className ?? "w-full h-9 min-h-9 py-1 text-sm"}
      >
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {RANK_OPTIONS.map((r) => (
          <SelectItem key={r.value} value={r.value}>
            {r.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
