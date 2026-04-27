/**
 * Generic schema-driven filter bar.
 *
 * Mirrors the visual style of the Verify page FilterPanel + FilterChips:
 * - White rounded card with border
 * - Search field on the left (label-less, fixed width)
 * - Other filters in a responsive grid to the right
 * - Active-filter chips below with X-to-remove and Clear all link
 *
 * Pages declare their filter shape via a `fields` array; this component
 * does NOT fetch any data — option lists are passed in via props so each
 * page controls its own data sources.
 */

import { Search, X } from "lucide-react";
import { Badge } from "./badge";
import { DateRangePicker } from "./date-range-picker";
import { Input } from "./input";
import { MultiSelect, type MultiSelectOption } from "./multi-select";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./select";

// Filter values are loose key/string maps. Multi-selects are string[].
export type FilterValues = Record<string, string | string[] | undefined>;

type FieldBase = {
  /** Key into the filter values object. */
  key: string;
  /** Display label (used both above the control and in chips). */
  label: string;
};

export type FilterFieldDef =
  | (FieldBase & {
      kind: "search";
      placeholder?: string;
    })
  | (FieldBase & {
      kind: "multi-select";
      options: MultiSelectOption[];
      placeholder?: string;
      summary?: (count: number) => string;
    })
  | (FieldBase & {
      kind: "date";
      min?: string;
      max?: string;
    })
  | (FieldBase & {
      // Two-key date range. The single `key` on the base is the FROM key;
      // toKey is the TO key. Both store ISO date strings (YYYY-MM-DD).
      kind: "date_range";
      toKey: string;
      min?: string;
      max?: string;
    })
  | (FieldBase & {
      kind: "select";
      options: { value: string; label: string }[];
      placeholder?: string;
    });

interface FilterBarProps {
  /** Current filter values, controlled by parent. */
  value: FilterValues;
  /** Called when any filter changes. Parent persists / drives queries. */
  onChange: (next: FilterValues) => void;
  /** Filter definitions. Search field (if any) renders separately first. */
  fields: FilterFieldDef[];
  /** Optional row count summary appended to the chips row. */
  filteredCount?: number;
  /** Singular/plural noun for the count display. */
  countLabel?: { singular: string; plural: string };
}

const DATE_INPUT_CLASSES =
  "flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring [&::-webkit-date-and-time-value]:text-left [&::-webkit-calendar-picker-indicator]:absolute [&::-webkit-calendar-picker-indicator]:right-3 relative";

/** Set or remove a key on the filter object based on whether the value is "empty". */
function setField(
  value: FilterValues,
  key: string,
  next: string | string[] | undefined
): FilterValues {
  const cleared =
    next === undefined ||
    next === "" ||
    (Array.isArray(next) && next.length === 0);
  const out = { ...value };
  if (cleared) {
    delete out[key];
  } else {
    out[key] = next;
  }
  return out;
}

/** Whether the field has a non-empty value in the current filter object. */
function hasValue(value: FilterValues, key: string): boolean {
  const v = value[key];
  if (v === undefined || v === "") return false;
  if (Array.isArray(v) && v.length === 0) return false;
  return true;
}

/** Whether a field has any value set, accounting for date_range's two keys. */
function fieldHasValue(value: FilterValues, field: FilterFieldDef): boolean {
  if (field.kind === "date_range") {
    return hasValue(value, field.key) || hasValue(value, field.toKey);
  }
  return hasValue(value, field.key);
}

/** Total number of active filter fields. */
function activeFieldCount(value: FilterValues, fields: FilterFieldDef[]): number {
  return fields.filter((f) => fieldHasValue(value, f)).length;
}

export function FilterBar({
  value,
  onChange,
  fields,
  filteredCount,
  countLabel,
}: FilterBarProps) {
  const hasAnyActive = activeFieldCount(value, fields) > 0;

  // Static class map so Tailwind's purger picks up every variant.
  // The grid auto-matches the field count at lg+, so each filter row
  // always fits in one row regardless of how many fields a page has.
  const lgColsClass: Record<number, string> = {
    1: "lg:grid-cols-1",
    2: "lg:grid-cols-2",
    3: "lg:grid-cols-3",
    4: "lg:grid-cols-4",
    5: "lg:grid-cols-5",
    6: "lg:grid-cols-6",
  };
  const gridCols = lgColsClass[fields.length] ?? "lg:grid-cols-6";

  // Helpers for chip option-label lookups (for multi-select chips)
  const optionLabel = (field: FilterFieldDef, val: string): string => {
    if (field.kind === "multi-select" || field.kind === "select") {
      return field.options.find((o) => o.value === val)?.label ?? val;
    }
    return val;
  };

  // Build the chips array for the bottom row
  const chips: { key: string; label: string; onRemove: () => void }[] = [];
  for (const field of fields) {
    if (!fieldHasValue(value, field)) continue;
    const v = value[field.key];

    if (field.kind === "search") {
      chips.push({
        key: `${field.key}`,
        label: `${field.label}: ${v}`,
        onRemove: () => onChange(setField(value, field.key, undefined)),
      });
      continue;
    }

    if (field.kind === "date") {
      chips.push({
        key: `${field.key}`,
        label: `${field.label}: ${v}`,
        onRemove: () => onChange(setField(value, field.key, undefined)),
      });
      continue;
    }

    if (field.kind === "date_range") {
      const from = value[field.key] as string | undefined;
      const to = value[field.toKey] as string | undefined;
      if (from || to) {
        const summary = from && to ? `${from} – ${to}` : from || to;
        chips.push({
          key: `${field.key}`,
          label: `${field.label}: ${summary}`,
          onRemove: () => {
            const next = setField(value, field.key, undefined);
            onChange(setField(next, field.toKey, undefined));
          },
        });
      }
      continue;
    }

    if (field.kind === "select") {
      chips.push({
        key: `${field.key}`,
        label: `${field.label}: ${optionLabel(field, v as string)}`,
        onRemove: () => onChange(setField(value, field.key, undefined)),
      });
      continue;
    }

    if (field.kind === "multi-select") {
      const arr = (v as string[]) ?? [];
      if (arr.length <= 2) {
        for (const item of arr) {
          chips.push({
            key: `${field.key}-${item}`,
            label: `${field.label}: ${optionLabel(field, item)}`,
            onRemove: () =>
              onChange(setField(value, field.key, arr.filter((x) => x !== item))),
          });
        }
      } else {
        chips.push({
          key: `${field.key}`,
          label: `${arr.length} ${field.label.toLowerCase()}`,
          onRemove: () => onChange(setField(value, field.key, undefined)),
        });
      }
    }
  }

  return (
    <div className="rounded-lg border bg-white pt-2 pb-3 px-3 space-y-3">
      <div className={`grid grid-cols-1 sm:grid-cols-2 ${gridCols} gap-4`}>
        {fields.map((field) => (
          <div key={field.key} className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">
              {field.label}
            </label>
            {field.kind === "search" && (
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder={field.placeholder ?? "Search..."}
                  value={(value[field.key] as string | undefined) ?? ""}
                  onChange={(e) =>
                    onChange(setField(value, field.key, e.target.value))
                  }
                  className="pl-9 h-9"
                />
              </div>
            )}
            {field.kind === "multi-select" && (
              <MultiSelect
                options={field.options}
                value={(value[field.key] as string[] | undefined) ?? []}
                onChange={(v) =>
                  onChange(setField(value, field.key, v))
                }
                placeholder={field.placeholder ?? `All ${field.label.toLowerCase()}`}
                searchPlaceholder={`Search ${field.label.toLowerCase()}...`}
                summary={field.summary}
              />
            )}
            {field.kind === "date" && (
              <input
                type="date"
                className={DATE_INPUT_CLASSES}
                value={(value[field.key] as string | undefined) ?? ""}
                min={field.min}
                max={field.max}
                onChange={(e) =>
                  onChange(setField(value, field.key, e.target.value))
                }
              />
            )}
            {field.kind === "date_range" && (
              <DateRangePicker
                from={(value[field.key] as string | undefined) ?? null}
                to={(value[field.toKey] as string | undefined) ?? null}
                onChange={({ from, to }) => {
                  const next = setField(value, field.key, from);
                  onChange(setField(next, field.toKey, to));
                }}
                minDate={field.min}
                maxDate={field.max}
              />
            )}
            {field.kind === "select" && (
              <Select
                value={(value[field.key] as string | undefined) ?? ""}
                onValueChange={(v) =>
                  onChange(setField(value, field.key, v))
                }
              >
                <SelectTrigger className="h-9 min-h-0 text-sm">
                  <SelectValue placeholder={field.placeholder ?? "Select..."} />
                </SelectTrigger>
                <SelectContent>
                  {field.options.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>
        ))}
      </div>

      {/* Active-filter chips + clear all + count */}
      {(hasAnyActive || filteredCount !== undefined) && (
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
                type="button"
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
          {hasAnyActive && (
            <button
              onClick={() => onChange({})}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors"
              type="button"
            >
              Clear all
            </button>
          )}
          {filteredCount !== undefined && countLabel && (
            <span className="text-sm text-muted-foreground ml-auto">
              {filteredCount} {filteredCount === 1 ? countLabel.singular : countLabel.plural}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
