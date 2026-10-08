/**
 * The label filter control, one for every page that filters by label:
 * Labels and Counts, the Map, and the dashboard's Explore tab.
 *
 * It owns everything those pages used to repeat around the shared tree
 * modal: the label-tree query (same key everywhere, so they share a cache),
 * the trigger button, the rule that everything ticked means no filter,
 * and the flat list for projects without a taxonomy tree. The page only
 * holds the value.
 */

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { eventsApi } from "../../api/events";
import { speciesLabelMap } from "../../lib/species-name-mode";
import { Button } from "../ui/button";
import { MultiSelect } from "../ui/multi-select";
import { LabelFilterModal } from "./LabelFilterModal";

interface LabelFilterFieldProps {
  projectId: string;
  /** Picked labels; undefined means no filter (every label). */
  value: string[] | undefined;
  onChange: (labels: string[] | undefined) => void;
  /** Unit the tree's counts are in; the Labels page varies it per tab. */
  countBy?: string;
  /** Narrow the tree's counts to what the page is looking at. */
  siteIds?: string[];
  dateFrom?: string;
  dateTo?: string;
}

export function LabelFilterField({
  projectId,
  value,
  onChange,
  countBy = "event",
  siteIds,
  dateFrom,
  dateTo,
}: LabelFilterFieldProps) {
  const [open, setOpen] = useState(false);

  const { data: labelTree } = useQuery({
    queryKey: ["label-tree", projectId, countBy, siteIds, dateFrom, dateTo],
    queryFn: () =>
      eventsApi.getLabelTree(projectId, countBy, { siteIds, dateFrom, dateTo }),
    enabled: !!projectId,
  });
  const hasTree = !!labelTree?.tree?.length;
  // Display names in the user's common/scientific setting, the same map
  // the filter chips use, so the button and the chip beside it agree.
  // Also the options of the flat list for projects without a tree.
  const { data: filterOptions } = useQuery({
    queryKey: ["event-filter-options", projectId],
    queryFn: () => eventsApi.getFilterOptions(projectId),
    enabled: !!projectId,
  });
  const names = useMemo(
    () => (filterOptions ? speciesLabelMap(filterOptions) : {}),
    [filterOptions],
  );

  // Everything ticked is the same as no filter, and keeps the URL clean.
  const apply = (labels: string[]) => {
    const all = labelTree?.all_leaf_ids.length ?? Infinity;
    onChange(labels.length === 0 || labels.length >= all ? undefined : labels);
  };

  if (labelTree !== undefined && !hasTree) {
    return (
      <MultiSelect
        options={(filterOptions?.labels ?? []).map((lbl) => ({
          value: lbl,
          label: names[lbl] ?? lbl,
        }))}
        value={value ?? []}
        onChange={(v) => onChange(v.length ? v : undefined)}
        placeholder="All labels"
        searchPlaceholder="Search labels..."
        emptyMessage="No labels found."
        summary={(n) => `${n} label${n > 1 ? "s" : ""}`}
        capitalize
      />
    );
  }

  const picked = value ?? [];
  const text =
    picked.length === 0
      ? "All labels"
      : picked.length === 1
        ? (names[picked[0]] ?? "1 label")
        : `${picked.length} labels`;

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        className="h-9 w-full justify-start text-sm font-normal"
        disabled={!hasTree}
        onClick={() => setOpen(true)}
      >
        <span className="truncate">{text}</span>
      </Button>
      {hasTree && (
        <LabelFilterModal
          preBuiltTree={labelTree!.tree}
          allLeafIds={labelTree!.all_leaf_ids}
          selectedLabels={picked}
          onApply={apply}
          open={open}
          onOpenChange={setOpen}
          countUnit={labelTree!.count_unit}
        />
      )}
    </>
  );
}
