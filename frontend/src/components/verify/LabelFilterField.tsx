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

import { eventsApi, type LabelTreeScope } from "../../api/events";
import { collectLiveLeafIds } from "../../lib/taxonomy-utils";
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
  /** The page's other active filters: the count beside a label is what
   *  the page shows with that label ticked alone. */
  scope?: LabelTreeScope;
}

export function LabelFilterField({
  projectId,
  value,
  onChange,
  countBy = "event",
  scope,
}: LabelFilterFieldProps) {
  const [open, setOpen] = useState(false);

  const { data: labelTree } = useQuery({
    queryKey: ["label-tree", projectId, countBy, scope],
    queryFn: () => eventsApi.getLabelTree(projectId, countBy, scope),
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

  // Every live label ticked is the same as no filter, and keeps the URL
  // clean. Dead leaves (nothing under the page's filters) cannot be
  // ticked, so they do not count towards "everything".
  const apply = (labels: string[]) => {
    const live = labelTree ? collectLiveLeafIds(labelTree.tree) : null;
    const everything =
      live !== null && [...live].every((id) => labels.includes(id));
    onChange(labels.length === 0 || everything ? undefined : labels);
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
