/**
 * Output filter card for the Save outputs step.
 *
 * One global label exclusion list that applies to every output
 * module: Separate / Visualise / Blur / EXIF (file-level skip when
 * all labels are excluded) and CSV / XLSX / recognition JSON
 * (row-level filter on the excluded labels).
 *
 * The tree itself is the same ``LabelFilterModal`` the verify page
 * uses — single source of truth, project-scoped to the labels that
 * actually have detections in this run. We use **inclusion**
 * semantics in the UI (pick what to keep) and convert to an
 * exclusion list at the API boundary, matching the verify
 * convention.
 *
 * The card hides itself when the label tree is empty (no labelled
 * detections in the run).
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Filter } from "lucide-react";

import { Button } from "../../../components/ui/button";
import { Card, CardContent } from "../../../components/ui/card";
import { LabelFilterModal } from "../../../components/verify/LabelFilterModal";
import { eventsApi } from "../../../api/events";
import type { UseSaveOutputsFormResult } from "./useSaveOutputsForm";

export function OutputFilterCard({
  form,
  projectId,
}: {
  form: UseSaveOutputsFormResult;
  projectId: string;
}) {
  const [modalOpen, setModalOpen] = useState(false);

  // Project-scoped label tree — same shape and source the verify
  // page uses. Only labels that actually have detections in this
  // run land here.
  const { data: labelTree, isLoading } = useQuery({
    queryKey: ["label-tree", projectId],
    queryFn: () => eventsApi.getLabelTree(projectId),
    staleTime: 30_000,
  });

  // Hide card while loading or when the run has no labels.
  if (isLoading) return null;
  if (!labelTree || labelTree.tree.length === 0) return null;

  const allLeafIds = labelTree.all_leaf_ids;
  const totalLabels = allLeafIds.length;
  const excluded = form.excludedLabelIds;
  // Translate exclusion → inclusion for the verify-style modal.
  const includedIds = allLeafIds.filter((id) => !excluded.includes(id));
  const includedCount = includedIds.length;

  const noFilter = excluded.length === 0;
  const summary = noFilter
    ? `All ${totalLabels} labels included.`
    : `${includedCount} of ${totalLabels} labels included.`;

  return (
    <Card>
      <CardContent className="flex items-center gap-3 p-4 text-sm">
        <Filter className="h-5 w-5 shrink-0 text-primary" />
        <div className="flex-1">
          <p className="font-medium">Filter outputs by label</p>
          <p className="text-xs text-muted-foreground">{summary}</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setModalOpen(true)}
          className="shrink-0"
        >
          {noFilter ? "Pick labels..." : "Edit..."}
        </Button>
      </CardContent>

      <LabelFilterModal
        preBuiltTree={labelTree.tree}
        allLeafIds={allLeafIds}
        selectedLabels={noFilter ? [] : includedIds}
        onApply={(labels) => {
          // labels = the included leaf IDs. Convert to the exclusion
          // list our backend expects: every leaf not in `labels`.
          // If everything is selected (or nothing is — both mean
          // "no filter"), clear the exclusion set.
          if (labels.length === 0 || labels.length >= allLeafIds.length) {
            form.setExcludedLabelIds([]);
          } else {
            const includedSet = new Set(labels);
            form.setExcludedLabelIds(
              allLeafIds.filter((id) => !includedSet.has(id)),
            );
          }
        }}
        open={modalOpen}
        onOpenChange={setModalOpen}
        countUnit={labelTree.count_unit}
      />
    </Card>
  );
}
