/**
 * "Choose folder" dialog for a single group of missing deployments.
 *
 * Opened from a RelinkGroupBanner when the user wants to pick a new
 * folder manually (either because auto-suggest couldn't find one, or
 * they rejected the suggestion). Shows the affected deployments as
 * a friendly list (site name + start date) with per-row checkboxes
 * to exclude individual deployments, then runs bulk-relink on save.
 */

import { useEffect, useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, XCircle } from "lucide-react";
import { toast } from "sonner";
import { deploymentsApi } from "../../api/deployments";
import type { BulkRelinkResponse, DeploymentResponse } from "../../api/types";
import {
  leafName,
  replacePrefix,
  type PathItem,
  type PrefixGroup,
} from "../../lib/path-utils";
import { Button } from "../ui/button";
import { Checkbox } from "../ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { Label } from "../ui/label";
import { ScrollArea } from "../ui/scroll-area";
import { FolderSelector } from "../analyses/FolderSelector";

export type DeploymentPathItem = PathItem;

interface BulkRelinkDeploymentDialogProps {
  projectId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The group of broken deployments this dialog acts on. */
  group: PrefixGroup<DeploymentPathItem> | null;
  /** Optional pre-filled "new folder" — usually the backend's auto-suggestion. */
  initialNewParentPath?: string | null;
  /** Full deployment records keyed by id, for displaying site/date in the preview. */
  deploymentsById: Map<string, DeploymentResponse>;
  /** Site name lookup for display in the preview list. */
  siteNames: Record<string, string>;
}

export function BulkRelinkDeploymentDialog({
  projectId,
  open,
  onOpenChange,
  group,
  initialNewParentPath,
  deploymentsById,
  siteNames,
}: BulkRelinkDeploymentDialogProps) {
  const queryClient = useQueryClient();

  const [newParentPath, setNewParentPath] = useState<string | null>(null);
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const [result, setResult] = useState<BulkRelinkResponse | null>(null);

  // Reset state each time the dialog opens.
  useEffect(() => {
    if (open) {
      setNewParentPath(initialNewParentPath ?? null);
      setExcluded(new Set());
      setResult(null);
    }
  }, [open, initialNewParentPath]);

  const missingLeaf = group ? leafName(group.prefix) : "";

  // Build the replacements for this group, applying exclusions.
  const replacements = useMemo(() => {
    if (!group || !newParentPath || !newParentPath.trim()) return [];
    return group.items
      .filter((item) => !excluded.has(item.id))
      .map((item) => ({
        deployment_id: item.id,
        new_folder_path: replacePrefix(
          item.folder_path,
          group.prefix,
          newParentPath
        ),
      }));
  }, [group, newParentPath, excluded]);

  const bulkRelinkMutation = useMutation({
    mutationFn: () => deploymentsApi.bulkRelink({ replacements }),
    onSuccess: (response) => {
      setResult(response);
      queryClient.invalidateQueries({ queryKey: ["deployments", projectId] });
      queryClient.invalidateQueries({ queryKey: ["deployment-stats", projectId] });

      const successes = response.results.filter((r) => r.success).length;
      const failures = response.results.length - successes;
      if (failures === 0) {
        toast.success(
          `Reconnected ${successes} deployment${successes === 1 ? "" : "s"}`
        );
      } else if (successes === 0) {
        toast.error(
          "Those files don't match. Try a different folder."
        );
      } else {
        toast.warning(
          `Reconnected ${successes} of ${response.results.length}. Some didn't match.`
        );
      }
    },
  });

  const toggleExcluded = (id: string) => {
    setExcluded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const canApply =
    !!group &&
    !!newParentPath &&
    newParentPath.trim() !== "" &&
    newParentPath !== group.prefix.replace(/\/$/, "") &&
    replacements.length > 0 &&
    !bulkRelinkMutation.isPending;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            {missingLeaf ? `Find folder for ${missingLeaf}` : "Find folder"}
          </DialogTitle>
          <DialogDescription>
            Tell us where these files are now. AddaxAI will check a few
            sample files to make sure it's the right folder before updating
            anything.
          </DialogDescription>
        </DialogHeader>

        {result ? (
          <ResultPanel
            results={result.results}
            deploymentsById={deploymentsById}
            siteNames={siteNames}
          />
        ) : group ? (
          <div className="space-y-4">
            {/* Was / Is-now path stack */}
            <div className="space-y-2">
              <Label>Was</Label>
              <div className="rounded-md border bg-muted/40 px-3 py-2 text-sm text-muted-foreground font-mono break-words">
                {group.prefix}
              </div>
            </div>

            <div className="space-y-2">
              <Label>Is now</Label>
              <FolderSelector
                value={newParentPath}
                onChange={setNewParentPath}
                hideLabel
                hideScanResult
              />
            </div>

            {/* Affected-deployments list: name + date front and center */}
            {newParentPath && (
              <div className="space-y-2">
                <Label>
                  {replacements.length} of {group.items.length} deployment
                  {group.items.length === 1 ? "" : "s"} will be updated
                </Label>
                <ScrollArea className="h-60 rounded-md border">
                  <div className="divide-y">
                    {group.items.map((item) => {
                      const dep = deploymentsById.get(item.id);
                      const isExcluded = excluded.has(item.id);
                      const siteName = dep
                        ? siteNames[dep.site_id] ?? "Unknown site"
                        : "Unknown";
                      const relativeOld = item.folder_path.slice(
                        group.prefix.length
                      );
                      return (
                        <div
                          key={item.id}
                          className="flex items-start gap-3 px-3 py-2"
                        >
                          <Checkbox
                            checked={!isExcluded}
                            onCheckedChange={() => toggleExcluded(item.id)}
                            className="mt-1 shrink-0"
                          />
                          <div className="flex-1 min-w-0 space-y-0.5">
                            <div className="font-medium text-sm">
                              {siteName}
                              {dep?.start_date && (
                                <span className="text-muted-foreground font-normal">
                                  {" · "}
                                  {dep.start_date}
                                </span>
                              )}
                            </div>
                            {relativeOld && (
                              <div className="font-mono text-xs text-muted-foreground break-all">
                                …/{relativeOld.replace(/^\//, "")}
                              </div>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </ScrollArea>
              </div>
            )}

            {bulkRelinkMutation.isError && (
              <p className="text-sm font-medium text-destructive">
                Something went wrong:{" "}
                {bulkRelinkMutation.error instanceof Error
                  ? bulkRelinkMutation.error.message
                  : String(bulkRelinkMutation.error)}
              </p>
            )}
          </div>
        ) : null}

        <DialogFooter>
          {result ? (
            <Button type="button" onClick={() => onOpenChange(false)}>
              Done
            </Button>
          ) : (
            <>
              <Button
                type="button"
                variant="outline"
                onClick={() => onOpenChange(false)}
              >
                Cancel
              </Button>
              <Button
                type="button"
                onClick={() => bulkRelinkMutation.mutate()}
                disabled={!canApply}
              >
                {bulkRelinkMutation.isPending
                  ? "Checking files..."
                  : `Find ${replacements.length} folder${replacements.length === 1 ? "" : "s"}`}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

interface ResultPanelProps {
  results: BulkRelinkResponse["results"];
  deploymentsById: Map<string, DeploymentResponse>;
  siteNames: Record<string, string>;
}

function ResultPanel({ results, deploymentsById, siteNames }: ResultPanelProps) {
  const successes = results.filter((r) => r.success);
  const failures = results.filter((r) => !r.success);

  return (
    <div className="space-y-4">
      {successes.length > 0 && (
        <div
          className="rounded-lg border p-3"
          style={{
            color: "#0f6064",
            backgroundColor: "#0f60641a",
            borderColor: "#0f606433",
          }}
        >
          <div className="flex items-center gap-2 text-sm font-medium">
            <CheckCircle2 className="h-4 w-4" />
            Reconnected {successes.length} deployment
            {successes.length === 1 ? "" : "s"}
          </div>
        </div>
      )}

      {failures.length > 0 && (
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-3 space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium text-destructive">
            <XCircle className="h-4 w-4" />
            {failures.length} deployment
            {failures.length === 1 ? "" : "s"} didn't match
          </div>
          <ScrollArea className="max-h-48">
            <div className="space-y-2 text-xs text-destructive/90">
              {failures.map((r) => {
                const dep = deploymentsById.get(r.deployment_id);
                const siteName = dep
                  ? siteNames[dep.site_id] ?? "Unknown site"
                  : "Unknown";
                return (
                  <div key={r.deployment_id} className="space-y-0.5">
                    <div className="font-medium">
                      {siteName}
                      {dep?.start_date && ` · ${dep.start_date}`}
                    </div>
                    {r.mismatches.length > 0 && (
                      <div className="font-mono text-[10px] opacity-80">
                        {r.mismatches.length} sample file
                        {r.mismatches.length === 1 ? "" : "s"} didn't match
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </ScrollArea>
          <p className="text-xs text-destructive/90">
            Try choosing a different folder.
          </p>
        </div>
      )}

    </div>
  );
}
