/**
 * Per-group recovery banner for missing deployment folders.
 *
 * Shown one-per-group at the top of the deployments page when one or
 * more deployments can't find their files. The grouping and the
 * auto-suggested replacement come pre-computed from the backend's
 * /api/deployments/group-broken endpoint — the banner just renders
 * what it's given and runs the bulk-relink mutation.
 *
 *   - Confirmation card if `group.suggested_path` is set: "Looks like
 *     you moved X to Y. Yes / No, I know where it is".
 *   - Manual prompt otherwise: a Choose-folder button that opens
 *     BulkRelinkDeploymentDialog in manual-picker mode.
 *
 * Multiple banners stack naturally: one drive rename = one banner,
 * ten renamed folders = ten banners.
 */

import { useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Folder, FolderSearch } from "lucide-react";
import { toast } from "sonner";
import { deploymentsApi } from "../../api/deployments";
import type {
  BulkRelinkResponse,
  DeploymentResponse,
  GroupBrokenGroup,
} from "../../api/types";
import {
  diffPaths,
  leafName,
  replacePrefix,
} from "../../lib/path-utils";
import type { PrefixGroup } from "../../lib/path-utils";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
import {
  BulkRelinkDeploymentDialog,
  type DeploymentPathItem,
} from "./BulkRelinkDeploymentDialog";

interface RelinkGroupBannerProps {
  group: GroupBrokenGroup;
  projectId: string;
  siteNames: Record<string, string>;
  deploymentsById: Map<string, DeploymentResponse>;
}

export function RelinkGroupBanner({
  group,
  projectId,
  siteNames,
  deploymentsById,
}: RelinkGroupBannerProps) {
  const queryClient = useQueryClient();
  const [dialogOpen, setDialogOpen] = useState(false);
  // The refused attempt from the one-click confirm, handed to the dialog so
  // its per-file reasons survive. Cleared whenever the user opens the dialog
  // themselves, so a stale refusal never greets a fresh attempt.
  const [failedAttempt, setFailedAttempt] = useState<BulkRelinkResponse | null>(
    null
  );

  const openDialogFresh = () => {
    setFailedAttempt(null);
    setDialogOpen(true);
  };

  const missingLeaf = leafName(group.prefix);
  const missingPath = group.prefix.replace(/\/+$/, "");
  const suggestedPath = group.suggested_path;
  const count = group.items.length;

  // Synthesize the legacy PrefixGroup shape the dialog expects.
  // Trailing slash matters so replacePrefix can substitute cleanly.
  const dialogGroup = useMemo<PrefixGroup<DeploymentPathItem>>(
    () => ({
      prefix: missingPath + "/",
      items: group.items,
    }),
    [missingPath, group.items]
  );

  const relinkMutation = useMutation({
    mutationFn: (newParentPath: string) => {
      const replacements = group.items.map((item) => ({
        deployment_id: item.id,
        new_folder_path: replacePrefix(
          item.folder_path,
          dialogGroup.prefix,
          newParentPath
        ),
      }));
      return deploymentsApi.bulkRelink({ replacements });
    },
    onSuccess: (response) => {
      queryClient.invalidateQueries({ queryKey: ["deployments", projectId] });
      queryClient.invalidateQueries({
        queryKey: ["deployment-stats", projectId],
      });
      const successes = response.results.filter((r) => r.success).length;
      const failures = response.results.length - successes;
      if (failures === 0) {
        toast.success(
          `Reconnected ${successes} deployment${successes === 1 ? "" : "s"}`
        );
      } else if (successes === 0) {
        // Hand the refusal to the dialog rather than dropping it. The
        // response already names every file and says whether it was
        // missing or a different size, which is the only way to tell a
        // wrong folder from a folder whose contents changed. Without
        // this the user got one generic sentence and the same banner
        // back, with no way to work out what to do next.
        setFailedAttempt(response);
        toast.error("Those files don't match. See what didn't match.");
        setDialogOpen(true);
      } else {
        toast.warning(
          `Reconnected ${successes} of ${response.results.length}. Some didn't match.`
        );
      }
    },
  });

  const titleLeaf = missingLeaf ? (
    <>
      {" in "}
      <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-[0.9em] font-normal">
        {missingLeaf}
      </code>
    </>
  ) : null;

  return (
    <>
      <Callout variant="error" className="mb-3">
        <div className="space-y-3">
          <div className="text-sm font-medium">
            Can't find files for {count} deployment{count === 1 ? "" : "s"}
            {titleLeaf}
          </div>

          {suggestedPath ? (
            <ConfirmationCard
              missingPath={missingPath}
              suggestedPath={suggestedPath}
              busy={relinkMutation.isPending}
              onConfirm={() => relinkMutation.mutate(suggestedPath)}
              onReject={openDialogFresh}
            />
          ) : (
            <ManualPrompt
              missingPath={missingPath}
              onChoose={openDialogFresh}
            />
          )}
        </div>
      </Callout>

      <BulkRelinkDeploymentDialog
        projectId={projectId}
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        group={dialogGroup}
        initialNewParentPath={suggestedPath}
        initialResult={failedAttempt}
        deploymentsById={deploymentsById}
        siteNames={siteNames}
      />
    </>
  );
}

interface ConfirmationCardProps {
  missingPath: string;
  suggestedPath: string;
  busy: boolean;
  onConfirm: () => void;
  onReject: () => void;
}

function ConfirmationCard({
  missingPath,
  suggestedPath,
  busy,
  onConfirm,
  onReject,
}: ConfirmationCardProps) {
  const diff = diffPaths(missingPath, suggestedPath);

  return (
    <div className="space-y-3">
      <LabeledPath label="The data was at" diff={diff} side="old" />
      <LabeledPath
        label="But it looks like it is now at"
        diff={diff}
        side="new"
      />
      <div className="flex flex-wrap gap-2 pt-2">
        <Button type="button" size="sm" onClick={onConfirm} disabled={busy}>
          {busy ? "Checking files..." : "Yes, that is correct"}
        </Button>
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={onReject}
          disabled={busy}
        >
          No, I know where it is
        </Button>
      </div>
    </div>
  );
}

interface LabeledPathProps {
  label: string;
  diff: ReturnType<typeof diffPaths>;
  side: "old" | "new";
}

function LabeledPath({ label, diff, side }: LabeledPathProps) {
  const prefix = diff.prefixParts.join("/");
  const mid = (side === "old" ? diff.oldMidParts : diff.newMidParts).join("/");
  const suffix = diff.suffixParts.join("/");

  const prefixNeedsTrailingSlash = prefix && (mid || suffix);
  const midNeedsTrailingSlash = mid && suffix;

  return (
    <div className="space-y-1">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="ml-3 flex items-center gap-2 rounded-md border bg-white px-3 py-2 text-xs">
        <Folder className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
        <span className="font-mono break-all text-muted-foreground">
          {prefix}
          {prefixNeedsTrailingSlash ? "/" : ""}
          {mid && (
            <span className="font-semibold text-foreground">{mid}</span>
          )}
          {midNeedsTrailingSlash ? "/" : ""}
          {suffix}
        </span>
      </div>
    </div>
  );
}

interface ManualPromptProps {
  missingPath: string;
  onChoose: () => void;
}

function ManualPrompt({ missingPath, onChoose }: ManualPromptProps) {
  return (
    <div className="space-y-3">
      <div className="space-y-1">
        <div className="text-xs text-muted-foreground">The data was at</div>
        <div className="ml-3 flex items-center gap-2 rounded-md border bg-white px-3 py-2 text-xs">
          <Folder className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
          <span className="font-mono break-all text-muted-foreground">
            {missingPath}
          </span>
        </div>
      </div>
      <div className="flex items-center gap-3 pt-1">
        <Button type="button" size="sm" onClick={onChoose}>
          <FolderSearch className="h-4 w-4 mr-1.5" />
          Choose folder
        </Button>
      </div>
    </div>
  );
}
