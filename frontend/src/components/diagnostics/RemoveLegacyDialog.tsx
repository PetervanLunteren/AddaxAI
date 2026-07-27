/**
 * Offer to delete a legacy AddaxAI (v5 / v6) install.
 *
 * AddaxAI 7 installs to different locations than 6, so upgrading leaves
 * two full copies on the machine, the old one holding tens of GB of
 * analysis environments and model weights. This dialog clears the old
 * one out. Opened automatically once per launch by MenuCommands, and on
 * demand from the Help menu.
 *
 * Deliberately a plain confirm rather than TypeToConfirmDialog: nothing
 * removed here is irreversible (reinstalling AddaxAI 6 brings it all
 * back), and a type-to-confirm gate on a dialog that opens by itself is
 * hostile. ResetAppDialog uses one because it can wipe the database.
 */

import { useEffect, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";
import { setupApi } from "../../api/setup";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
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

interface RemoveLegacyDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  dontAskAgain: boolean;
  onDontAskAgainChange: (value: boolean) => void;
}

const IS_MAC = navigator.platform.includes("Mac");

export function RemoveLegacyDialog({
  open,
  onOpenChange,
  dontAskAgain,
  onDontAskAgainChange,
}: RemoveLegacyDialogProps) {
  // Polls only while a removal runs. Deleting a legacy tree means
  // hundreds of thousands of small files and takes minutes on Windows.
  const { data: status, refetch } = useQuery({
    queryKey: ["legacy-install"],
    queryFn: setupApi.getLegacyInstall,
    enabled: open,
    refetchInterval: (query) =>
      query.state.data?.removal_in_progress ? 1500 : false,
  });

  const remove = useMutation({
    mutationFn: setupApi.removeLegacyInstall,
    // Pick up removal_in_progress straight away so polling starts, and
    // again on failure so a 409 surfaces the real state.
    onSettled: () => refetch(),
  });

  const removable = status?.removable_paths ?? [];
  const manual = status?.manual_paths ?? [];
  const removing = Boolean(status?.removal_in_progress) || remove.isPending;
  const error =
    status?.removal_error ??
    (remove.isError ? (remove.error as Error).message : null);

  // Three things this dialog can be showing. "gone" only after a
  // removal actually ran, so opening it from the menu on a clean
  // machine does not claim to have removed something.
  const view =
    removable.length > 0 ? "found" : remove.isSuccess ? "gone" : "clean";

  // Remember the version we saw, because after a successful removal the
  // scan no longer finds it and the success line would otherwise drop
  // back to a bare "AddaxAI 6".
  const [seenVersion, setSeenVersion] = useState<string | null>(null);
  useEffect(() => {
    if (status?.version) setSeenVersion(status.version);
  }, [status?.version]);

  const version = seenVersion ? `AddaxAI ${seenVersion}` : "AddaxAI 6";

  const description = {
    found: `${version} is still on this computer. The new version does not need it, and it takes up a lot of space.`,
    gone: `${version} has been removed.`,
    clean: "No older version of AddaxAI was found.",
  }[view];

  return (
    <Dialog open={open} onOpenChange={(o) => !removing && onOpenChange(o)}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Trash2 className="h-5 w-5" />
            Remove the old AddaxAI?
          </DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>

        {view === "found" && (
          <div className="space-y-3">
            <ul className="space-y-1">
              {removable.map((path) => (
                <li key={path}>
                  <code className="text-xs break-all">{path}</code>
                </li>
              ))}
            </ul>

            <Callout variant="info">
              <strong>Your photos, videos and results stay untouched.</strong>{" "}
              You can install the old version again at any time from the
              AddaxAI website.
            </Callout>

            {IS_MAC && (
              <p className="text-xs text-muted-foreground">
                macOS will ask permission to use your Desktop. That is the old
                shortcut being removed.
              </p>
            )}
          </div>
        )}

        {manual.length > 0 && (
          <Callout variant="warning">
            <p>
              This folder needs administrator rights, so you have to delete it
              yourself:
            </p>
            <ul className="mt-1.5 space-y-1">
              {manual.map((path) => (
                <li key={path}>
                  <code className="text-xs break-all">{path}</code>
                </li>
              ))}
            </ul>
          </Callout>
        )}

        {error && <Callout variant="error">{error}</Callout>}

        {view === "found" && (
          <div className="flex items-center gap-2">
            <Checkbox
              checked={dontAskAgain}
              onCheckedChange={(c) => onDontAskAgainChange(Boolean(c))}
            />
            <Label
              onClick={() => onDontAskAgainChange(!dontAskAgain)}
              className="cursor-pointer"
            >
              Don't ask me again
            </Label>
          </div>
        )}

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={removing}
          >
            {view === "found" ? "Skip for now" : "Close"}
          </Button>
          {view === "found" && (
            <Button
              type="button"
              variant="destructive"
              onClick={() => remove.mutate()}
              disabled={removing}
            >
              {removing ? "Removing..." : "Remove"}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
