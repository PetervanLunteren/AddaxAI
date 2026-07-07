/**
 * Promote-to-research-project dialog.
 *
 * Folder runs default to a UTC timezone and the folder's basename as
 * the project name. Promotion lifts the run into the full Research
 * projects experience and asks the user for:
 *
 * - A project name (prefilled with the folder run's name)
 * - An IANA timezone (the only field a research project genuinely
 *   needs that a folder run does not — every other configuration
 *   field has a sensible default)
 *
 * Mechanically the promotion is a single PATCH on the project that
 * flips `mode` from 'folder_run' to 'research', clears
 * `folder_run_state`, and updates the two prompted fields. The same
 * row, same id, same history — verifications carried out during the
 * folder run remain attached.
 */

import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";

import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { TimezoneSelect } from "../ui/timezone-select";
import { projectsApi } from "../../api/projects";

interface PromoteDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The project id of the folder run being promoted. */
  runId: string;
  /** Current folder-run name, used as the default project name. */
  defaultName: string;
}

/** Best-effort browser-system timezone. Falls back to UTC if the
 * Intl API is unavailable (older WebViews on Linux). */
function detectSystemTimezone(): string {
  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
    return tz || "UTC";
  } catch {
    return "UTC";
  }
}

export function PromoteDialog({
  open,
  onOpenChange,
  runId,
  defaultName,
}: PromoteDialogProps) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [name, setName] = useState(defaultName);
  const [timezone, setTimezone] = useState(detectSystemTimezone());
  const [error, setError] = useState<string | null>(null);

  // Re-seed when the dialog reopens with a different run.
  useEffect(() => {
    if (open) {
      setName(defaultName);
      setError(null);
    }
  }, [open, defaultName]);

  const promote = useMutation({
    mutationFn: () =>
      projectsApi.update(runId, {
        mode: "research",
        name: name.trim(),
        timezone,
        folder_run_state: null,
      }),
    onSuccess: () => {
      // The newly-promoted project should show up in the Research
      // projects list. Invalidate that plus the folder-run + project
      // detail queries the dashboard route will hit on mount.
      queryClient.invalidateQueries({ queryKey: ["projects", "research"] });
      queryClient.invalidateQueries({ queryKey: ["folder-run", runId] });
      queryClient.invalidateQueries({ queryKey: ["projects", runId] });
      onOpenChange(false);
      navigate(`/projects/${runId}/dashboard`);
    },
    onError: (err) => {
      setError(
        err instanceof Error ? err.message : "Could not promote the run",
      );
    },
  });

  const canSubmit = name.trim().length > 0 && !promote.isPending;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Turn this run into a project</DialogTitle>
          <DialogDescription>
            Keep this folder run as a full project with species counts,
            verification history, dashboards, maps, and exports. The
            analysis you already ran carries over, nothing is re-run.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          <div className="space-y-2">
            <Label htmlFor="promote-name">Project name</Label>
            <Input
              id="promote-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoFocus
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="promote-tz">Camera timezone</Label>
            <TimezoneSelect value={timezone} onChange={setTimezone} />
            <p className="text-xs text-muted-foreground">
              The wall-clock timezone the cameras were configured to.
              Used by activity plots and Camtrap DP export. Pick a
              regional zone (DST aware) or a fixed offset depending on
              how the cameras were set up.
            </p>
          </div>

          {error && (
            <p className="text-sm text-destructive">{error}</p>
          )}
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={promote.isPending}
          >
            Cancel
          </Button>
          <Button
            onClick={() => promote.mutate()}
            disabled={!canSubmit}
          >
            {promote.isPending ? "Creating..." : "Create project"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
