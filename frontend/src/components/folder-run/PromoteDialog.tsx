/**
 * Promote-to-research-project dialog.
 *
 * Folder runs default to the folder's basename as the project name.
 * Promotion lifts the run into the full Research projects experience
 * and asks the user for:
 *
 * - A project name (prefilled with the folder run's name)
 * - An optional description
 *
 * No timezone is asked for. A folder run is created with a placeholder
 * UTC timezone (it never exposes the sun / Camtrap features that need
 * one), so promotion clears it back to null — exactly the state a fresh
 * project starts in. The backend then auto-derives the real camera
 * timezone from the first site's coordinates (see crud/site.py), the
 * same path new projects take. This matters: leaving the placeholder
 * "UTC" in place would make the site auto-derive skip (it only fills a
 * null timezone), silently pinning the project to UTC forever.
 *
 * Mechanically the promotion is a single PATCH on the project that
 * flips `mode` from 'folder_run' to 'research', clears
 * `folder_run_state`, nulls the timezone, and sets name + description.
 * The same row, same id, same history — verifications carried out
 * during the folder run remain attached. Every other processing
 * setting (models, thresholds, smoothing) is already on this row from
 * the run and carries over untouched.
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
import { Textarea } from "../ui/textarea";
import { projectsApi } from "../../api/projects";

interface PromoteDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The project id of the folder run being promoted. */
  runId: string;
  /** Current folder-run name, used as the default project name. */
  defaultName: string;
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
  const [description, setDescription] = useState("");
  const [error, setError] = useState<string | null>(null);

  // Re-seed when the dialog reopens with a different run.
  useEffect(() => {
    if (open) {
      setName(defaultName);
      setDescription("");
      setError(null);
    }
  }, [open, defaultName]);

  const promote = useMutation({
    mutationFn: () =>
      projectsApi.update(runId, {
        mode: "research",
        name: name.trim(),
        description: description.trim() || null,
        // Explicit null clears the folder-run placeholder so the project
        // starts like a fresh one: the first sited site derives the real
        // camera timezone from its GPS.
        timezone: null,
        folder_run_state: null,
      }),
    onSuccess: () => {
      // The newly-promoted project should show up in the Research
      // projects list. Invalidate that plus the folder-run + project
      // detail queries the dashboard route will hit on mount.
      queryClient.invalidateQueries({ queryKey: ["projects", "research"] });
      // The run row is gone once it is a project, so a refetch would only
      // log a 404; drop the cached entry instead.
      queryClient.removeQueries({ queryKey: ["folder-run", runId] });
      queryClient.invalidateQueries({ queryKey: ["projects", runId] });
      // ...and out of the step-1 "recent runs" list, which only carries
      // mode='folder_run' rows. This is a different key from the
      // ["folder-run", runId] detail above, so it needs its own call:
      // left stale, the list would keep offering a run that now 404s.
      queryClient.invalidateQueries({ queryKey: ["folder-runs"] });
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
            <Label htmlFor="promote-description">Description</Label>
            <Textarea
              id="promote-description"
              placeholder="Notes about purpose, location, team members, etc."
              className="resize-y"
              rows={2}
              maxLength={500}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
            <div className="flex justify-end">
              <p
                className={`text-xs ${
                  description.length > 450
                    ? "text-orange-600"
                    : "text-muted-foreground"
                }`}
              >
                {description.length} / 500
              </p>
            </div>
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
