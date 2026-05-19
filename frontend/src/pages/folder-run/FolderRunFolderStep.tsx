/**
 * Step 1: Choose folder.
 *
 * Reuses FolderSelector (already used by AddDeploymentCard and the
 * Timelapse window). After the folder is scanned, we probe the
 * backend for an existing folder-run project pointing at the same
 * source folder. When one exists the "already analysed" notice card
 * replaces the Continue button and forces the user to pick between
 * opening the previous run or discarding it.
 *
 * Validation: a folder must be selected, the scan must complete, the
 * folder must contain media, and EXIF DateTimeOriginal must be present
 * on at least the sampled files (the analysis pipeline crashes on
 * missing timestamps; we surface that early instead of letting the
 * run fail in step 3).
 */

import { useEffect, useRef, useState } from "react";
import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowRight, History } from "lucide-react";

import { Button } from "../../components/ui/button";
import {
  Card,
  CardContent,
  CardFooter,
} from "../../components/ui/card";
import { StepHeader } from "../../components/folder-run/StepHeader";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../../components/ui/dialog";
import { FolderSelector } from "../../components/analyses/FolderSelector";
import { useFolderScan } from "../../hooks/useFolderScan";
import {
  folderRunsApi,
  type FolderRunLookup,
} from "../../api/folder-runs";
import { useFolderRun } from "./FolderRunLayout";

export function FolderRunFolderStep() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { runId: currentRunId, run: currentRun } = useFolderRun();
  const [folderPath, setFolderPath] = useState<string | null>(null);
  const [discardOpen, setDiscardOpen] = useState(false);

  // When the user back-navigates to Step 1 from inside a run (chip
  // click, breadcrumb, etc.), pre-fill the selector with the folder
  // they already picked. The ref makes this fire exactly once per
  // mount: if the user then hits the FolderSelector's "Change"
  // button to clear the path, the effect won't re-prefill it.
  const hasPrefilledRef = useRef(false);
  useEffect(() => {
    if (
      !hasPrefilledRef.current &&
      currentRun?.queue_entry?.folder_path
    ) {
      hasPrefilledRef.current = true;
      setFolderPath(currentRun.queue_entry.folder_path);
    }
  }, [currentRun]);

  const { data: scanResult, isLoading: isScanning } =
    useFolderScan(folderPath);

  // Lookup only fires after the folder scan tells us we have a valid
  // folder with media — no point probing for paths the user is
  // mid-typing or that turned out to be empty.
  const lookupReady =
    !!folderPath && !!scanResult && scanResult.total_count > 0;
  const { data: lookupRun, isLoading: isLookingUp } = useQuery({
    queryKey: ["folder-run-lookup", folderPath],
    queryFn: () => folderRunsApi.lookup(folderPath!),
    enabled: lookupReady,
    staleTime: 30_000,
  });

  // The lookup endpoint matches any folder-run project pointing at
  // this folder, including the one the user is currently inside.
  // Suppress the "previous run" notice when the match is the user's
  // own run — that's not a previous run, that's where they are.
  const existingRun =
    lookupRun && lookupRun.id === currentRunId ? null : lookupRun;

  const createRun = useMutation({
    mutationFn: folderRunsApi.create,
    onSuccess: (run) => {
      // Prime the cache so the next step mounts with the run loaded.
      queryClient.setQueryData(["folder-run", run.project.id], run);
      // After a discard-and-create, the old lookup result is stale.
      queryClient.invalidateQueries({
        queryKey: ["folder-run-lookup", folderPath],
      });
      navigate(`/folder-runs/${run.project.id}/${run.step}`);
    },
  });

  const hasFiles = !!scanResult && scanResult.total_count > 0;
  const missingTimestamps = scanResult?.missing_datetime ?? false;

  // Continue button is hidden only when a *different* run already
  // exists for this folder — that's the case where the notice card
  // takes over with Open/Discard actions. When we're already inside
  // a run, Continue stays visible but switches to navigate-forward
  // mode instead of creating a duplicate run.
  const showContinueButton =
    !!folderPath && !isScanning && !isLookingUp && !existingRun;
  const canContinue =
    showContinueButton && hasFiles && !missingTimestamps;

  const handleSubmit = () => {
    if (!folderPath || !scanResult) return;
    // When the user is inside an existing run AND the path they're
    // looking at is still that run's folder, Continue is just
    // forward navigation — no new project needed. If they changed
    // the folder (Change button → new pick), we fall through to
    // createRun and the new folder gets its own run.
    if (
      currentRunId &&
      folderPath === currentRun?.queue_entry?.folder_path
    ) {
      navigate(`/folder-runs/${currentRunId}/model`);
      return;
    }
    createRun.mutate({
      source_folder: folderPath,
      image_count: scanResult.image_count,
      video_count: scanResult.video_count,
    });
  };

  const handleOpenPrevious = () => {
    if (!existingRun) return;
    navigate(`/folder-runs/${existingRun.id}/${existingRun.step}`);
  };

  const handleConfirmDiscard = () => {
    if (!folderPath || !scanResult) return;
    setDiscardOpen(false);
    createRun.mutate({
      source_folder: folderPath,
      image_count: scanResult.image_count,
      video_count: scanResult.video_count,
      force_new: true,
    });
  };

  return (
    <>
      <StepHeader
        title="Choose folder"
        caption="Pick the folder with the images or videos you want to analyse."
      />
      <Card>
        <CardContent className="space-y-4 p-6">
          <FolderSelector value={folderPath} onChange={setFolderPath} />

          {existingRun && (
            <PreviousRunNotice
              run={existingRun}
              isBusy={createRun.isPending}
              onOpenPrevious={handleOpenPrevious}
              onDiscard={() => setDiscardOpen(true)}
            />
          )}

          {createRun.isError && (
            <p className="text-sm text-destructive">
              Could not start the folder run:{" "}
              {createRun.error instanceof Error
                ? createRun.error.message
                : "unknown error"}
            </p>
          )}
        </CardContent>

        {showContinueButton && (
          <CardFooter className="justify-end">
            <Button
              size="lg"
              onClick={handleSubmit}
              disabled={!canContinue || createRun.isPending}
              className="gap-2"
            >
              {createRun.isPending ? "Starting..." : "Continue"}
              <ArrowRight className="h-4 w-4" />
            </Button>
          </CardFooter>
        )}

        <DiscardConfirmDialog
          open={discardOpen}
          run={existingRun ?? null}
          onCancel={() => setDiscardOpen(false)}
          onConfirm={handleConfirmDiscard}
        />
      </Card>
    </>
  );
}

// ─────────────────────────────────────────────────────────────────
// Notice card + destructive confirm dialog
// ─────────────────────────────────────────────────────────────────

function PreviousRunNotice({
  run,
  isBusy,
  onOpenPrevious,
  onDiscard,
}: {
  run: FolderRunLookup;
  isBusy: boolean;
  onOpenPrevious: () => void;
  onDiscard: () => void;
}) {
  return (
    <div className="rounded-md border bg-card-background p-4">
      <div className="flex items-start gap-3">
        <History className="mt-0.5 h-5 w-5 shrink-0 text-primary" />
        <div className="flex-1 space-y-3">
          <div>
            <p className="text-sm font-medium">
              You analysed this folder before
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {formatRunSummary(run)}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {formatRunCounts(run)}
            </p>
            {formatVerificationProgress(run) && (
              <p className="mt-1 text-xs text-muted-foreground">
                {formatVerificationProgress(run)}
              </p>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="sm"
              onClick={onOpenPrevious}
              disabled={isBusy}
            >
              Open previous run
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={onDiscard}
              disabled={isBusy}
              className="text-destructive hover:text-destructive"
            >
              Discard and start over
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

function DiscardConfirmDialog({
  open,
  run,
  onCancel,
  onConfirm,
}: {
  open: boolean;
  run: FolderRunLookup | null;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <Dialog open={open} onOpenChange={(v) => (v ? null : onCancel())}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Discard the previous run?</DialogTitle>
          <DialogDescription>
            Starting a new run will delete the previous run from the
            database, including any manual verifications on it. The
            output files you already saved to disk are not affected.
            {run && run.verified_file_count > 0 && (
              <>
                {" "}This previous run has{" "}
                <span className="font-medium">
                  {run.verified_file_count.toLocaleString()} verified
                  file{run.verified_file_count === 1 ? "" : "s"}
                </span>
                .
              </>
            )}
            <br />
            <span className="mt-2 block text-xs">
              A database snapshot from earlier today is in your
              backups folder if you change your mind.
            </span>
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button variant="destructive" onClick={onConfirm}>
            Discard and start over
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ─────────────────────────────────────────────────────────────────
// Formatters for the notice card
// ─────────────────────────────────────────────────────────────────

function formatRunSummary(run: FolderRunLookup): string {
  const date = formatDate(run.updated_at_utc);
  // Prefer the friendly names; fall back to the ids when the manifest
  // doesn't know the model (catalog drift, fresh install). The backend
  // already does that fallback, so we just trust whatever it gave us.
  const models = [
    run.detection_model_name ?? run.detection_model_id,
    run.classification_model_name ?? run.classification_model_id,
  ]
    .filter(Boolean)
    .join(" + ");
  return models ? `${date} · ${models}` : date;
}

function formatRunCounts(run: FolderRunLookup): string {
  const parts: string[] = [];
  parts.push(
    `${run.file_count.toLocaleString()} file${
      run.file_count === 1 ? "" : "s"
    }`,
  );
  if (run.detection_count > 0) {
    parts.push(
      `${run.detection_count.toLocaleString()} detection${
        run.detection_count === 1 ? "" : "s"
      }`,
    );
  }
  if (run.species_count > 0) {
    parts.push(`${run.species_count.toLocaleString()} species`);
  }
  return parts.join(" · ");
}

function formatVerificationProgress(run: FolderRunLookup): string | null {
  // No detections to verify yet (analysis didn't run or every file
  // was blank) → suppress the line entirely.
  if (run.detection_count === 0) return null;

  const verified = run.verified_detection_count;
  const total = run.detection_count;
  if (verified === total) {
    return `All ${total.toLocaleString()} observation${
      total === 1 ? "" : "s"
    } verified`;
  }
  const pct = Math.round((verified / total) * 100);
  return `${verified.toLocaleString()} of ${total.toLocaleString()} observation${
    total === 1 ? "" : "s"
  } verified (${pct}%)`;
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      day: "numeric",
      month: "short",
      year: "numeric",
    });
  } catch {
    return iso;
  }
}
