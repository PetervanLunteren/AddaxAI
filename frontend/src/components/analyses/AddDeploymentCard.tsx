/**
 * Add Deployment Card Component
 *
 * Redesigned to match Create Project modal style.
 * Clean, simple inputs with info tooltips and inline validation.
 */

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { Plus } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Callout } from "@/components/ui/callout";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { deploymentQueueApi } from "@/api/deployment-queue";
import { deploymentsApi } from "@/api/deployments";
import { Label } from "@/components/ui/label";
import { FieldHeader } from "@/components/ui/field-header";
import { Textarea } from "@/components/ui/textarea";
import { TagsEditor } from "@/components/ui/tags-editor";
import { FolderSelector } from "./FolderSelector";
import { SiteSelector } from "./SiteSelector";
import { AddSiteModal } from "./AddSiteModal";
import { DatetimeOffsetModal } from "./DatetimeOffsetModal";
import { useFolderScan } from "@/hooks/useFolderScan";

interface AddDeploymentCardProps {
  projectId: string;
}

export function AddDeploymentCard({ projectId }: AddDeploymentCardProps) {
  const queryClient = useQueryClient();

  // Form state
  const [folderPath, setFolderPath] = useState<string | null>(null);
  const [siteId, setSiteId] = useState<string | null>(null);
  const [showAddSiteModal, setShowAddSiteModal] = useState(false);
  const [, setTouchedFields] = useState({ folder: false, site: false });
  const [datetimeOffsetSeconds, setDatetimeOffsetSeconds] = useState(0);
  const [fileMtimeChecked, setFileMtimeChecked] = useState(false);
  const [offsetModalOpen, setOffsetModalOpen] = useState(false);
  const [notes, setNotes] = useState("");
  const [tags, setTags] = useState<Record<string, string>>({});
  // Reprocess confirmation: shown when the folder is already a deployment.
  const [confirmReprocess, setConfirmReprocess] = useState(false);

  // Get folder scan results for validation
  const { data: scanResult, isLoading: isScanning } = useFolderScan(folderPath);

  // Existing queue entries and project deployments — used to block
  // re-adding a folder that's already accounted for. We block on:
  //   - any deployment in this project pointing at the same folder
  //     (it's already in the DB, no duplicates)
  //   - any queue entry with status pending/processing (it's about to
  //     become a deployment)
  // Failed queue entries do NOT block: they were never persisted as a
  // deployment, so re-adding the same folder is a legitimate retry.
  const { data: queueEntries } = useQuery({
    queryKey: ["deployment-queue", projectId],
    queryFn: () => deploymentQueueApi.list(projectId),
  });
  const { data: projectDeployments } = useQuery({
    queryKey: ["deployments", projectId],
    queryFn: () => deploymentsApi.list({ projectId }),
  });

  // Add to queue mutation
  const addToQueue = useMutation({
    mutationFn: (data: {
      folder_path: string;
      site_id: string | null;
      video_count: number;
      image_count: number;
      datetime_offset_seconds: number | null;
      use_file_mtime_fallback: boolean;
      notes: string | null;
      tags: Record<string, string>;
    }) =>
      deploymentQueueApi.create({
        project_id: projectId,
        folder_path: data.folder_path,
        site_id: data.site_id,
        video_count: data.video_count,
        image_count: data.image_count,
        datetime_offset_seconds: data.datetime_offset_seconds || null,
        use_file_mtime_fallback: data.use_file_mtime_fallback,
        notes: data.notes,
        tags: data.tags,
      }),
    onSuccess: () => {
      // Refresh queue
      queryClient.invalidateQueries({ queryKey: ["deployment-queue", projectId] });

      // Clear form
      setFolderPath(null);
      setSiteId(null);
      setDatetimeOffsetSeconds(0);
      setFileMtimeChecked(false);
      setNotes("");
      setTags({});
    },
    onError: (error) => {
      // Only show error alerts
      alert(`Failed to add to queue: ${error instanceof Error ? error.message : "Unknown error"}`);
    },
  });

  // Validation
  const hasFiles = scanResult && scanResult.total_count > 0;
  // Only send the opt-in while the scan still reports no capture dates.
  // Guards the case where the box is ticked and a later refetch turns up
  // real dates: the checkbox unmounts, but raw state would persist.
  const useFileMtimeFallback = Boolean(
    fileMtimeChecked && scanResult?.missing_datetime,
  );
  const blockingDeployment = folderPath
    ? projectDeployments?.find((d) => d.folder_path === folderPath)
    : undefined;
  const blockingQueueEntry = folderPath
    ? queueEntries?.find(
        (e) =>
          e.folder_path === folderPath &&
          (e.status === "pending" || e.status === "processing"),
      )
    : undefined;
  // A pending/processing queue entry still blocks (can't reprocess something
  // mid-flight). An existing deployment no longer blocks: adding it again
  // reprocesses the folder after a confirm (see handleSubmit).
  // Missing capture dates no longer block adding a deployment: the backend
  // ingests date-less files (they drop out of time-based stats), and the
  // folder scan shows a non-blocking note about it.
  const isValid = Boolean(
    folderPath && hasFiles && !blockingQueueEntry && !isScanning,
  );

  // Validation messages (for button tooltip). Site is optional (users
  // can run deployment-agnostic batches), so a missing site no longer
  // blocks submission.
  const validationMessages: string[] = [];
  if (!folderPath) {
    validationMessages.push("Select a folder");
  } else if (isScanning) {
    validationMessages.push("Scanning folder...");
  } else if (!hasFiles) {
    validationMessages.push("Selected folder contains no images");
  } else if (blockingQueueEntry) {
    validationMessages.push(
      `This folder is already in the queue (status: ${blockingQueueEntry.status}). Remove it from the queue first.`,
    );
  }

  // Reprocess: delete the existing deployment (cascades away its files,
  // detections, events, observations and verifications), then queue the
  // folder again. Sequential; if the delete fails nothing is queued.
  const reprocess = useMutation({
    mutationFn: async () => {
      if (!blockingDeployment || !folderPath || !scanResult) return;
      await deploymentsApi.delete(blockingDeployment.id);
      await deploymentQueueApi.create({
        project_id: projectId,
        folder_path: folderPath,
        site_id: siteId,
        video_count: scanResult.video_count,
        image_count: scanResult.image_count,
        datetime_offset_seconds: datetimeOffsetSeconds || null,
        use_file_mtime_fallback: useFileMtimeFallback,
        notes: notes.trim() || null,
        tags,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["deployments", projectId] });
      queryClient.invalidateQueries({ queryKey: ["deployment-queue", projectId] });
      setConfirmReprocess(false);
      setFolderPath(null);
      setSiteId(null);
      setDatetimeOffsetSeconds(0);
      setFileMtimeChecked(false);
      setNotes("");
      setTags({});
    },
    onError: (error) => {
      alert(
        `Failed to reprocess: ${error instanceof Error ? error.message : "Unknown error"}`,
      );
    },
  });

  const handleSubmit = () => {
    if (!folderPath || !scanResult) return;

    // Already a deployment: confirm the overwrite before reprocessing.
    if (blockingDeployment) {
      setConfirmReprocess(true);
      return;
    }

    addToQueue.mutate({
      folder_path: folderPath,
      site_id: siteId,
      video_count: scanResult.video_count,
      image_count: scanResult.image_count,
      datetime_offset_seconds: datetimeOffsetSeconds || null,
      use_file_mtime_fallback: useFileMtimeFallback,
      notes: notes.trim() || null,
      tags,
    });
  };

  const handleSiteCreated = (newSiteId: string) => {
    setSiteId(newSiteId);
  };

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>New deployment</CardTitle>
          <CardDescription>
            Pick a folder to add it to the queue. Everything else is optional
            and can be added later.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Folder selector */}
          <FolderSelector
            value={folderPath}
            onChange={(path) => {
              setFolderPath(path);
              setDatetimeOffsetSeconds(0); // Reset offset when folder changes
              setFileMtimeChecked(false); // and the file-date opt-in
              setTouchedFields((prev) => ({ ...prev, folder: true }));
            }}
            datetimeOffsetSeconds={datetimeOffsetSeconds}
            onAdjustDates={() => setOffsetModalOpen(true)}
            missingDateNote="AddaxAI will still detect and classify these files, but with no date they are left out of time-based stats, charts, and trap-night effort."
            useFileMtimeFallback={useFileMtimeFallback}
            onUseFileMtimeFallbackChange={setFileMtimeChecked}
            caption="The folder with the images or videos you want to analyse. Subfolders are included."
          />

          {/* Site selector (optional). When the user leaves it blank
              the deployment is created without a camera site, which is
              valid for batches spanning multiple locations or backlog
              data where the location is unknown. */}
          <SiteSelector
            projectId={projectId}
            value={siteId}
            onChange={(id) => {
              setSiteId(id);
              setTouchedFields((prev) => ({ ...prev, site: true }));
            }}
            onAddNew={() => setShowAddSiteModal(true)}
            deploymentGps={scanResult?.gps_location ?? null}
          />

          {/* Notes */}
          <div className="space-y-2">
            <FieldHeader
              label={<Label htmlFor="deployment-notes">Notes</Label>}
              caption="Free-text for your own records. Shown on the deployment's info panel."
            />
            <Textarea
              id="deployment-notes"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              maxLength={1000}
              placeholder="e.g., Lens was covered in baboon fingerprints"
            />
          </div>

          {/* Tags */}
          <TagsEditor
            value={tags}
            onChange={setTags}
            keyPlaceholder="e.g., season"
            valuePlaceholder="e.g., wet"
            description="Labels to group and filter deployments later."
          />
        </CardContent>

        <CardFooter className="flex-col gap-3 items-stretch">
          {/* Surface duplicate-blocker messages above the disabled button so
              users don't have to hover to see why they're blocked. */}
          {(blockingDeployment || blockingQueueEntry) && (
            <Callout
              variant="warning"
              size="compact"
              action={
                blockingDeployment ? (
                  <Button
                    asChild
                    variant="outline"
                    size="sm"
                    className="shrink-0 border-amber-300 bg-white text-amber-900 hover:bg-amber-100"
                  >
                    <Link
                      to={`/projects/${projectId}/deployments?info=${blockingDeployment.id}`}
                    >
                      View
                    </Link>
                  </Button>
                ) : undefined
              }
            >
              {blockingDeployment ? (
                <>This folder is already a deployment in this project.</>
              ) : (
                <>
                  This folder is already in the queue (status:{" "}
                  <strong>{blockingQueueEntry?.status}</strong>).
                </>
              )}
            </Callout>
          )}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="w-full">
                  <Button
                    onClick={handleSubmit}
                    disabled={!isValid || addToQueue.isPending || reprocess.isPending}
                    className="w-full"
                    size="lg"
                  >
                    <Plus className="h-4 w-4 mr-2" />
                    {addToQueue.isPending
                      ? "Adding..."
                      : blockingDeployment
                        ? "Reprocess folder"
                        : "Add to queue"}
                  </Button>
                </div>
              </TooltipTrigger>
              {!isValid && validationMessages.length > 0 && (
                <TooltipContent>
                  <div className="space-y-1">
                    {validationMessages.map((msg, index) => (
                      <p key={index} className="text-sm">
                        • {msg}
                      </p>
                    ))}
                  </div>
                </TooltipContent>
              )}
            </Tooltip>
          </TooltipProvider>
        </CardFooter>
      </Card>

      {/* Datetime offset modal */}
      {folderPath && scanResult && (
        <DatetimeOffsetModal
          open={offsetModalOpen}
          onOpenChange={setOffsetModalOpen}
          sampleFiles={scanResult.sample_files}
          folderPath={folderPath}
          currentOffsetSeconds={datetimeOffsetSeconds}
          onApply={setDatetimeOffsetSeconds}
          useFileMtimeFallback={useFileMtimeFallback}
        />
      )}

      {/* Reprocess confirmation */}
      <AlertDialog open={confirmReprocess} onOpenChange={setConfirmReprocess}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Reprocess this folder?</AlertDialogTitle>
            <AlertDialogDescription>
              This folder is already in the database. Processing it again
              overwrites the previous predictions, including any verifications
              and count confirmations. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={reprocess.isPending}>
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={(e) => {
                e.preventDefault();
                reprocess.mutate();
              }}
              disabled={reprocess.isPending}
            >
              {reprocess.isPending ? "Reprocessing..." : "Overwrite and reprocess"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Add site modal */}
      <AddSiteModal
        projectId={projectId}
        open={showAddSiteModal}
        onOpenChange={setShowAddSiteModal}
        onSiteCreated={handleSiteCreated}
        initialLocation={
          scanResult?.gps_location
            ? { lat: scanResult.gps_location.latitude, lon: scanResult.gps_location.longitude }
            : undefined
        }
      />
    </>
  );
}
