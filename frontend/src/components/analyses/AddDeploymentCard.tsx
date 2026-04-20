/**
 * Add Deployment Card Component
 *
 * Redesigned to match Create Project modal style.
 * Clean, simple inputs with info tooltips and inline validation.
 */

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
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
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { deploymentQueueApi } from "@/api/deployment-queue";
import { deploymentsApi } from "@/api/deployments";
import { Label } from "@/components/ui/label";
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
  const [touchedFields, setTouchedFields] = useState({ folder: false, site: false });
  const [datetimeOffsetSeconds, setDatetimeOffsetSeconds] = useState(0);
  const [offsetModalOpen, setOffsetModalOpen] = useState(false);
  const [notes, setNotes] = useState("");
  const [tags, setTags] = useState<Record<string, string>>({});

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
  const isDuplicate = Boolean(blockingDeployment || blockingQueueEntry);
  const isValid =
    folderPath && hasFiles && !isDuplicate && !isScanning && !scanResult?.missing_datetime;

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
  } else if (blockingDeployment) {
    validationMessages.push(
      `This folder is already a deployment in this project (id ${blockingDeployment.id.slice(0, 8)}). Delete it from the Deployments page first.`,
    );
  } else if (blockingQueueEntry) {
    validationMessages.push(
      `This folder is already in the queue (status: ${blockingQueueEntry.status}). Remove it from the queue first.`,
    );
  }
  if (scanResult?.missing_datetime) {
    validationMessages.push("DateTime metadata is required but not found in files");
  }

  const handleSubmit = () => {
    if (!folderPath || !scanResult) return;

    addToQueue.mutate({
      folder_path: folderPath,
      site_id: siteId,
      video_count: scanResult.video_count,
      image_count: scanResult.image_count,
      datetime_offset_seconds: datetimeOffsetSeconds || null,
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
            Configure a new deployment to analyze camera trap images
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Folder selector */}
          <FolderSelector
            value={folderPath}
            onChange={(path) => {
              setFolderPath(path);
              setDatetimeOffsetSeconds(0); // Reset offset when folder changes
              setTouchedFields((prev) => ({ ...prev, folder: true }));
            }}
            datetimeOffsetSeconds={datetimeOffsetSeconds}
            onAdjustDates={() => setOffsetModalOpen(true)}
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
            allowEmpty
          />

          {/* Notes */}
          <div className="space-y-2">
            <Label htmlFor="deployment-notes">Notes</Label>
            <Textarea
              id="deployment-notes"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              maxLength={1000}
              placeholder="e.g., Camera angled slightly left to avoid sun glare"
            />
          </div>

          {/* Tags */}
          <TagsEditor value={tags} onChange={setTags} />
        </CardContent>

        <CardFooter className="flex-col gap-3 items-stretch">
          {/* Surface duplicate-blocker messages above the disabled button so
              users don't have to hover to see why they're blocked. */}
          {(blockingDeployment || blockingQueueEntry) && (
            <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
              {blockingDeployment ? (
                <>
                  This folder is already a deployment in this project. Delete it
                  from the{" "}
                  <a
                    href={`/projects/${projectId}/deployments`}
                    className="font-medium underline underline-offset-2"
                  >
                    Deployments page
                  </a>{" "}
                  first (deployment id{" "}
                  <code className="font-mono text-xs">
                    {blockingDeployment.id.slice(0, 8)}
                  </code>
                  ).
                </>
              ) : (
                <>
                  This folder is already in the queue (status:{" "}
                  <strong>{blockingQueueEntry?.status}</strong>). Remove the
                  queue entry first if you want to re-add this folder.
                </>
              )}
            </div>
          )}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="w-full">
                  <Button
                    onClick={handleSubmit}
                    disabled={!isValid || addToQueue.isPending}
                    className="w-full"
                    size="lg"
                  >
                    <Plus className="h-4 w-4 mr-2" />
                    {addToQueue.isPending ? "Adding..." : "Add to queue"}
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
        />
      )}

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
