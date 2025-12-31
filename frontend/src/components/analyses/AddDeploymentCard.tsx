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
import { FolderSelector } from "./FolderSelector";
import { SiteSelector } from "./SiteSelector";
import { AddSiteModal } from "./AddSiteModal";
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

  // Get folder scan results for validation
  const { data: scanResult, isLoading: isScanning } = useFolderScan(folderPath);

  // Get existing queue entries to check for duplicates
  const { data: queueEntries } = useQuery({
    queryKey: ["deployment-queue", projectId],
    queryFn: () => deploymentQueueApi.list(projectId),
  });

  // Add to queue mutation
  const addToQueue = useMutation({
    mutationFn: (data: { folder_path: string; site_id: string }) =>
      deploymentQueueApi.create({
        project_id: projectId,
        folder_path: data.folder_path,
        site_id: data.site_id,
      }),
    onSuccess: () => {
      // Refresh queue
      queryClient.invalidateQueries({ queryKey: ["deployment-queue", projectId] });

      // Clear folder (but keep site selected)
      setFolderPath(null);
    },
    onError: (error) => {
      // Only show error alerts
      alert(`Failed to add to queue: ${error instanceof Error ? error.message : "Unknown error"}`);
    },
  });

  // Validation
  const hasFiles = scanResult && scanResult.total_count > 0;
  const isDuplicate =
    folderPath && queueEntries?.some((e) => e.folder_path === folderPath);
  const isValid = folderPath && hasFiles && siteId && !isDuplicate && !isScanning;

  // Validation messages (for button tooltip)
  const validationMessages: string[] = [];
  if (!folderPath) {
    validationMessages.push("Select a folder");
  } else if (isScanning) {
    validationMessages.push("Scanning folder...");
  } else if (!hasFiles) {
    validationMessages.push("Selected folder contains no images");
  } else if (isDuplicate) {
    validationMessages.push("This folder is already in the queue");
  }
  if (!siteId) {
    validationMessages.push("Select a camera site");
  }

  const handleSubmit = () => {
    if (!folderPath || !siteId) return;

    addToQueue.mutate({
      folder_path: folderPath,
      site_id: siteId,
    });
  };

  const handleSiteCreated = (newSiteId: string) => {
    setSiteId(newSiteId);
  };

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Add deployment to queue</CardTitle>
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
              setTouchedFields((prev) => ({ ...prev, folder: true }));
            }}
          />

          {/* Site selector */}
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
        </CardContent>

        <CardFooter>
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
