/**
 * Add Deployment Card Component
 *
 * Form to configure and add a deployment to the analysis queue.
 * Components:
 * - FolderSelector (with scan validation)
 * - SiteSelector (with add new site option)
 * - ProjectModelsInfo (readonly)
 * - Add to queue button (with validation)
 *
 * Validation:
 * - Folder must have files
 * - Site must be selected
 * - Folder not already in queue
 */

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus, AlertCircle } from "lucide-react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { deploymentQueueApi } from "@/api/deployment-queue";
import { FolderSelector } from "./FolderSelector";
import { SiteSelector } from "./SiteSelector";
import { ProjectModelsInfo } from "./ProjectModelsInfo";
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

      // Success feedback
      // TODO: Replace with proper toast notification
      alert("Added to queue successfully!");
    },
    onError: (error) => {
      alert(`Failed to add to queue: ${error instanceof Error ? error.message : "Unknown error"}`);
    },
  });

  // Validation
  const hasFiles = scanResult && scanResult.total_count > 0;
  const isDuplicate =
    folderPath && queueEntries?.some((e) => e.folder_path === folderPath);
  const isValid = folderPath && hasFiles && siteId && !isDuplicate && !isScanning;

  // Validation errors
  const errors: string[] = [];
  if (folderPath && !isScanning && !hasFiles) {
    errors.push("No images found in selected folder");
  }
  if (isDuplicate) {
    errors.push("This folder is already in the queue");
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
            onChange={setFolderPath}
            error={errors.length > 0 ? errors[0] : undefined}
          />

          {/* Site selector */}
          <SiteSelector
            projectId={projectId}
            value={siteId}
            onChange={setSiteId}
            onAddNew={() => setShowAddSiteModal(true)}
          />

          {/* Project models info (readonly) */}
          <ProjectModelsInfo projectId={projectId} />

          {/* Validation errors */}
          {errors.length > 0 && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                <ul className="list-disc list-inside space-y-1">
                  {errors.map((error, index) => (
                    <li key={index}>{error}</li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          )}
        </CardContent>

        <CardFooter>
          <Button
            onClick={handleSubmit}
            disabled={!isValid || addToQueue.isPending}
            className="w-full"
            size="lg"
          >
            <Plus className="h-4 w-4 mr-2" />
            {addToQueue.isPending ? "Adding..." : "Add to queue"}
          </Button>
        </CardFooter>
      </Card>

      {/* Add site modal */}
      <AddSiteModal
        projectId={projectId}
        open={showAddSiteModal}
        onOpenChange={setShowAddSiteModal}
        onSiteCreated={handleSiteCreated}
      />
    </>
  );
}
