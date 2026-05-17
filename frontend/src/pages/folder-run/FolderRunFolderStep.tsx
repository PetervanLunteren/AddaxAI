/**
 * Step 1: Choose folder.
 *
 * Reuses FolderSelector (already used by AddDeploymentCard and the
 * Timelapse window), then on submit creates a folder run via
 * /api/folder-runs and navigates to step 2.
 *
 * Validation: a folder must be selected, the scan must complete, the
 * folder must contain media, and EXIF DateTimeOriginal must be present
 * on at least the sampled files (the analysis pipeline crashes on
 * missing timestamps; we surface that early instead of letting the
 * run fail in step 3).
 */

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { Button } from "../../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "../../components/ui/card";
import { FolderSelector } from "../../components/analyses/FolderSelector";
import { useFolderScan } from "../../hooks/useFolderScan";
import { folderRunsApi } from "../../api/folder-runs";

export function FolderRunFolderStep() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [folderPath, setFolderPath] = useState<string | null>(null);

  const { data: scanResult, isLoading: isScanning } =
    useFolderScan(folderPath);

  const createRun = useMutation({
    mutationFn: folderRunsApi.create,
    onSuccess: (run) => {
      // Prime the cache so step 2 mounts with the run already loaded.
      // The endpoint resumes when the same folder has been analysed
      // before (legacy AddaxAI behaviour), so `run.step` may be a
      // later step; the FolderRunResumeIndex on /folder-runs/:id
      // takes the user to the right place if they revisit by id.
      queryClient.setQueryData(["folder-run", run.project.id], run);
      navigate(`/folder-runs/${run.project.id}/${run.step}`);
    },
  });

  const hasFiles = !!scanResult && scanResult.total_count > 0;
  const missingTimestamps = scanResult?.missing_datetime ?? false;
  const canContinue =
    !!folderPath && hasFiles && !isScanning && !missingTimestamps;

  const handleSubmit = () => {
    if (!folderPath || !scanResult) return;
    createRun.mutate({
      source_folder: folderPath,
      image_count: scanResult.image_count,
      video_count: scanResult.video_count,
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Choose media folder</CardTitle>
        <CardDescription>
          Select the folder with images or videos you want to analyse.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        <FolderSelector
          value={folderPath}
          onChange={setFolderPath}
        />

        {createRun.isError && (
          <p className="text-sm text-destructive">
            Could not start the folder run:{" "}
            {createRun.error instanceof Error
              ? createRun.error.message
              : "unknown error"}
          </p>
        )}
      </CardContent>

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
    </Card>
  );
}
