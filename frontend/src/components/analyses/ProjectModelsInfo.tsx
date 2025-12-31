/**
 * Project Models Info Component
 *
 * Simplified version matching Create Project modal style.
 * - Clean readonly display with info tooltip
 * - Button to view full settings in modal
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Settings, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { projectsApi } from "@/api/projects";
import { modelsApi } from "@/api/models";
import { ProjectSettingsModal } from "./ProjectSettingsModal";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface ProjectModelsInfoProps {
  projectId: string;
}

export function ProjectModelsInfo({ projectId }: ProjectModelsInfoProps) {
  const [showModal, setShowModal] = useState(false);

  // Fetch project data
  const { data: project, isLoading: projectLoading } = useQuery({
    queryKey: ["projects", projectId],
    queryFn: () => projectsApi.get(projectId),
  });

  // Fetch detection models
  const { data: detectionModels } = useQuery({
    queryKey: ["models", "detection"],
    queryFn: () => modelsApi.listDetectionModels(),
  });

  // Fetch classification models
  const { data: classificationModels } = useQuery({
    queryKey: ["models", "classification"],
    queryFn: () => modelsApi.listClassificationModels(),
  });

  if (projectLoading || !project) {
    return (
      <div className="rounded-lg border border-gray-200 bg-gray-50 p-4">
        <div className="animate-pulse space-y-2">
          <div className="h-4 bg-gray-200 rounded w-1/3"></div>
          <div className="h-3 bg-gray-200 rounded w-1/2"></div>
        </div>
      </div>
    );
  }

  // Find model objects
  const detectionModel = detectionModels?.find(
    (m) => m.model_id === project.detection_model_id
  );
  const classificationModel = project.classification_model_id
    ? classificationModels?.find((m) => m.model_id === project.classification_model_id)
    : null;

  // Count excluded species
  const excludedCount = project.excluded_classes?.length || 0;

  return (
    <TooltipProvider>
      <div className="space-y-2">
        {/* Label with info tooltip */}
        <label className="flex items-center gap-1.5 text-sm font-medium">
          Analysis configuration
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
            </TooltipTrigger>
            <TooltipContent>
              <p className="max-w-xs">
                These settings apply to all deployments in this project. Deployments will use the models and species configured here.
              </p>
            </TooltipContent>
          </Tooltip>
        </label>

        {/* Settings summary */}
        <div className="rounded-lg border border-gray-200 bg-gray-50 p-4 space-y-3">
          {/* Detection model */}
          <div>
            <p className="text-xs text-gray-600 mb-1">Detection model</p>
            <div className="flex items-center gap-2">
              {detectionModel && <span className="text-lg">{detectionModel.emoji}</span>}
              <span className="text-sm font-medium">
                {detectionModel?.friendly_name || project.detection_model_id}
              </span>
            </div>
          </div>

          {/* Classification model */}
          <div>
            <p className="text-xs text-gray-600 mb-1">Classification model</p>
            <div className="flex items-center gap-2">
              {classificationModel && <span className="text-lg">{classificationModel.emoji}</span>}
              <span className="text-sm font-medium">
                {classificationModel?.friendly_name || project.classification_model_id || "None"}
              </span>
            </div>
          </div>

          {/* Species selection */}
          {project.classification_model_id && (
            <div>
              <p className="text-xs text-gray-600 mb-1">Species selection</p>
              <span className="text-sm">
                {excludedCount > 0 ? `${excludedCount} species excluded` : "All species"}
              </span>
            </div>
          )}

          {/* View/Edit button */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowModal(true)}
            className="w-full mt-2"
          >
            <Settings className="h-4 w-4 mr-2" />
            View settings
          </Button>
        </div>
      </div>

      {/* Settings modal */}
      <ProjectSettingsModal
        projectId={projectId}
        open={showModal}
        onOpenChange={setShowModal}
      />
    </TooltipProvider>
  );
}
