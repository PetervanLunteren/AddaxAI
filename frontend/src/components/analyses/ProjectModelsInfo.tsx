/**
 * Project Models Info Component
 *
 * Displays readonly information about project's model configuration.
 * Shows detection model, classification model, and species selection.
 * Links to project settings for editing.
 */

import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Settings, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { projectsApi } from "@/api/projects";
import { mlModelsApi } from "@/api/ml-models";

interface ProjectModelsInfoProps {
  projectId: string;
}

export function ProjectModelsInfo({ projectId }: ProjectModelsInfoProps) {
  const navigate = useNavigate();

  // Fetch project data
  const { data: project, isLoading: projectLoading } = useQuery({
    queryKey: ["projects", projectId],
    queryFn: () => projectsApi.get(projectId),
  });

  // Fetch detection models
  const { data: detectionModels } = useQuery({
    queryKey: ["ml-models", "detection"],
    queryFn: () => mlModelsApi.listDetection(),
  });

  // Fetch classification models
  const { data: classificationModels } = useQuery({
    queryKey: ["ml-models", "classification"],
    queryFn: () => mlModelsApi.listClassification(),
  });

  if (projectLoading || !project) {
    return (
      <div className="rounded-lg border border-gray-200 bg-gray-50 p-4">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/3 mb-2"></div>
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

  // Count excluded species (inverse of selected)
  const excludedCount = project.excluded_classes?.length || 0;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">Analysis configuration</label>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigate(`/projects/${projectId}/settings`)}
        >
          <Settings className="h-3 w-3 mr-1" />
          Edit
        </Button>
      </div>

      <div className="rounded-lg border border-gray-200 bg-gray-50 p-4 space-y-3">
        {/* Detection model */}
        <div>
          <p className="text-xs text-gray-600 mb-1">Detection model</p>
          <div className="flex items-center gap-2">
            {detectionModel ? (
              <>
                <span className="text-lg">{detectionModel.emoji}</span>
                <span className="text-sm font-medium">{detectionModel.friendly_name}</span>
              </>
            ) : (
              <span className="text-sm text-gray-500">{project.detection_model_id}</span>
            )}
          </div>
        </div>

        {/* Classification model */}
        <div>
          <p className="text-xs text-gray-600 mb-1">Classification model</p>
          <div className="flex items-center gap-2">
            {classificationModel ? (
              <>
                <span className="text-lg">{classificationModel.emoji}</span>
                <span className="text-sm font-medium">{classificationModel.friendly_name}</span>
              </>
            ) : (
              <span className="text-sm text-gray-500">
                {project.classification_model_id || "None"}
              </span>
            )}
          </div>
        </div>

        {/* Species selection */}
        <div>
          <p className="text-xs text-gray-600 mb-1">Species selection</p>
          <div className="flex items-center gap-2">
            <span className="text-sm">
              {excludedCount > 0
                ? `${excludedCount} species excluded`
                : "All species included"}
            </span>
          </div>
        </div>

        {/* Info note */}
        <div className="flex items-start gap-2 pt-2 border-t border-gray-200">
          <Info className="h-4 w-4 text-blue-600 mt-0.5 shrink-0" />
          <p className="text-xs text-gray-600">
            These settings apply to all deployments in this project. Deployments will use the
            models and species configured here.
          </p>
        </div>
      </div>
    </div>
  );
}
