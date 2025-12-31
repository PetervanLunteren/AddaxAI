/**
 * Project Settings Modal
 *
 * Displays readonly project settings (detection model, classification model, species selection).
 * Simple design matching Create Project modal style.
 */

import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Settings } from "lucide-react";
import { projectsApi } from "@/api/projects";
import { modelsApi } from "@/api/models";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

interface ProjectSettingsModalProps {
  projectId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ProjectSettingsModal({
  projectId,
  open,
  onOpenChange,
}: ProjectSettingsModalProps) {
  const navigate = useNavigate();

  // Fetch project data
  const { data: project } = useQuery({
    queryKey: ["projects", projectId],
    queryFn: () => projectsApi.get(projectId),
    enabled: open,
  });

  // Fetch models for display names
  const { data: detectionModels = [] } = useQuery({
    queryKey: ["models", "detection"],
    queryFn: () => modelsApi.listDetectionModels(),
    enabled: open,
  });

  const { data: classificationModels = [] } = useQuery({
    queryKey: ["models", "classification"],
    queryFn: () => modelsApi.listClassificationModels(),
    enabled: open,
  });

  if (!project) {
    return null;
  }

  // Find model names
  const detectionModel = detectionModels.find(
    (m) => m.model_id === project.detection_model_id
  );
  const classificationModel = classificationModels.find(
    (m) => m.model_id === project.classification_model_id
  );

  // Species selection summary
  const totalClasses = classificationModel?.description?.match(/\d+/)?.[0] || "?";
  const excludedCount = project.excluded_classes?.length || 0;
  const selectedCount = excludedCount > 0
    ? `${Number(totalClasses) - excludedCount} species`
    : "All species";

  const handleEdit = () => {
    onOpenChange(false);
    navigate(`/projects/${projectId}/settings`);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Project settings</DialogTitle>
          <DialogDescription>
            Current analysis models and species configuration
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Detection model */}
          <div>
            <p className="text-sm font-medium text-gray-700 mb-1">Detection model</p>
            <p className="text-sm text-gray-900">
              {detectionModel?.emoji} {detectionModel?.friendly_name || project.detection_model_id}
            </p>
          </div>

          {/* Classification model */}
          <div>
            <p className="text-sm font-medium text-gray-700 mb-1">Classification model</p>
            <p className="text-sm text-gray-900">
              {classificationModel?.emoji}{" "}
              {classificationModel?.friendly_name || project.classification_model_id || "None"}
            </p>
          </div>

          {/* Species selection */}
          {project.classification_model_id && (
            <div>
              <p className="text-sm font-medium text-gray-700 mb-1">Species selection</p>
              <p className="text-sm text-gray-900">{selectedCount}</p>
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Close
          </Button>
          <Button onClick={handleEdit}>
            <Settings className="h-4 w-4 mr-2" />
            Edit settings
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
