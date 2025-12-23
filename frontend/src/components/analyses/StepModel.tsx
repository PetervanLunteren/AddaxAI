/**
 * Step 3: Model - Detection and classification model selection with modern styling.
 */

import React from "react";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { useWizard } from "@/components/wizard/WizardContext";
import { Brain, Info, CheckCircle2, AlertCircle } from "lucide-react";

interface StepModelProps {
  detectionModelId: string | null;
  classificationModelId: string | null;
  onDetectionModelChange: (modelId: string | null) => void;
  onClassificationModelChange: (modelId: string | null) => void;
}

// TODO: Fetch from API
const DETECTION_MODELS = [
  { id: "megadetector_v5a", name: "MegaDetector v5a", emoji: "🦁", status: "ready", description: "General wildlife detector" },
];

const CLASSIFICATION_MODELS = [
  { id: "EUR-DF-v1-3", name: "European Deciduous Forest v1.3", emoji: "🦌", status: "ready", description: "European species" },
  { id: "AFR-BASIC-v1", name: "African Basic v1", emoji: "🦒", status: "not_ready", description: "African species" },
];

export function StepModel({
  detectionModelId,
  classificationModelId,
  onDetectionModelChange,
  onClassificationModelChange,
}: StepModelProps) {
  const { setCanGoNext } = useWizard();

  // Check if selected models are ready
  const detectionModel = DETECTION_MODELS.find((m) => m.id === detectionModelId);
  const classificationModel = CLASSIFICATION_MODELS.find((m) => m.id === classificationModelId);

  const allModelsReady =
    (!detectionModelId || detectionModel?.status === "ready") &&
    (!classificationModelId || classificationModel?.status === "ready");

  // Update navigation based on model status
  React.useEffect(() => {
    setCanGoNext(allModelsReady && detectionModelId !== null);
  }, [allModelsReady, detectionModelId, setCanGoNext]);

  const handleShowInfo = (modelId: string) => {
    // TODO: Open modal with model details
    console.log("Show info for model:", modelId);
  };

  return (
    <div className="space-y-8">
      {/* Detection Model */}
      <div>
        <Label htmlFor="detectionModel" className="text-lg font-semibold text-gray-900 flex items-center gap-2">
          <Brain className="w-5 h-5 text-blue-600" />
          Animal Detection Model
        </Label>
        <p className="text-sm text-gray-600 mt-2 mb-4">
          Detects animals, people, and vehicles in images. Required for all analyses.
        </p>

        <div className="space-y-3">
          <Select value={detectionModelId || undefined} onValueChange={onDetectionModelChange}>
            <SelectTrigger className="w-full h-12 text-base">
              <SelectValue placeholder="Select detection model" />
            </SelectTrigger>
            <SelectContent>
              {DETECTION_MODELS.map((model) => (
                <SelectItem key={model.id} value={model.id} className="text-base py-3">
                  <div className="flex items-center justify-between w-full">
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{model.emoji}</span>
                      <span>{model.name}</span>
                    </div>
                    {model.status === "ready" ? (
                      <CheckCircle2 className="w-4 h-4 text-green-600" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-orange-600" />
                    )}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {detectionModelId && detectionModel && (
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg animate-in fade-in slide-in-from-top-2 duration-300">
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-3 flex-1">
                  <span className="text-2xl">{detectionModel.emoji}</span>
                  <div>
                    <p className="text-sm font-medium text-blue-900">{detectionModel.name}</p>
                    <p className="text-xs text-blue-700 mt-1">{detectionModel.description}</p>
                    <div className="flex items-center gap-1 mt-2">
                      {detectionModel.status === "ready" ? (
                        <>
                          <CheckCircle2 className="w-3.5 h-3.5 text-green-600" />
                          <span className="text-xs text-green-700 font-medium">Ready to use</span>
                        </>
                      ) : (
                        <>
                          <AlertCircle className="w-3.5 h-3.5 text-orange-600" />
                          <span className="text-xs text-orange-700 font-medium">Download required</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => handleShowInfo(detectionModelId)}
                  className="shrink-0"
                >
                  <Info className="w-4 h-4" />
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Classification Model */}
      <div>
        <Label htmlFor="classificationModel" className="text-lg font-semibold text-gray-900 flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-600" />
          Species Identification Model
        </Label>
        <p className="text-sm text-gray-600 mt-2 mb-4">
          Identifies specific species from detected animals. Optional but recommended.
        </p>

        <div className="space-y-3">
          <Select
            value={classificationModelId || undefined}
            onValueChange={onClassificationModelChange}
          >
            <SelectTrigger className="w-full h-12 text-base">
              <SelectValue placeholder="Select classification model (optional)" />
            </SelectTrigger>
            <SelectContent>
              {CLASSIFICATION_MODELS.map((model) => (
                <SelectItem key={model.id} value={model.id} className="text-base py-3">
                  <div className="flex items-center justify-between w-full">
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{model.emoji}</span>
                      <span>{model.name}</span>
                    </div>
                    {model.status === "ready" ? (
                      <CheckCircle2 className="w-4 h-4 text-green-600" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-orange-600" />
                    )}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {classificationModelId && classificationModel && (
            <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg animate-in fade-in slide-in-from-top-2 duration-300">
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-3 flex-1">
                  <span className="text-2xl">{classificationModel.emoji}</span>
                  <div>
                    <p className="text-sm font-medium text-purple-900">{classificationModel.name}</p>
                    <p className="text-xs text-purple-700 mt-1">{classificationModel.description}</p>
                    <div className="flex items-center gap-1 mt-2">
                      {classificationModel.status === "ready" ? (
                        <>
                          <CheckCircle2 className="w-3.5 h-3.5 text-green-600" />
                          <span className="text-xs text-green-700 font-medium">Ready to use</span>
                        </>
                      ) : (
                        <>
                          <AlertCircle className="w-3.5 h-3.5 text-orange-600" />
                          <span className="text-xs text-orange-700 font-medium">Download required</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => handleShowInfo(classificationModelId)}
                  className="shrink-0"
                >
                  <Info className="w-4 h-4" />
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Warning if models not ready */}
      {!allModelsReady && (
        <div className="p-4 bg-orange-50 border border-orange-200 rounded-lg animate-in fade-in slide-in-from-top-2 duration-300">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-orange-600 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-orange-900">Models not ready</p>
              <p className="text-sm text-orange-700 mt-1">
                Please download and prepare selected models before proceeding. You can do this in the Models page.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
