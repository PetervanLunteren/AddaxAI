/**
 * Model Info Sheet Component
 *
 * Displays detailed information about a classification or detection model
 * in a slide-out drawer from the right.
 */

import { useQuery } from "@tanstack/react-query";
import { ExternalLink, X } from "lucide-react";
import { modelsApi } from "@/api/models";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import type { ModelInfo } from "@/api/types";

interface ModelInfoSheetProps {
  modelId: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ModelInfoSheet({ modelId, open, onOpenChange }: ModelInfoSheetProps) {
  // Fetch all classification models to find the selected one
  const { data: classificationModels } = useQuery({
    queryKey: ["models", "classification"],
    queryFn: () => modelsApi.listClassificationModels(),
    enabled: open && !!modelId,
  });

  // Fetch all detection models
  const { data: detectionModels } = useQuery({
    queryKey: ["models", "detection"],
    queryFn: () => modelsApi.listDetectionModels(),
    enabled: open && !!modelId,
  });

  // Fetch taxonomy to get class count
  const { data: taxonomy } = useQuery({
    queryKey: ["taxonomy", modelId],
    queryFn: () => modelsApi.getTaxonomy(modelId!),
    enabled: open && !!modelId && modelId !== "none",
  });

  // Find the selected model
  const model = [...(classificationModels || []), ...(detectionModels || [])].find(
    (m) => m.model_id === modelId
  );

  if (!model) return null;

  // Format classes list
  const classList = taxonomy?.all_classes || [];

  // Normalize class names: remove underscores, all lowercase
  const formatClassName = (className: string) => {
    // Replace underscores with spaces and make lowercase
    return className.replace(/_/g, " ").toLowerCase();
  };

  // Format all classes with sentence case (only first letter of entire list capitalized)
  const formattedClassList = classList.map(formatClassName).join(", ");
  const formattedClasses = formattedClassList.length > 0
    ? formattedClassList.charAt(0).toUpperCase() + formattedClassList.slice(1) + "."
    : "";

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full sm:max-w-xl overflow-y-auto">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2 text-xl">
            <span className="text-2xl">{model.emoji}</span>
            {model.friendly_name}
          </SheetTitle>
          <SheetDescription>
            {model.type === "detection" ? "Detection model" : "Classification model"}
          </SheetDescription>
        </SheetHeader>

        <div className="mt-6 space-y-6">
          {/* Description */}
          <div>
            <h3 className="text-sm font-semibold mb-2">Description</h3>
            <p className="text-sm text-gray-700 leading-relaxed">{model.description}</p>
          </div>

          <Separator />

          {/* Classes (for classification models) */}
          {model.type === "classification" && classList.length > 0 && (
            <>
              <div>
                <h3 className="text-sm font-semibold mb-2">
                  Classes ({classList.length})
                </h3>
                <p className="text-sm text-gray-700">{formattedClasses}</p>
              </div>
              <Separator />
            </>
          )}

          {/* Developer */}
          {model.developer && (
            <>
              <div>
                <h3 className="text-sm font-semibold mb-2">Developer</h3>
                <p className="text-sm text-gray-700">{model.developer}</p>
              </div>
              <Separator />
            </>
          )}

          {/* Owner (if different from developer) */}
          {model.owner && (
            <>
              <div>
                <h3 className="text-sm font-semibold mb-2">Owner</h3>
                <p className="text-sm text-gray-700">{model.owner}</p>
              </div>
              <Separator />
            </>
          )}

          {/* More Information */}
          {model.info_url && (
            <>
              <div>
                <h3 className="text-sm font-semibold mb-2">More information</h3>
                <a
                  href={model.info_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-primary hover:opacity-80 underline flex items-center gap-1"
                >
                  {model.info_url}
                  <ExternalLink className="h-3 w-3" />
                </a>
              </div>
              <Separator />
            </>
          )}

          {/* Citation */}
          {model.citation && (
            <>
              <div>
                <h3 className="text-sm font-semibold mb-2">Citation</h3>
                <a
                  href={model.citation}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-primary hover:opacity-80 underline flex items-center gap-1"
                >
                  {model.citation}
                  <ExternalLink className="h-3 w-3" />
                </a>
              </div>
              <Separator />
            </>
          )}

          {/* License */}
          {model.license && (
            <>
              <div>
                <h3 className="text-sm font-semibold mb-2">License</h3>
                <a
                  href={model.license}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-primary hover:opacity-80 underline flex items-center gap-1"
                >
                  {model.license}
                  <ExternalLink className="h-3 w-3" />
                </a>
              </div>
              <Separator />
            </>
          )}

          {/* Version Requirement */}
          {model.min_app_version && (
            <div>
              <h3 className="text-sm font-semibold mb-2">Version requirement</h3>
              <p className="text-sm text-gray-700">
                {(() => {
                  const currentVersion = "0.1.0"; // TODO: Get from API
                  const meetsRequirement = currentVersion >= model.min_app_version;
                  return (
                    <>
                      Minimum AddaxAI version required is v{model.min_app_version}, while your current version is v{currentVersion}.{" "}
                      {meetsRequirement ? (
                        <span>You're good to go.</span>
                      ) : (
                        <>
                          Please{" "}
                          <a
                            href="https://addaxdatascience.com/addaxai/#install"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-primary hover:opacity-80 underline inline-flex items-center gap-1"
                          >
                            update AddaxAI
                            <ExternalLink className="h-3 w-3" />
                          </a>
                          .
                        </>
                      )}
                    </>
                  );
                })()}
              </p>
            </div>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
