/**
 * Model Info Sheet Component
 *
 * Displays detailed information about a detection, classification, or embedding
 * model in a slide-out drawer from the right.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ExternalLink } from "lucide-react";
import { modelsApi } from "@/api/models";
import { api } from "@/lib/api-client";
import { formatVersion, satisfiesMinVersion } from "@/lib/version";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Separator } from "@/components/ui/separator";

interface ModelInfoSheetProps {
  modelId: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ModelInfoSheet({ modelId, open, onOpenChange }: ModelInfoSheetProps) {
  const [exampleImageFailed, setExampleImageFailed] = useState(false);
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

  // Fetch all embedding models
  const { data: embeddingModels } = useQuery({
    queryKey: ["models", "embedding"],
    queryFn: () => modelsApi.listEmbeddingModels(),
    enabled: open && !!modelId,
  });

  // Fetch taxonomy to get class count
  const { data: taxonomy } = useQuery({
    queryKey: ["taxonomy", modelId],
    queryFn: () => modelsApi.getTaxonomy(modelId!),
    enabled: open && !!modelId && modelId !== "none",
  });

  // Running app version, served by /health. Backed by the repo-root
  // VERSION file (see backend/app/__init__.py). Cached for the session
  // since the value can't change without a backend restart.
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: () => api.get<{ version: string }>("/health"),
    staleTime: Infinity,
  });
  const currentVersion = health?.version ?? null;

  // Find the selected model
  const model = [...(classificationModels || []), ...(detectionModels || []), ...(embeddingModels || [])].find(
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

  // Format the classes with sentence case (only the first letter capitalized).
  // Long lists (SpeciesNet has 2000+) are truncated to keep the panel readable;
  // the full count stays in the heading.
  const MAX_CLASSES_SHOWN = 100;
  const classNames = classList.map(formatClassName);
  const visibleClassList = classNames.slice(0, MAX_CLASSES_SHOWN).join(", ");
  const sentenceCased = visibleClassList.length > 0
    ? visibleClassList.charAt(0).toUpperCase() + visibleClassList.slice(1)
    : "";
  const remainingClasses = classNames.length - Math.min(classNames.length, MAX_CLASSES_SHOWN);
  const formattedClasses = sentenceCased
    ? remainingClasses > 0
      ? `${sentenceCased} … +${remainingClasses} more`
      : `${sentenceCased}.`
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
            {model.type === "detection" ? "Detection model" : model.type === "embedding" ? "Embedding model" : "Classification model"}
          </SheetDescription>
        </SheetHeader>

        <div className="mt-6 space-y-6">
          {/* Description */}
          <div>
            <h3 className="text-sm font-semibold mb-2">Description</h3>
            <p className="text-sm text-gray-700 leading-relaxed">{model.description}</p>
          </div>

          {/* Example picture: what the model expects to see. A URL from the
              manifest, so it needs the network; when it does not load the
              block disappears rather than showing a broken image. */}
          {model.example_image_url && !exampleImageFailed && (
            <div>
              <h3 className="text-sm font-semibold mb-2">Example image</h3>
              <img
                src={model.example_image_url}
                alt={`Example image for ${model.friendly_name}`}
                className="max-w-full rounded-md border"
                onError={() => setExampleImageFailed(true)}
              />
              <p className="mt-1 text-xs text-muted-foreground">
                The kind of photo this model was trained on. Compare it with
                your own.
              </p>
            </div>
          )}

          <Separator />

          {/* Embedding specs (for embedding models) */}
          {model.type === "embedding" && model.embedding_dim && (
            <>
              <div>
                <h3 className="text-sm font-semibold mb-2">Specifications</h3>
                <p className="text-sm text-gray-700">
                  {model.embedding_dim}-dimensional feature vectors
                </p>
              </div>
              <Separator />
            </>
          )}

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
          {model.min_app_version && currentVersion && (
            <div>
              <h3 className="text-sm font-semibold mb-2">Version requirement</h3>
              <p className="text-sm text-gray-700">
                {(() => {
                  // Numeric comparison, not string comparison: "7.0.10"
                  // sorts below "7.0.9" character by character, which
                  // would tell users on a new enough build to update.
                  // An unparseable version counts as not meeting the
                  // requirement, so we never claim a version is fine
                  // when we could not actually check it.
                  const meetsRequirement =
                    satisfiesMinVersion(
                      currentVersion,
                      model.min_app_version
                    ) === true;
                  return (
                    <>
                      Minimum AddaxAI version required is {formatVersion(model.min_app_version)}, while your current version is {formatVersion(currentVersion)}.{" "}
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
