/**
 * The licence step before a model that runs on ultralytics is set up.
 * Asked once per installation: the backend refuses to prepare such a model
 * until it is accepted, and `useModelPrepare` opens this on that refusal.
 *
 * A plain confirm, like SignOffDialog: it opens with Cancel focused, so an
 * Enter the user was already leaning on does not accept. The list of models
 * comes from the catalog (env "ultralytics"), never from a list of ids
 * here, so it stays true when the catalog changes.
 */

import { useQuery } from "@tanstack/react-query";
import { ExternalLink, Loader2 } from "lucide-react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "../ui/alert-dialog";
import { modelsApi } from "../../api/models";
import { AGPL_LICENCE_URL, runsOnUltralytics } from "../../lib/agpl";

function listNames(names: string[]): string {
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

export function AgplLicenceDialog({
  modelId,
  accepting,
  saveFailed,
  onAccept,
  onCancel,
}: {
  /** The model waiting for the licence step, or null when closed. */
  modelId: string | null;
  accepting: boolean;
  saveFailed: boolean;
  onAccept: () => void;
  onCancel: () => void;
}) {
  // The same keys the model sheet uses, so these are usually cached.
  const { data: detection = [] } = useQuery({
    queryKey: ["models", "detection"],
    queryFn: modelsApi.listDetectionModels,
    enabled: modelId !== null,
  });
  const { data: classification = [] } = useQuery({
    queryKey: ["models", "classification"],
    queryFn: modelsApi.listClassificationModels,
    enabled: modelId !== null,
  });
  const affected = [...detection, ...classification].filter(runsOnUltralytics);
  const name = affected.find((m) => m.model_id === modelId)?.friendly_name ?? modelId;
  const others = affected.filter((m) => m.model_id !== modelId).map((m) => m.friendly_name);

  return (
    <AlertDialog open={modelId !== null} onOpenChange={(open) => !open && onCancel()}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>This model uses software with its own licence</AlertDialogTitle>
          <AlertDialogDescription asChild>
            <div className="space-y-2">
              <p>
                You can use {name} to analyse your own images and videos and
                use the results, for research, commercial or personal work.
              </p>
              <p>
                You cannot build it into an app, website or service that you
                give or sell to others, unless you also share that software&apos;s source code
                under the same licence (AGPL-3.0).
              </p>
              <p>
                {others.length > 0 && <>The same applies to {listNames(others)}. </>}
                You only see this once.{" "}
                <a
                  href={AGPL_LICENCE_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline hover:opacity-80 inline-flex items-center gap-1"
                >
                  Read the licence
                  <ExternalLink className="h-3 w-3" />
                </a>
              </p>
              {saveFailed && (
                <p style={{ color: "#882000" }}>Could not save that choice. Try again.</p>
              )}
            </div>
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={accepting}>Cancel</AlertDialogCancel>
          <AlertDialogAction
            disabled={accepting}
            onClick={(event) => {
              // Stay open while the choice is saved, and on a failure.
              event.preventDefault();
              onAccept();
            }}
          >
            {accepting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Saving...
              </>
            ) : (
              "I understand"
            )}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
