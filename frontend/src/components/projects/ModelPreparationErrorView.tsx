/**
 * Model Preparation Error View Component
 *
 * Shows when model preparation fails.
 * Offers retry and cancel options.
 */

import { AlertTriangle, RotateCw, ArrowLeft } from "lucide-react";
import { Button } from "../ui/button";
import { DialogDescription, DialogHeader, DialogTitle } from "../ui/dialog";
import { ContinueWithoutRevocationChecks } from "../setup/ContinueWithoutRevocationChecks";

interface ModelPreparationErrorViewProps {
  errorMessage: string;
  onRetry: () => void;
  onCancel: () => void;
  /** From the failed task. Only "tls_revocation" today, which unlocks
   * the extra choice below the common-causes list. */
  errorKind?: string;
}

export function ModelPreparationErrorView({
  errorMessage,
  onRetry,
  onCancel,
  errorKind,
}: ModelPreparationErrorViewProps) {
  const isRevocationFailure = errorKind === "tls_revocation";

  return (
    <>
      <DialogHeader>
        <DialogTitle>Preparation failed</DialogTitle>
        <DialogDescription>An error occurred while preparing the model</DialogDescription>
      </DialogHeader>

      <div className="py-6 space-y-6">
        {/* Error Icon */}
        <div className="flex flex-col items-center gap-4 text-center">
          <div className="rounded-full bg-red-50 p-4">
            <AlertTriangle className="h-8 w-8 text-red-600" />
          </div>

          {/* Error Message */}
          <div className="space-y-2">
            <h3 className="font-semibold text-red-900">Preparation failed</h3>
            <p className="text-sm text-muted-foreground max-w-md">{errorMessage}</p>
          </div>
        </div>

        {/* Troubleshooting Tips. Suppressed for a cause we can name: a
            guess list under a precise diagnosis only muddies it. */}
        {!isRevocationFailure && (
          <div className="bg-muted/50 rounded-md p-4 text-sm space-y-2">
            <p className="font-medium">Common causes:</p>
            <ul className="list-disc list-inside space-y-1 text-muted-foreground">
              <li>Network connection interrupted</li>
              <li>Insufficient disk space</li>
              <li>Download server temporarily unavailable</li>
            </ul>
          </div>
        )}

        {isRevocationFailure && (
          <ContinueWithoutRevocationChecks onRetry={onRetry} />
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex justify-between gap-3">
        <Button type="button" variant="outline" onClick={onCancel} className="gap-2">
          <ArrowLeft className="h-4 w-4" />
          Back to form
        </Button>
        <Button type="button" onClick={onRetry} className="gap-2">
          <RotateCw className="h-4 w-4" />
          Retry preparation
        </Button>
      </div>
    </>
  );
}
