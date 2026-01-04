/**
 * Model Status Badge Component
 *
 * Shows preparation status for a classification model
 * with action button if preparation is needed.
 */

import { AlertTriangle, Download } from "lucide-react";
import { Button } from "../ui/button";
import { Alert, AlertDescription, AlertTitle } from "../ui/alert";
import type { ModelStatusResponse } from "../../api/types";

interface ModelStatusBadgeProps {
  status: ModelStatusResponse;
  onPrepare: () => void;
  isPreparing: boolean;
}

export function ModelStatusBadge({ status, onPrepare, isPreparing }: ModelStatusBadgeProps) {
  // Model is ready - don't show any badge (button state is sufficient indicator)
  if (status.status === "ready") {
    return null;
  }

  return (
    <Alert variant="default" className="border-yellow-200 bg-yellow-50">
      <AlertTriangle className="h-4 w-4 text-yellow-700" />
      <AlertTitle className="text-yellow-900">Setup required</AlertTitle>
      <AlertDescription className="space-y-3 mt-2">
        <div className="text-sm text-yellow-800">
          <p>This model needs a one-time setup before it can be used.</p>
          <p>We'll download the model and prepare the environment.</p>
          <p className="text-xs text-yellow-600 mt-1">This may take a few minutes.</p>
        </div>

        {/* Action button */}
        <div className="flex justify-end">
          <Button onClick={onPrepare} disabled={isPreparing} size="sm" className="gap-2">
            <Download className="h-3.5 w-3.5" />
            {isPreparing ? "Preparing..." : "Start setup"}
          </Button>
        </div>
      </AlertDescription>
    </Alert>
  );
}
