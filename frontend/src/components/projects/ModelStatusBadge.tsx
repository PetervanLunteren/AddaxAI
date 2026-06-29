/**
 * Model Status Badge Component
 *
 * Shows preparation status for a classification model
 * with action button if preparation is needed.
 */

import { Download } from "lucide-react";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
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
    <Callout
      variant="warning"
      title="Setup required"
      action={
        <Button
          type="button"
          onClick={onPrepare}
          disabled={isPreparing}
          size="sm"
          className="gap-2"
        >
          <Download className="h-3.5 w-3.5" />
          {isPreparing ? "Preparing..." : "Start setup"}
        </Button>
      }
    >
      <p>This model needs a one-time setup before it can be used.</p>
      <p>We'll download the model and prepare the environment.</p>
      <p className="mt-1 text-xs opacity-80">This may take a few minutes.</p>
    </Callout>
  );
}
