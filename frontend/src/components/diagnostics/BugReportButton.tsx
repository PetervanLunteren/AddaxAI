/**
 * Inline bug-report button.
 *
 * Same icon-button shape as the floating one in AppHamburger, but
 * rendered in the normal flex flow next to a page's action button so
 * flexbox handles alignment for free. Use this on project pages whose
 * headers already have a right-aligned action button (Sites,
 * Deployments) to avoid the floating button overlapping them on
 * narrow viewports.
 */

import { useMutation } from "@tanstack/react-query";
import { Bug } from "lucide-react";
import { toast } from "sonner";
import { diagnosticsApi } from "../../api/diagnostics";
import { Button } from "../ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";

export function BugReportButton() {
  const download = useMutation({
    mutationFn: () => diagnosticsApi.downloadDiagnosticZip(),
    onSuccess: () => {
      toast.success("Diagnostic report saved to Downloads");
    },
    onError: (err: Error) => {
      toast.error(`Could not build diagnostic report: ${err.message}`);
    },
  });

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            type="button"
            variant="outline"
            size="icon"
            onClick={() => download.mutate()}
            disabled={download.isPending}
            aria-label="Export bug report"
          >
            <Bug className="h-4 w-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Export bug report</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
