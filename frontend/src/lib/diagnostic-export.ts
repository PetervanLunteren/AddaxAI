/**
 * Building a diagnostic report, in one place.
 *
 * Two entry points want this: the native Help menu (via MenuCommands) and
 * the error block on a failed analysis, where the user is already looking
 * at something they cannot fix themselves. Both must behave identically,
 * so neither owns the logic.
 */

import { toast } from "sonner";
import { diagnosticsApi } from "../api/diagnostics";

/**
 * Build and download the diagnostic zip, surfacing a toast on failure.
 *
 * Success is deliberately silent here: in Electron the main process fires
 * its own download-complete toast (DownloadCompleteToasts in App.tsx),
 * which names the file and offers "Show in folder". A toast here would
 * duplicate it.
 */
export async function exportDiagnosticReport(): Promise<void> {
  try {
    await diagnosticsApi.downloadDiagnosticZip();
  } catch (err) {
    toast.error(
      `Could not build diagnostic report: ${(err as Error).message}`,
    );
  }
}
