/**
 * Building a diagnostic report, in one place.
 *
 * Every entry point goes through here: the native Help menu (via
 * MenuCommands), the error blocks on a failed analysis, where the user is
 * already looking at something they cannot fix themselves, and the crash
 * banner. All must behave identically, so none owns the logic.
 */

import { toast } from "sonner";
import { diagnosticsApi } from "../api/diagnostics";

let building = false;

/**
 * Build and download the diagnostic zip.
 *
 * The backend builds the whole zip before it sends a byte, and that can
 * take long enough for the click to look dead (forum report 2026-11-11),
 * so a loading toast stays up while it works. A click during a build is
 * ignored: each build is a full backend run that ends in its own zip.
 *
 * Success only dismisses the loading toast: in Electron the main process
 * fires its own download-complete toast (DownloadCompleteToasts in
 * App.tsx), which names the file and offers "Show in folder". A success
 * toast here would duplicate it.
 */
export async function exportDiagnosticReport(): Promise<void> {
  if (building) return;
  building = true;
  const id = toast.loading("Building the diagnostic report…");
  try {
    await diagnosticsApi.downloadDiagnosticZip();
    toast.dismiss(id);
  } catch (err) {
    toast.error(
      `Could not build diagnostic report: ${(err as Error).message}`,
      { id },
    );
  } finally {
    building = false;
  }
}
