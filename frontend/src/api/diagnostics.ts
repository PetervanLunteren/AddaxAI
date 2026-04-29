/**
 * Diagnostics API.
 *
 * Two endpoints:
 *   downloadDiagnosticZip()  — fetches the ZIP and triggers a browser
 *                              download. The user emails it to support.
 *   getLastLaunchStatus()    — was the previous shutdown clean? Drives
 *                              the "previous run crashed" banner.
 */

import { API_BASE_URL, api } from "../lib/api-client";

export interface LastLaunchStatus {
  previous_shutdown_clean: boolean;
  snapshot_present: boolean;
  current_launch_at?: string;
}

export const diagnosticsApi = {
  getLastLaunchStatus: () =>
    api.get<LastLaunchStatus>("/api/logs/last-launch-status"),

  /**
   * Download the diagnostic ZIP. Streams the response into a Blob and
   * triggers a browser download via an anchor click. We don't use a
   * direct <a href=...> because the backend sets Content-Disposition
   * with a timestamped filename and we want the browser to honour it.
   */
  async downloadDiagnosticZip(): Promise<void> {
    const res = await fetch(`${API_BASE_URL}/api/logs/diagnostic-zip`);
    if (!res.ok) {
      throw new Error(
        `Failed to build diagnostic report (status ${res.status})`,
      );
    }
    const blob = await res.blob();
    const filename = parseFilename(res.headers.get("content-disposition")) ||
      "addaxai-diagnostics.zip";
    triggerBrowserDownload(blob, filename);
  },
};

function parseFilename(header: string | null): string | null {
  if (!header) return null;
  const match = /filename="?([^"]+)"?/.exec(header);
  return match ? match[1] : null;
}

function triggerBrowserDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Free the blob; doing it on next tick ensures the click finished.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
