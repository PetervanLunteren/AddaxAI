/**
 * TypeScript declarations for Electron APIs exposed via preload script
 */

export interface ElectronAPI {
  /**
   * Host OS as reported by Node's `process.platform`. Synchronous —
   * set at preload time. Renderer-side platform gates read this without
   * an IPC round-trip.
   */
  platform: NodeJS.Platform;
  selectFolder: () => Promise<string | null>;
  /**
   * Open a single-file picker. `filters` is the standard Electron
   * file-filter shape; the dialog rejects extensions not listed.
   * Returns the chosen absolute path, or null when the user cancels.
   */
  openFile: (opts?: {
    title?: string;
    filters?: { name: string; extensions: string[] }[];
  }) => Promise<string | null>;
  showItemInFolder: (filePath: string) => Promise<void>;
  /**
   * Open a file or directory with the OS default handler. For directories
   * this opens the folder in the system file manager. Returns an error
   * string on failure, empty string on success.
   */
  openPath: (targetPath: string) => Promise<string>;
  /**
   * Quit the app cleanly. Used by the Reset flow after wipe completes
   * so the next launch starts from a fresh state.
   */
  quitApp: () => Promise<void>;
  /**
   * Relaunch the app: exit and immediately start a fresh process.
   * Used by Restore-from-backup so the user does not have to reopen
   * the app manually after the DB swap.
   */
  relaunchApp: () => Promise<void>;
  /**
   * Runtime app version, e.g. "0.2.0-beta.1". Stamped into the bundle
   * by the release workflow.
   */
  getVersion: () => Promise<string>;
  isElectron: () => boolean;
  /**
   * Subscribe to download completions (files the main process auto-saved
   * to the Downloads folder). Returns an unsubscribe function.
   */
  onDownloadComplete: (
    callback: (info: {
      filename: string;
      path: string;
      success: boolean;
    }) => void,
  ) => () => void;
  /**
   * Resolve the absolute filesystem path of a `File` produced by a
   * drag-and-drop event. Wraps Electron's `webUtils.getPathForFile()`.
   */
  getDroppedFolderPath: (file: File) => string;
}

declare global {
  interface Window {
    electronAPI?: ElectronAPI;
  }
}

export {};
