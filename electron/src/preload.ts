/**
 * Electron preload script
 *
 * Exposes limited Electron APIs to the renderer process via contextBridge.
 * Following security best practices: no nodeIntegration, contextIsolation enabled.
 */

import { contextBridge, ipcRenderer, webUtils } from 'electron';

// Expose safe Electron APIs to renderer
contextBridge.exposeInMainWorld('electronAPI', {
  /**
   * Synchronous OS identifier ('win32', 'darwin', 'linux'). Used by the
   * renderer to gate Windows-only features (e.g. Timelapse integration) without
   * an IPC round-trip on every render. Set at preload time, never changes.
   */
  platform: process.platform,

  /**
   * Open native folder picker dialog
   * @returns Selected folder path, or null if cancelled
   */
  selectFolder: async (): Promise<string | null> => {
    return await ipcRenderer.invoke('dialog:selectFolder');
  },

  /**
   * Reveal a file in the native file explorer (Finder / Explorer)
   */
  showItemInFolder: async (filePath: string): Promise<void> => {
    return await ipcRenderer.invoke('shell:showItemInFolder', filePath);
  },

  /**
   * Open a file or directory with the OS default handler.
   * For directories this opens the folder in the system file manager.
   * Returns an error string on failure, empty string on success.
   */
  openPath: async (targetPath: string): Promise<string> => {
    return await ipcRenderer.invoke('shell:openPath', targetPath);
  },

  /**
   * Quit the app cleanly. Used by the Reset flow after the backend
   * wipes user data so the next launch starts fresh.
   */
  quitApp: async (): Promise<void> => {
    return await ipcRenderer.invoke('app:quit');
  },

  /**
   * Return the runtime app version (e.g. "0.2.0-beta.1"). Comes from
   * electron/package.json, which the release workflow rewrites from
   * the git tag at build time.
   */
  getVersion: async (): Promise<string> => {
    return await ipcRenderer.invoke('app:getVersion');
  },

  /**
   * Open the Timelapse Analyser integration in a separate BrowserWindow.
   * `prefilledPath` is optional; when present the form starts with that
   * folder selected (used by the `--timelapse <folder>` CLI launcher).
   */
  openTimelapseWindow: async (prefilledPath?: string): Promise<void> => {
    return await ipcRenderer.invoke('window:openTimelapse', prefilledPath);
  },

  /**
   * Check if running in Electron (vs browser)
   */
  isElectron: (): boolean => {
    return true;
  },

  /**
   * Resolve the absolute filesystem path of a `File` from a drag-and-drop
   * event. Electron 32+ removed the legacy `File.path` property, so the
   * renderer has to call `webUtils.getPathForFile()` for dropped folders
   * and files. Synchronous and side-effect free.
   */
  getDroppedFolderPath: (file: File): string => {
    return webUtils.getPathForFile(file);
  },
});
