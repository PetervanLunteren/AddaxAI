/**
 * Electron preload script
 *
 * Exposes limited Electron APIs to the renderer process via contextBridge.
 * Following security best practices: no nodeIntegration, contextIsolation enabled.
 */

import { contextBridge, ipcRenderer } from 'electron';

// Expose safe Electron APIs to renderer
contextBridge.exposeInMainWorld('electronAPI', {
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
   * Check if running in Electron (vs browser)
   */
  isElectron: (): boolean => {
    return true;
  },
});
