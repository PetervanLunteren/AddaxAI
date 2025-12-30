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
   * Check if running in Electron (vs browser)
   */
  isElectron: (): boolean => {
    return true;
  },
});
