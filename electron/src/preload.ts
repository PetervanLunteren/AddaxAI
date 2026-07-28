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
   * renderer to gate platform-specific behavior without an IPC round-trip
   * on every render. Set at preload time, never changes.
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
   * Open native single-file picker dialog. Optional `filters` constrain
   * the picker by extension (e.g. `.db` for Restore-from-backup).
   * Returns the selected absolute path, or null when the user cancels.
   */
  openFile: async (opts?: {
    title?: string;
    filters?: { name: string; extensions: string[] }[];
    defaultPath?: string;
  }): Promise<string | null> => {
    return await ipcRenderer.invoke('dialog:openFile', opts);
  },

  /**
   * Pick a backup and schedule it as the next-launch restore, then
   * relaunch. Only used by the startup error page: when the backend
   * refuses a database the in-app restore dialog never loads, so this
   * is the way back. The backend validates the file when it consumes
   * the marker on the next launch.
   */
  restoreDatabase: async (): Promise<void> => {
    return await ipcRenderer.invoke('db:restore');
  },

  /**
   * Schedule the database for deletion on the next launch, then
   * relaunch. The confirm dialog is native because the app's own
   * type-RESET dialog is part of the frontend, which is not running
   * when this is needed. The backend snapshots the database before it
   * deletes it.
   */
  resetDatabase: async (): Promise<void> => {
    return await ipcRenderer.invoke('db:reset');
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
   * Relaunch the app: schedule a fresh start, then exit. Used by the
   * Restore-from-backup flow so the user does not need to reopen the
   * app manually after the DB swap.
   */
  relaunchApp: async (): Promise<void> => {
    return await ipcRenderer.invoke('app:relaunch');
  },

  /**
   * Retry bringing the backend up. Used by the startup error page's
   * Retry button when the backend failed to start.
   */
  retryBackend: async (): Promise<void> => {
    return await ipcRenderer.invoke('app:retryBackend');
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
   * Check if running in Electron (vs browser)
   */
  isElectron: (): boolean => {
    return true;
  },

  /**
   * Subscribe to download completions. Fired once per file that the main
   * process auto-saves to the Downloads folder. Returns an unsubscribe
   * function. Used to show a "saved to Downloads" toast with a
   * reveal-in-folder action.
   */
  onDownloadComplete: (
    callback: (info: {
      filename: string;
      path: string;
      success: boolean;
    }) => void,
  ): (() => void) => {
    const listener = (
      _event: unknown,
      info: { filename: string; path: string; success: boolean },
    ) => callback(info);
    ipcRenderer.on('download:complete', listener);
    return () => ipcRenderer.removeListener('download:complete', listener);
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

  /**
   * Subscribe to application-menu commands. The native menu (built in the
   * main process) sends a string id when a renderer-backed item is clicked;
   * the <MenuCommands> component runs the matching action. Returns an
   * unsubscribe function.
   */
  onMenuCommand: (callback: (id: string) => void): (() => void) => {
    const listener = (_event: unknown, id: string) => callback(id);
    ipcRenderer.on('menu:command', listener);
    return () => ipcRenderer.removeListener('menu:command', listener);
  },

  /**
   * Tell the main process which species-name mode is active so the
   * View → Species names radio shows the right checkmark. Sent on mount
   * and after each change. One-way; the renderer's localStorage stays the
   * single source of truth.
   */
  setSpeciesNameMenuMode: (mode: 'common' | 'scientific'): void => {
    ipcRenderer.send('menu:species-mode', mode);
  },

  /**
   * Tell the main process whether first-run setup has finished, so the
   * setup-only menu items (Home, backup/restore, backups folder, species
   * names) can be disabled during the wizard and enabled afterward. Sent
   * on mount and whenever the setup-ready state changes.
   */
  setMenuSetupReady: (ready: boolean): void => {
    ipcRenderer.send('menu:setup-state', ready);
  },
});
