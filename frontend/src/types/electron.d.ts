/**
 * TypeScript declarations for Electron APIs exposed via preload script
 */

export interface ElectronAPI {
  selectFolder: () => Promise<string | null>;
  showItemInFolder: (filePath: string) => Promise<void>;
  /**
   * Open a file or directory with the OS default handler. For directories
   * this opens the folder in the system file manager. Returns an error
   * string on failure, empty string on success.
   */
  openPath: (targetPath: string) => Promise<string>;
  isElectron: () => boolean;
}

declare global {
  interface Window {
    electronAPI?: ElectronAPI;
  }
}

export {};
