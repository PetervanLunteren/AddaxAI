/**
 * TypeScript declarations for Electron APIs exposed via preload script
 */

export interface ElectronAPI {
  selectFolder: () => Promise<string | null>;
  showItemInFolder: (filePath: string) => Promise<void>;
  isElectron: () => boolean;
}

declare global {
  interface Window {
    electronAPI?: ElectronAPI;
  }
}

export {};
