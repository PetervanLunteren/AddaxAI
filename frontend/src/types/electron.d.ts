/**
 * TypeScript declarations for Electron APIs exposed via preload script
 */

export interface ElectronAPI {
  selectFolder: () => Promise<string | null>;
  isElectron: () => boolean;
}

declare global {
  interface Window {
    electronAPI?: ElectronAPI;
  }
}

export {};
