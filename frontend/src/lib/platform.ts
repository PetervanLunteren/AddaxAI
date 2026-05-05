/**
 * Platform detection utilities
 *
 * Detect whether the app is running in Electron or browser (dev mode)
 */

/**
 * Check if running in Electron environment
 */
export function isElectron(): boolean {
  return typeof window !== 'undefined' && !!window.electronAPI;
}

/**
 * Check if running in development mode (browser)
 */
export function isDevelopment(): boolean {
  return !isElectron();
}

/**
 * Get the platform name
 */
export function getPlatform(): 'electron' | 'browser' {
  return isElectron() ? 'electron' : 'browser';
}

/**
 * Whether the host is Windows. Returns true for Electron-on-Windows
 * and also for the dev browser (where there's no Electron, so we treat
 * "anywhere a developer might be testing" as compatible).
 *
 * Used to gate Windows-only features such as Timelapse mode (which
 * integrates with Timelapse Analyser, a Windows-only desktop app).
 */
export function isWindowsOrDev(): boolean {
  if (typeof window === 'undefined') return false;
  if (!window.electronAPI) return true; // dev browser, no Electron
  return window.electronAPI.platform === 'win32';
}
