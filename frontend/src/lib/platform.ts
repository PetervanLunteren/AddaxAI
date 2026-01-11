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
