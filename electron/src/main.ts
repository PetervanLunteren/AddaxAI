/**
 * Electron main process
 *
 * Responsibilities:
 * - Start FastAPI backend server
 * - Create browser window pointing to backend
 * - Handle application lifecycle
 * - Clean shutdown of backend on quit
 */

import { app, BrowserWindow, crashReporter, session, shell, ipcMain, dialog } from 'electron';
import { spawn, ChildProcess } from 'child_process';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';

let mainWindow: BrowserWindow | null = null;
let backendProcess: ChildProcess | null = null;
const BACKEND_PORT = 8000;
const BACKEND_URL = `http://localhost:${BACKEND_PORT}`;

/**
 * Parse `--timelapse <folder>` out of process.argv.
 *
 * Used by Saul Greenberg's Timelapse Analyser to spawn AddaxAI in
 * Timelapse-only mode (no main projects window). The shim installer
 * drops an open.bat that translates the legacy `open.bat timelapse <dir>`
 * command into `AddaxAI.exe --timelapse "<dir>"`, so this flag is the
 * single integration point for both the new and legacy invocation paths.
 *
 * Returns null when the flag is absent. Returns "" (empty string) when
 * the flag is present without an argument — still a valid signal to
 * open the Timelapse integration window, just without a pre-filled folder.
 */
function parseTimelapseArg(argv: string[]): string | null {
  const idx = argv.findIndex((a) => a === '--timelapse');
  if (idx === -1) return null;
  return argv[idx + 1] || '';
}

// Native Chromium / V8 crashes (renderer segfault, OOM, GPU process
// crash) bypass uncaughtException entirely. crashReporter writes a
// minidump to disk at the cross-platform location below; the user can
// attach it to a support bundle. submitURL is required by the API but
// uploadToServer:false guarantees we never send it anywhere.
const CRASH_DUMP_DIR = path.join(os.homedir(), 'AddaxAI', 'crash-dumps');
try {
  fs.mkdirSync(CRASH_DUMP_DIR, { recursive: true });
  app.setPath('crashDumps', CRASH_DUMP_DIR);
  crashReporter.start({
    productName: 'AddaxAI',
    companyName: 'AddaxAI',
    submitURL: 'https://invalid.invalid/never-uploaded',
    uploadToServer: false,
    ignoreSystemCrashHandler: false,
  });
} catch (e) {
  console.error('[Electron] Failed to start crashReporter:', e);
}

// Single-instance lock. If another AddaxAI is already running, we
// MUST bail out before `snapshotPreviousShutdown()` runs below, or
// we'd poison `.last-launch-status.json` (the running instance has
// already consumed the sentinel on its own startup, so the second
// instance sees it missing and writes `previous_shutdown_clean: false`).
// The running instance's backend then surfaces a false-positive crash
// banner.
//
// The `second-instance` handler forwards a `--timelapse <folder>`
// invocation (Saul's Timelapse Analyser shim, or the user double-
// clicking AddaxAI.exe while it is already open) to the already-
// running instance: it opens a Timelapse window in place. Without an
// argument we just surface the existing main window.
if (!app.requestSingleInstanceLock()) {
  app.quit();
  process.exit(0);
}

app.on('second-instance', (_event, argv) => {
  const tlPath = parseTimelapseArg(argv);
  if (tlPath !== null && process.platform === 'win32') {
    void createTimelapseWindow(tlPath || undefined);
    return;
  }
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
  } else {
    void createWindow();
  }
});

// Crash sentinel pair:
//   .last-shutdown-clean  — written by Electron on graceful exit; absence
//                            on next launch implies the previous run
//                            crashed (OOM / SIGKILL / panic / power loss).
//   .last-launch-status.json — snapshot we write at startup capturing
//                            "was the previous shutdown clean?". The
//                            backend (and frontend banner via the
//                            backend) read this file. Snapshotting at
//                            launch is necessary because we delete the
//                            sentinel here so the next crash also gets
//                            detected; without the snapshot, backend
//                            calls during the same session would always
//                            see "no sentinel" and report a false crash.
const SHUTDOWN_SENTINEL = path.join(os.homedir(), 'AddaxAI', '.last-shutdown-clean');
const LAUNCH_STATUS = path.join(os.homedir(), 'AddaxAI', '.last-launch-status.json');

function snapshotPreviousShutdown(): void {
  try {
    fs.mkdirSync(path.dirname(LAUNCH_STATUS), { recursive: true });
    const wasClean = fs.existsSync(SHUTDOWN_SENTINEL);
    fs.writeFileSync(
      LAUNCH_STATUS,
      JSON.stringify(
        {
          previous_shutdown_clean: wasClean,
          current_launch_at: new Date().toISOString(),
        },
        null,
        2,
      ),
      'utf8',
    );
    if (wasClean) {
      // Consume the sentinel: a future crash this session must produce a
      // *new* "missing sentinel" signal next time, not be masked by the
      // old one.
      try {
        fs.unlinkSync(SHUTDOWN_SENTINEL);
      } catch {
        /* ignore */
      }
    } else {
      console.warn(
        '[Electron] Previous shutdown was not clean; the app may have crashed.',
      );
    }
  } catch (e) {
    console.error('[Electron] Failed to snapshot shutdown state:', e);
  }
}

function writeShutdownSentinel(): void {
  try {
    fs.mkdirSync(path.dirname(SHUTDOWN_SENTINEL), { recursive: true });
    fs.writeFileSync(SHUTDOWN_SENTINEL, new Date().toISOString(), 'utf8');
  } catch (e) {
    console.error('[Electron] Failed to write shutdown sentinel:', e);
  }
}

snapshotPreviousShutdown();

/**
 * Start the FastAPI backend server
 */
async function startBackend(): Promise<void> {
  return new Promise((resolve, reject) => {
    console.log('[Electron] Starting backend server...');

    const isDev = !app.isPackaged;

    let backendExecutable: string;
    let backendCwd: string;
    let backendArgs: string[] = [];

    if (isDev) {
      // Development: Use venv Python with uvicorn
      const backendDir = path.join(__dirname, '..', '..', 'backend');
      const pythonPath = path.join(backendDir, 'venv', 'bin', 'python');

      if (!fs.existsSync(pythonPath)) {
        reject(new Error(`Python not found: ${pythonPath}`));
        return;
      }

      backendExecutable = pythonPath;
      backendCwd = backendDir;
      backendArgs = [
        '-m', 'uvicorn',
        'app.main:app',
        '--host', '127.0.0.1',
        '--port', String(BACKEND_PORT),
        '--log-level', 'info'
      ];

      console.log('[Electron] Development mode - using venv Python');
    } else {
      // Production: Use PyInstaller bundled executable
      // Windows requires .exe extension, macOS/Linux do not
      const exeName = process.platform === 'win32' ? 'backend.exe' : 'backend';
      backendExecutable = path.join(process.resourcesPath, 'backend', exeName);
      backendCwd = process.cwd(); // Current working directory for database/files

      if (!fs.existsSync(backendExecutable)) {
        reject(new Error(`Backend executable not found: ${backendExecutable}`));
        return;
      }

      console.log('[Electron] Production mode - using PyInstaller executable');
    }

    console.log('[Electron] Starting backend:', backendExecutable);

    backendProcess = spawn(backendExecutable, backendArgs, {
      cwd: backendCwd,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        ...(isDev ? { PYTHONPATH: backendCwd } : {})
      }
    });

    // Log backend output
    backendProcess.stdout?.on('data', (data) => {
      console.log('[Backend]', data.toString().trim());
    });

    backendProcess.stderr?.on('data', (data) => {
      console.error('[Backend Error]', data.toString().trim());
    });

    backendProcess.on('error', (error) => {
      console.error('[Electron] Failed to start backend:', error);
      reject(error);
    });

    backendProcess.on('exit', (code, signal) => {
      console.log(`[Electron] Backend exited with code ${code} and signal ${signal}`);
      backendProcess = null;
    });

    // Wait for backend to be ready
    waitForBackend(BACKEND_URL)
      .then(() => {
        console.log('[Electron] Backend is ready');
        resolve();
      })
      .catch(reject);
  });
}

/**
 * Wait for backend to respond to health check
 */
async function waitForBackend(url: string, maxAttempts = 30): Promise<void> {
  const http = require('http');

  for (let i = 0; i < maxAttempts; i++) {
    try {
      const healthCheck = await new Promise<boolean>((resolve) => {
        // Use explicit options to force IPv4 connection
        const options = {
          hostname: '127.0.0.1',
          port: BACKEND_PORT,
          path: '/health',
          family: 4, // Force IPv4
          timeout: 2000
        };

        const req = http.get(options, (res: any) => {
          console.log(`[Electron] Health check response: ${res.statusCode}`);
          resolve(res.statusCode === 200);
        });
        req.on('error', (err: any) => {
          console.log(`[Electron] Health check error: ${err.message}`);
          resolve(false);
        });
        req.on('timeout', () => {
          console.log(`[Electron] Health check timeout`);
          req.destroy();
          resolve(false);
        });
      });

      if (healthCheck) {
        console.log(`[Electron] Backend health check passed after ${i + 1} attempts`);
        return;
      }
    } catch (error) {
      console.log(`[Electron] Health check exception:`, error);
    }

    console.log(`[Electron] Waiting for backend... (attempt ${i + 1}/${maxAttempts})`);
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  throw new Error('Backend failed to start within 30 seconds');
}

/**
 * Stop the backend server
 */
function stopBackend(): void {
  if (backendProcess) {
    console.log('[Electron] Stopping backend server...');
    backendProcess.kill('SIGTERM');
    backendProcess = null;
  }
}

/**
 * Create the main application window
 */
async function createWindow(): Promise<void> {
  console.log('[Electron] Creating main window...');

  const appTitle = `AddaxAI v${app.getVersion()}`;

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1024,
    minHeight: 768,
    title: appTitle,
    // Show the native menu bar on Windows / Linux. Beta testers
    // benefit from seeing File / Edit / View / Window affordances
    // (especially View → Force Reload for clearing stale state). The
    // menu is slightly ugly because it sits flush against the white
    // app header with no visual separator, but discoverability wins
    // over aesthetics. No-op on macOS where the menu lives on the
    // system menu bar.
    autoHideMenuBar: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true,
      preload: path.join(__dirname, 'preload.js'), // Compiled from preload.ts
    },
    show: false, // Don't show until ready
  });

  // Attach all listeners that depend on a window event BEFORE loading
  // the URL. ready-to-show fires once when the renderer has rendered for
  // the first time; on Windows it consistently fires *before*
  // loadURL(...) resolves, so attaching the listener after the load
  // (as we used to) misses the event entirely and the window stays
  // invisible while the renderer happily polls the backend in the
  // background. Attach early; show() is then a one-liner.
  mainWindow.once('ready-to-show', () => {
    mainWindow?.show();
  });

  // The HTML <title> would otherwise override our window title with
  // "AddaxAI" (no version) once the page loads. Block that so the
  // version stays visible in the title bar at all times.
  mainWindow.on('page-title-updated', (event) => {
    event.preventDefault();
  });

  // Clear the renderer's HTTP cache before each load. The frontend is a
  // hashed-asset Vite SPA served from the bundled backend at a stable URL
  // (http://localhost:8000). On app upgrade the bundle ships new asset
  // hashes, but the renderer may still have an index.html cached from the
  // previous install referring to hashes that no longer exist, producing a
  // white / unstyled page on first launch. Wiping the cache at startup
  // makes that failure mode structurally impossible. The cost is a small
  // re-download from localhost on each launch, which is negligible.
  await session.defaultSession.clearCache();

  // Load the frontend from backend
  await mainWindow.loadURL(BACKEND_URL);

  // Belt-and-suspenders: if ready-to-show somehow didn't fire (Electron
  // bug, OS quirk, race we didn't anticipate), force the window visible
  // after loadURL resolves. show() is idempotent so calling it twice
  // is harmless. focus() ensures the window comes to the foreground on
  // Windows where the OS may otherwise leave it behind another app.
  if (!mainWindow.isVisible()) {
    mainWindow.show();
    mainWindow.focus();
  }

  // Open external links in browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  // Handle window close
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Open DevTools in development
  if (!app.isPackaged) {
    mainWindow.webContents.openDevTools();
  }
}

/**
 * Create the Timelapse Analyser integration window.
 *
 * Smaller than the main window because the form is a single-pane
 * focused workflow. The URL query carries the optional pre-filled
 * folder path so the renderer can populate the folder picker on first
 * paint when launched via `AddaxAI.exe --timelapse <folder>`.
 */
async function createTimelapseWindow(prefilledPath?: string): Promise<void> {
  // Narrower than the main app window: Timelapse is a single-column
  // focused form (folder, classifier, label selection, advanced
  // disclosure), not a dashboard or grid. The page content itself is
  // capped at max-w-5xl, so a wider window just gives empty side
  // margins. Users can still resize wider if they want.
  const win = new BrowserWindow({
    width: 1280,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    title: 'AddaxAI - Timelapse integration',
    autoHideMenuBar: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    show: false,
  });

  win.once('ready-to-show', () => win.show());
  win.on('page-title-updated', (e) => e.preventDefault());

  const query = prefilledPath
    ? `?path=${encodeURIComponent(prefilledPath)}`
    : '';
  await win.loadURL(`${BACKEND_URL}/timelapse${query}`);

  if (!win.isVisible()) {
    win.show();
    win.focus();
  }

  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  if (!app.isPackaged) {
    win.webContents.openDevTools({ mode: 'detach' });
  }
}

/**
 * IPC handlers
 */

// Handle folder selection dialog
ipcMain.handle('dialog:selectFolder', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory'],
    title: 'Select folder with camera trap images',
  });

  if (result.canceled) {
    return null;
  }

  return result.filePaths[0] || null;
});

// Handle single-file selection dialog. Caller can pass `filters` to
// constrain the picker (e.g. .db files for the Restore-from-backup flow).
// Returns the selected path or null when the user cancels.
ipcMain.handle(
  'dialog:openFile',
  async (
    _event,
    opts?: { title?: string; filters?: Electron.FileFilter[] },
  ) => {
    const result = await dialog.showOpenDialog({
      properties: ['openFile'],
      title: opts?.title ?? 'Select file',
      filters: opts?.filters,
    });
    if (result.canceled) return null;
    return result.filePaths[0] ?? null;
  },
);

// Reveal a file in the native file explorer
ipcMain.handle('shell:showItemInFolder', async (_event, filePath: string) => {
  shell.showItemInFolder(filePath);
});

// Open a file or directory in the OS default handler. For directories
// this opens the folder's contents in Finder / Explorer / etc, rather
// than highlighting it in the parent (which is what showItemInFolder
// does). Returns an error string on failure, empty string on success.
ipcMain.handle('shell:openPath', async (_event, targetPath: string) => {
  return await shell.openPath(targetPath);
});

// Quit the app cleanly. Used by the Reset flow in Settings: after the
// backend wipes user data, the renderer asks the main process to close
// so the next launch starts from a fresh state.
ipcMain.handle('app:quit', () => {
  app.quit();
});

// Relaunch the app: schedule a fresh start, then exit the current
// process. Used by the Restore-from-backup flow so the user does not
// have to manually double-click the app again to finish the restore.
// Reset still uses app:quit because its intent is "wipe and walk away".
ipcMain.handle('app:relaunch', () => {
  app.relaunch();
  app.exit(0);
});

// Open the Timelapse Analyser integration in a separate BrowserWindow.
// Called from the main app's hamburger menu and from the --timelapse
// CLI launcher. The window is intentionally a sibling of the main one
// (not modal) so users can keep the projects app open in the background.
ipcMain.handle(
  'window:openTimelapse',
  async (_event, prefilledPath?: string) => {
    await createTimelapseWindow(prefilledPath);
  },
);

// Return the runtime app version (e.g. "0.2.0-beta.1"). The version is
// written into electron/package.json by the release workflow's
// "Sync version from release tag" step, so this is always the actual
// shipping version. Used by the About page and the update-check.
ipcMain.handle('app:getVersion', () => {
  return app.getVersion();
});

/**
 * Application lifecycle handlers
 */

app.on('ready', async () => {
  try {
    await startBackend();
    // When launched via `AddaxAI.exe --timelapse <folder>` (Saul's
    // Timelapse integration / shim), open ONLY the Timelapse window.
    // The main projects window stays out of sight so the user is not
    // confused about which app they are working in.
    //
    // Timelapse Analyser is Windows-only, so the flag is only meaningful
    // on Windows. On macOS/Linux we ignore it and open the main window;
    // this is a defensive guard since the legacy shim that produces the
    // flag is itself Windows-only.
    const timelapsePath = parseTimelapseArg(process.argv);
    if (timelapsePath !== null && process.platform === 'win32') {
      await createTimelapseWindow(timelapsePath || undefined);
    } else {
      await createWindow();
    }
  } catch (error) {
    console.error('[Electron] Failed to start application:', error);
    app.quit();
  }
});

app.on('window-all-closed', () => {
  // On macOS, apps typically stay open until explicitly quit
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  // On macOS, re-create window when dock icon is clicked
  if (mainWindow === null) {
    createWindow();
  }
});

app.on('before-quit', () => {
  stopBackend();
  // Mark this shutdown as clean. If the process is killed before this
  // runs (SIGKILL, panic, OOM, power loss), the sentinel stays absent
  // and the next launch detects the crash.
  writeShutdownSentinel();
});

app.on('will-quit', () => {
  stopBackend();
});

// Handle uncaught errors
process.on('uncaughtException', (error) => {
  console.error('[Electron] Uncaught exception:', error);
});

process.on('unhandledRejection', (error) => {
  console.error('[Electron] Unhandled rejection:', error);
});
