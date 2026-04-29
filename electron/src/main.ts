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
    // Hide the native menu bar on Windows/Linux until the user presses
    // Alt. Without this it sits flush against the white app header and
    // there's no visual break between OS chrome and app chrome.
    // No-op on macOS (menu lives on the system menu bar).
    autoHideMenuBar: true,
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

/**
 * Application lifecycle handlers
 */

app.on('ready', async () => {
  try {
    await startBackend();
    await createWindow();
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
