/**
 * Electron main process
 *
 * Responsibilities:
 * - Start FastAPI backend server
 * - Create browser window pointing to backend
 * - Handle application lifecycle
 * - Clean shutdown of backend on quit
 */

import { app, BrowserWindow, shell, ipcMain, dialog } from 'electron';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

let mainWindow: BrowserWindow | null = null;
let backendProcess: ChildProcess | null = null;
const BACKEND_PORT = 8000;
const BACKEND_URL = `http://localhost:${BACKEND_PORT}`;

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

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1024,
    minHeight: 768,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true,
      preload: path.join(__dirname, 'preload.js'), // Compiled from preload.ts
    },
    show: false, // Don't show until ready
  });

  // Load the frontend from backend
  await mainWindow.loadURL(BACKEND_URL);

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow?.show();
  });

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
