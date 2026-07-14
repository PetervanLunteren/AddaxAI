/**
 * Electron main process
 *
 * Responsibilities:
 * - Start FastAPI backend server
 * - Create browser window pointing to backend
 * - Handle application lifecycle
 * - Clean shutdown of backend on quit
 */

import { app, BrowserWindow, crashReporter, session, shell, ipcMain, dialog, Menu } from 'electron';
import { spawn, execSync, ChildProcess } from 'child_process';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';

let mainWindow: BrowserWindow | null = null;
let backendProcess: ChildProcess | null = null;
// Populated by spawnBackend's handlers so waitForBackend can explain a
// startup death instead of waiting out the timer.
let backendSpawnError: Error | null = null;
let lastBackendExit: { code: number | null; signal: string | null } | null = null;
const BACKEND_PORT = 8000;
const BACKEND_URL = `http://localhost:${BACKEND_PORT}`;
const LOGS_DIR = path.join(os.homedir(), 'AddaxAI', 'logs');

/**
 * Parse `--timelapse <folder>` out of process.argv.
 *
 * Used by Saul Greenberg's Timelapse Analyser to spawn AddaxAI on a
 * given folder. The shim installer drops an open.bat that translates the
 * legacy `open.bat timelapse <dir>` command into
 * `AddaxAI.exe --timelapse "<dir>"`, so this flag is the single
 * integration point for both the new and legacy invocation paths.
 *
 * The flag now opens a folder analysis with the folder pre-filled (see
 * folderRunRouteForPath); the old dedicated timelapse window is gone.
 *
 * Returns null when the flag is absent. Returns "" (empty string) when
 * the flag is present without an argument — still a valid signal to
 * open a folder run, just without a pre-filled folder.
 */
function parseTimelapseArg(argv: string[]): string | null {
  const idx = argv.findIndex((a) => a === '--timelapse');
  if (idx === -1) return null;
  return argv[idx + 1] || '';
}

/**
 * In-app route a `--timelapse <folder>` launch should open: a new folder
 * run with the folder pre-filled via the `?path=` query the folder-run
 * setup step reads on first paint.
 */
function folderRunRouteForPath(folder: string): string {
  return folder
    ? `/folder-runs/new?path=${encodeURIComponent(folder)}`
    : '/folder-runs/new';
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
// running instance: it navigates the existing window to a new folder
// run with the folder pre-filled. Without an argument we just surface
// the existing main window.
if (!app.requestSingleInstanceLock()) {
  app.quit();
  process.exit(0);
}

app.on('second-instance', (_event, argv) => {
  const tlPath = parseTimelapseArg(argv);
  if (tlPath !== null && process.platform === 'win32') {
    const route = folderRunRouteForPath(tlPath);
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      void mainWindow.loadURL(`${BACKEND_URL}${route}`);
      mainWindow.show();
      mainWindow.focus();
    } else {
      void reopenWindow(route);
    }
    return;
  }
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
  } else {
    void reopenWindow();
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
    // Existence of LAUNCH_STATUS is our "have we ever launched on this
    // machine" signal. It is written every launch and is also part of
    // the reset wipe, so a fresh install and a post-reset launch both
    // look like first launches. Without this guard, the very first
    // launch always reports previous_shutdown_clean: false (because
    // SHUTDOWN_SENTINEL has never been written yet) and the user sees
    // a false-positive crash banner the moment setup completes.
    const haveLaunchedBefore = fs.existsSync(LAUNCH_STATUS);
    const wasClean = fs.existsSync(SHUTDOWN_SENTINEL);
    const previousShutdownClean = !haveLaunchedBefore || wasClean;
    fs.writeFileSync(
      LAUNCH_STATUS,
      JSON.stringify(
        {
          previous_shutdown_clean: previousShutdownClean,
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
    } else if (haveLaunchedBefore) {
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

const delay = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

type HealthBody = { status?: string; version?: string };

// How long a backend that never dies is allowed to take before we give
// up. First launch does the slow one-time work (PyInstaller unpack,
// alembic migrations, backups) behind /health, so this is generous.
const BACKEND_READY_TIMEOUT_MS = 180000;

/**
 * One IPv4 GET to /health. Resolves the parsed JSON body on a 200, or
 * null when nothing answers / it times out / the status is not 200.
 * A 200 with a non-JSON body (some other server on the port) resolves
 * to `{}`, which `isAddaxaiHealth` then rejects.
 */
function probeHealth(timeoutMs = 2000): Promise<HealthBody | null> {
  const http = require('http');
  return new Promise((resolve) => {
    const req = http.get(
      {
        hostname: '127.0.0.1',
        port: BACKEND_PORT,
        path: '/health',
        family: 4, // Force IPv4
        timeout: timeoutMs,
      },
      (res: any) => {
        if (res.statusCode !== 200) {
          res.resume();
          resolve(null);
          return;
        }
        let body = '';
        res.on('data', (chunk: any) => {
          body += chunk;
        });
        res.on('end', () => {
          try {
            resolve(JSON.parse(body));
          } catch {
            resolve({});
          }
        });
      },
    );
    req.on('error', () => resolve(null));
    req.on('timeout', () => {
      req.destroy();
      resolve(null);
    });
  });
}

/**
 * Is this /health body from an AddaxAI backend (vs some unrelated server
 * that happens to hold the port)? Our backend always returns
 * `{status: "healthy", version: "..."}`.
 */
function isAddaxaiHealth(body: HealthBody | null): boolean {
  return !!body && body.status === 'healthy' && typeof body.version === 'string';
}

/**
 * Kill whatever process is *listening* on `port`. Best-effort and
 * guarded: a missing tool or no-match just logs and returns. Only ever
 * called once we have already confirmed (via /health) that the listener
 * is a stale AddaxAI backend, never a foreign process.
 *
 * Critically, this must match only the LISTEN socket, not clients
 * connected to the port. Our own probeHealth opens a client connection
 * to 8000, so a broad `lsof -ti tcp:8000` returns Electron's own pid too
 * and we would SIGKILL ourselves. We restrict to the listening socket
 * and, belt-and-suspenders, never kill our own process.
 */
function killProcessOnPort(port: number): void {
  const pids = new Set<number>();
  try {
    if (process.platform === 'win32') {
      const out = execSync(`netstat -ano -p tcp | findstr LISTENING`, {
        encoding: 'utf8',
      });
      for (const line of out.split('\n')) {
        const parts = line.trim().split(/\s+/);
        // TCP  <local>  <foreign>  LISTENING  <pid>
        const local = parts[1];
        const pid = Number(parts[4]);
        if (local && local.endsWith(`:${port}`) && Number.isInteger(pid)) {
          pids.add(pid);
        }
      }
    } else {
      const out = execSync(`lsof -t -iTCP:${port} -sTCP:LISTEN`, {
        encoding: 'utf8',
      });
      for (const s of out.split('\n').map((x) => x.trim()).filter(Boolean)) {
        const pid = Number(s);
        if (Number.isInteger(pid)) pids.add(pid);
      }
    }
  } catch {
    // lsof / findstr exit non-zero when nothing matches — that's fine.
  }

  pids.delete(process.pid); // never kill ourselves
  if (pids.size === 0) {
    console.log(`[Electron] killProcessOnPort(${port}): no listener found`);
    return;
  }
  for (const pid of pids) {
    try {
      if (process.platform === 'win32') {
        execSync(`taskkill /PID ${pid} /F`);
      } else {
        process.kill(pid, 'SIGKILL');
      }
    } catch {
      /* already gone */
    }
  }
}

/**
 * Resolve the backend command for the current mode. Throws if the
 * executable / interpreter is missing.
 */
function resolveBackendCommand(): {
  executable: string;
  cwd: string;
  args: string[];
  isDev: boolean;
} {
  const isDev = !app.isPackaged;

  if (isDev) {
    // Development: venv Python with uvicorn
    const backendDir = path.join(__dirname, '..', '..', 'backend');
    const pythonPath = path.join(backendDir, 'venv', 'bin', 'python');
    if (!fs.existsSync(pythonPath)) {
      throw new Error(`Python not found: ${pythonPath}`);
    }
    return {
      executable: pythonPath,
      cwd: backendDir,
      args: [
        '-m', 'uvicorn',
        'app.main:app',
        '--host', '127.0.0.1',
        '--port', String(BACKEND_PORT),
        '--log-level', 'info',
      ],
      isDev,
    };
  }

  // Production: PyInstaller bundled executable (.exe on Windows).
  const exeName = process.platform === 'win32' ? 'backend.exe' : 'backend';
  const executable = path.join(process.resourcesPath, 'backend', exeName);
  if (!fs.existsSync(executable)) {
    throw new Error(`Backend executable not found: ${executable}`);
  }
  return { executable, cwd: process.cwd(), args: [], isDev };
}

/**
 * Spawn the backend process and wire up logging. Records early failures
 * (`backendSpawnError`) and the exit code (`lastBackendExit`) so
 * `waitForBackend` can fail fast instead of waiting out the timer when
 * the process dies during startup.
 */
function spawnBackend(): void {
  backendSpawnError = null;
  lastBackendExit = null;

  const { executable, cwd, args, isDev } = resolveBackendCommand();
  console.log(
    `[Electron] Starting backend (${isDev ? 'development' : 'production'}):`,
    executable,
  );

  const proc = spawn(executable, args, {
    cwd,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: {
      ...process.env,
      ...(isDev ? { PYTHONPATH: cwd } : {}),
    },
  });
  backendProcess = proc;

  proc.stdout?.on('data', (data) => {
    console.log('[Backend]', data.toString().trim());
  });
  proc.stderr?.on('data', (data) => {
    console.error('[Backend Error]', data.toString().trim());
  });
  proc.on('error', (error) => {
    backendSpawnError = error;
    console.error('[Electron] Failed to start backend:', error);
  });
  proc.on('exit', (code, signal) => {
    console.log(`[Electron] Backend exited with code ${code} and signal ${signal}`);
    lastBackendExit = { code, signal };
    if (backendProcess === proc) backendProcess = null;
  });
}

/**
 * Bring up a backend we own on BACKEND_PORT.
 *
 * Pre-flight: if something already answers /health, either it is a
 * stale AddaxAI backend (orphan from a previous run, or an old version
 * left after an update) — kill it and claim the port — or it is a
 * foreign process, in which case we refuse with a clear error. Then
 * spawn our own and wait for it, and finally re-check the version so we
 * never end up silently talking to a survivor.
 */
async function ensureBackend(): Promise<void> {
  const existing = await probeHealth(1500);
  if (existing) {
    if (isAddaxaiHealth(existing)) {
      console.warn(
        `[Electron] A backend is already on port ${BACKEND_PORT} ` +
          `(version ${existing.version}); reclaiming the port.`,
      );
      killProcessOnPort(BACKEND_PORT);
      // Wait for the port to actually free up (max ~5s).
      for (let i = 0; i < 10; i++) {
        if (!(await probeHealth(500))) break;
        await delay(500);
      }
    } else {
      throw new Error(
        `Port ${BACKEND_PORT} is in use by another application. Quit ` +
          `whatever is using it and relaunch AddaxAI.`,
      );
    }
  }

  spawnBackend();
  await waitForBackend();

  // Defensive: confirm we are talking to OUR backend, not a survivor
  // that our uvicorn failed to displace (e.g. it never freed the port).
  const health = await probeHealth(2000);
  const expected = app.getVersion();
  if (health?.version && health.version !== expected) {
    throw new Error(
      `A different AddaxAI backend (version ${health.version}) is running ` +
        `on port ${BACKEND_PORT} and could not be replaced. Fully quit any ` +
        `other AddaxAI window and relaunch.`,
    );
  }
  console.log('[Electron] Backend is ready');
}

/**
 * Wait for our backend to answer /health. Polls while the process is
 * alive; fails fast if it exits during startup (no waiting out the
 * timer); gives up after BACKEND_READY_TIMEOUT_MS for a wedged-but-alive
 * backend.
 */
async function waitForBackend(): Promise<void> {
  const start = Date.now();
  while (true) {
    if (!backendProcess) {
      const detail = backendSpawnError
        ? backendSpawnError.message
        : lastBackendExit
          ? `exit code ${lastBackendExit.code}`
          : 'unknown reason';
      throw new Error(
        `The backend stopped while starting up (${detail}). See the logs ` +
          `for details.`,
      );
    }
    const health = await probeHealth(2000);
    if (isAddaxaiHealth(health)) return;
    if (Date.now() - start > BACKEND_READY_TIMEOUT_MS) {
      throw new Error(
        'The backend is taking longer than expected to start. See the ' +
          'logs for details.',
      );
    }
    await delay(1000);
  }
}

/**
 * Stop the backend: SIGTERM, wait up to 5s for a clean exit, then
 * SIGKILL. uvicorn's graceful shutdown can hang forever on an open
 * connection, so the SIGKILL fallback is what guarantees no orphan.
 */
async function stopBackend(): Promise<void> {
  const proc = backendProcess;
  backendProcess = null;
  if (!proc || proc.exitCode !== null || proc.signalCode !== null) return;
  console.log('[Electron] Stopping backend server...');
  await new Promise<void>((resolve) => {
    const kill9 = setTimeout(() => {
      try {
        proc.kill('SIGKILL');
      } catch {
        /* already gone */
      }
      resolve();
    }, 5000);
    proc.once('exit', () => {
      clearTimeout(kill9);
      resolve();
    });
    try {
      proc.kill('SIGTERM');
    } catch {
      clearTimeout(kill9);
      resolve();
    }
  });
}

/**
 * Stop the backend and mark the shutdown clean. The single path every
 * quit / relaunch funnels through so the backend is always terminated
 * and the crash sentinel is always written.
 */
async function gracefulShutdown(): Promise<void> {
  await stopBackend();
  writeShutdownSentinel();
}

/**
 * A minimal self-contained HTML page (splash / error). Rendered from a
 * data: URL so no asset has to be bundled. No CSP is set so the error
 * page's inline button handlers can call window.electronAPI (exposed by
 * the preload, which loads for these pages too).
 */
function shellPage(bodyHtml: string): string {
  return `<!doctype html><html><head><meta charset="utf-8"><style>
    :root { color-scheme: light dark; }
    html, body { height: 100%; margin: 0; }
    body {
      display: flex; align-items: center; justify-content: center;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: #0f6064; color: #ffffff; text-align: center; padding: 2rem;
    }
    .box { max-width: 30rem; }
    h1 { font-size: 1.25rem; font-weight: 600; margin: 0 0 0.75rem; }
    p { margin: 0.5rem 0; line-height: 1.5; opacity: 0.92; }
    .msg { opacity: 0.85; font-size: 0.9rem; }
    .path { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.8rem; opacity: 0.8; }
    .spinner {
      width: 2.25rem; height: 2.25rem; margin: 0 auto 1rem;
      border: 3px solid rgba(255,255,255,0.3); border-top-color: #ffffff;
      border-radius: 50%; animation: spin 0.9s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .actions { margin-top: 1.25rem; display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap; }
    button {
      font: inherit; padding: 0.5rem 0.9rem; border-radius: 0.4rem;
      border: 1px solid rgba(255,255,255,0.5); background: rgba(255,255,255,0.12);
      color: #ffffff; cursor: pointer;
    }
    button:hover { background: rgba(255,255,255,0.22); }
  </style></head><body><div class="box">${bodyHtml}</div></body></html>`;
}

function splashHtml(): string {
  return shellPage(
    `<div class="spinner"></div><h1>Starting AddaxAI…</h1>` +
      `<p class="msg">First launch can take a minute while it sets things up.</p>`,
  );
}

function errorHtml(message: string): string {
  // Escape the backend message for safe HTML interpolation.
  const safe = message
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  // A JS string literal for the onclick, with its quotes HTML-escaped so
  // they don't terminate the double-quoted attribute. The browser decodes
  // &quot; back to " when parsing the attribute value.
  const logsArg = JSON.stringify(LOGS_DIR).replace(/"/g, '&quot;');
  const pathText = LOGS_DIR.replace(/&/g, '&amp;').replace(/</g, '&lt;');
  return shellPage(
    `<h1>AddaxAI could not start</h1>` +
      `<p class="msg">${safe}</p>` +
      `<p class="path">${pathText}</p>` +
      `<div class="actions">` +
      `<button onclick="window.electronAPI.openPath(${logsArg})">Open logs folder</button>` +
      `<button onclick="window.electronAPI.retryBackend()">Retry</button>` +
      `<button onclick="window.electronAPI.quitApp()">Quit</button>` +
      `</div>`,
  );
}

function loadHtml(html: string): Promise<void> {
  if (!mainWindow) return Promise.resolve();
  return mainWindow.loadURL(
    'data:text/html;charset=utf-8,' + encodeURIComponent(html),
  );
}

/**
 * Create the main application window and show the splash immediately.
 * The real SPA is loaded later by loadApp() once the backend is ready;
 * on failure showErrorPage() takes over. This is what keeps a slow or
 * failed backend from ever presenting a white screen.
 */
async function createWindow(): Promise<void> {
  console.log('[Electron] Creating main window...');

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1024,
    minHeight: 768,
    title: `AddaxAI v${app.getVersion()}`,
    // Show our custom application menu bar on Windows / Linux (built in
    // setupApplicationMenu). It holds every app-wide action: File (data
    // folders, backup/restore, quit), View (reload, species names), and
    // Help (documentation, diagnostics, reset, about). The bar sits flush
    // against the white app header with no visual separator, but
    // discoverability wins over aesthetics. No-op on macOS where the menu
    // lives on the system menu bar.
    autoHideMenuBar: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true,
      preload: path.join(__dirname, 'preload.js'), // Compiled from preload.ts
    },
    show: false, // Don't show until the splash has painted
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow?.show();
  });

  // The HTML <title> would otherwise override our window title with
  // "AddaxAI" (no version) once the page loads. Block that so the
  // version stays visible in the title bar at all times.
  mainWindow.on('page-title-updated', (event) => {
    event.preventDefault();
  });

  // Open external links in browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  if (!app.isPackaged) {
    mainWindow.webContents.openDevTools();
  }

  await loadHtml(splashHtml());
  if (!mainWindow.isVisible()) {
    mainWindow.show();
    mainWindow.focus();
  }
}

/**
 * Load the real frontend SPA into the existing window. An optional
 * in-app route (e.g. /folder-runs/new?path=... from the --timelapse
 * launcher) is appended so the router lands on it on first paint.
 */
async function loadApp(route = ''): Promise<void> {
  if (!mainWindow) return;

  // Clear the renderer's HTTP cache before loading the SPA. The frontend
  // is a hashed-asset Vite SPA served from the bundled backend at a stable
  // URL (http://localhost:8000). On app upgrade the bundle ships new asset
  // hashes, but the renderer may still have an index.html cached from the
  // previous install referring to hashes that no longer exist, producing a
  // white / unstyled page. Wiping the cache makes that structurally
  // impossible. The cost is a small re-download from localhost, negligible.
  await session.defaultSession.clearCache();
  await mainWindow.loadURL(`${BACKEND_URL}${route}`);

  // Belt-and-suspenders: if ready-to-show didn't fire, force visible.
  if (!mainWindow.isVisible()) {
    mainWindow.show();
    mainWindow.focus();
  }
}

/**
 * Replace the splash with an error page explaining why the backend did
 * not come up, with buttons to open the logs, retry, or quit.
 */
async function showErrorPage(message: string): Promise<void> {
  await loadHtml(errorHtml(message));
  if (mainWindow && !mainWindow.isVisible()) {
    mainWindow.show();
    mainWindow.focus();
  }
}

/**
 * Bring the backend up and load the app, showing the error page on any
 * failure instead of quitting. Used by the initial launch and the error
 * page's Retry button.
 */
async function startBackendAndLoad(route = ''): Promise<void> {
  try {
    await ensureBackend();
    await loadApp(route);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error('[Electron] Startup failed:', message);
    await showErrorPage(message);
  }
}

/**
 * Re-open a main window when one doesn't exist (macOS dock activate, or a
 * second-instance launch). The backend is already running at this point,
 * so we skip ensureBackend and go straight to the SPA.
 */
async function reopenWindow(route = ''): Promise<void> {
  await createWindow();
  await loadApp(route);
}

/**
 * Send a menu command to the focused renderer. The menu lives in the main
 * process, but almost every action's logic already lives in the renderer
 * (React Router navigation, the backup/restore/reset/updates dialogs, the
 * species-name preference). Rather than duplicate that logic here, each
 * renderer-backed menu item sends a single string command that the
 * <MenuCommands> component dispatches. Roles (reload, quit, copy/paste) and
 * external links (Documentation) are handled natively and never come here.
 */
function sendMenuCommand(id: string): void {
  const win = BrowserWindow.getFocusedWindow() ?? mainWindow;
  win?.webContents.send('menu:command', id);
}

/**
 * Build the application menu template. macOS gets the conventional app menu
 * as the first submenu (About / Quit live there); Windows and Linux fold
 * those entries into File and Help instead. Standard editing, reload, zoom
 * and window behaviour use built-in Electron roles so we don't reimplement
 * them.
 */
function buildMenuTemplate(): Electron.MenuItemConstructorOptions[] {
  const isMac = process.platform === 'darwin';

  const aboutItem: Electron.MenuItemConstructorOptions = {
    label: 'About AddaxAI',
    click: () => sendMenuCommand('about'),
  };
  const checkForUpdatesItem: Electron.MenuItemConstructorOptions = {
    label: 'Check for updates…',
    click: () => sendMenuCommand('check-updates'),
  };

  const template: Electron.MenuItemConstructorOptions[] = [];

  if (isMac) {
    template.push({
      label: 'AddaxAI',
      submenu: [
        aboutItem,
        checkForUpdatesItem,
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideOthers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' },
      ],
    });
  }

  template.push({
    label: 'File',
    submenu: [
      { id: 'new-project', label: 'New project…', click: () => sendMenuCommand('new-project') },
      { id: 'new-folder-run', label: 'Analyse a folder…', click: () => sendMenuCommand('new-folder-run') },
      { type: 'separator' },
      { id: 'nav-home', label: 'Home', click: () => sendMenuCommand('nav-home') },
      { type: 'separator' },
      { label: 'Open user data folder', click: () => sendMenuCommand('open-user-data') },
      { id: 'open-backups', label: 'Open backups folder', click: () => sendMenuCommand('open-backups') },
      { type: 'separator' },
      { id: 'backup', label: 'Back up database…', click: () => sendMenuCommand('backup') },
      { id: 'restore', label: 'Restore from backup…', click: () => sendMenuCommand('restore') },
      // Windows / Linux have no app menu, so Check for updates and Quit
      // live here instead.
      ...(isMac
        ? []
        : ([
            { type: 'separator' },
            checkForUpdatesItem,
            { type: 'separator' },
            { role: 'quit' },
          ] as Electron.MenuItemConstructorOptions[])),
    ],
  });

  template.push({ role: 'editMenu' });

  template.push({
    label: 'View',
    submenu: [
      { role: 'reload' },
      { role: 'forceReload' },
      { type: 'separator' },
      {
        id: 'species-names',
        label: 'Species names',
        submenu: [
          {
            id: 'species-common',
            label: 'Common',
            type: 'radio',
            checked: true,
            click: () => sendMenuCommand('species-common'),
          },
          {
            id: 'species-scientific',
            label: 'Scientific',
            type: 'radio',
            checked: false,
            click: () => sendMenuCommand('species-scientific'),
          },
        ],
      },
      { label: 'Language (coming soon)', enabled: false },
      { type: 'separator' },
      { role: 'togglefullscreen' },
      { role: 'toggleDevTools' },
      { type: 'separator' },
      { role: 'resetZoom' },
      { role: 'zoomIn' },
      { role: 'zoomOut' },
    ],
  });

  if (isMac) {
    template.push({ role: 'windowMenu' });
  }

  template.push({
    role: 'help',
    label: 'Help',
    submenu: [
      { label: 'Documentation (coming soon)', enabled: false },
      { label: 'Video tutorials (coming soon)', enabled: false },
      { type: 'separator' },
      { label: 'Export diagnostic report', click: () => sendMenuCommand('export-diagnostic') },
      { type: 'separator' },
      { label: 'Reset application…', click: () => sendMenuCommand('reset') },
      // About lives in the app menu on macOS; fold it into Help elsewhere.
      ...(isMac ? [] : ([{ type: 'separator' }, aboutItem] as Electron.MenuItemConstructorOptions[])),
    ],
  });

  return template;
}

// Menu items that only make sense once first-run setup has finished.
// Disabled until the renderer reports setup-ready over 'menu:setup-state':
//   - nav-home / open-backups: navigation and a folder that are empty or
//     loop back to the wizard during setup.
//   - backup / restore: nothing to back up yet, no backups to restore.
//   - species-names: no species data exists, so the toggle does nothing.
// Diagnostics, Reset, Check for updates, About, reload and quit stay
// enabled so a stuck setup can still be inspected or recovered.
const SETUP_GATED_MENU_IDS = [
  'new-project',
  'new-folder-run',
  'nav-home',
  'open-backups',
  'backup',
  'restore',
  'species-names',
] as const;

/**
 * Enable or disable the setup-gated menu items. Called whenever the
 * renderer reports its setup-ready state. Items default to disabled at
 * build time so a fresh install never shows a live-but-useless action
 * before the first status report lands.
 */
function setMenuSetupReady(ready: boolean): void {
  const menu = Menu.getApplicationMenu();
  if (!menu) return;
  for (const id of SETUP_GATED_MENU_IDS) {
    const item = menu.getMenuItemById(id);
    if (item) item.enabled = ready;
  }
}

/**
 * Build and install the application menu. Called once on startup. The
 * renderer keeps the View → Species names radio in sync via the
 * 'menu:species-mode' channel, and gates the setup-only items via
 * 'menu:setup-state'.
 */
function setupApplicationMenu(): void {
  Menu.setApplicationMenu(Menu.buildFromTemplate(buildMenuTemplate()));
  // Default to not-ready: the renderer re-enables these once setup
  // status reports ready. Avoids a flash of live items on first launch.
  setMenuSetupReady(false);
}

/**
 * IPC handlers
 */

// Handle folder selection dialog. The dialog is made window-modal to the
// calling window so users cannot click around the form while the picker
// is open, nor open two pickers in parallel by double-clicking the
// drop zone. Resolves the sender's window via event.sender so the
// handler attaches to whichever window made the call.
ipcMain.handle('dialog:selectFolder', async (event) => {
  const win = BrowserWindow.fromWebContents(event.sender);
  const options: Electron.OpenDialogOptions = {
    properties: ['openDirectory'],
    title: 'Select folder with camera trap images',
  };
  const result = win
    ? await dialog.showOpenDialog(win, options)
    : await dialog.showOpenDialog(options);

  if (result.canceled) {
    return null;
  }

  return result.filePaths[0] || null;
});

// Handle single-file selection dialog. Caller can pass `filters` to
// constrain the picker (e.g. .db files for the Restore-from-backup flow).
// Window-modal to the calling window for the same reason as selectFolder.
// Returns the selected path or null when the user cancels.
ipcMain.handle(
  'dialog:openFile',
  async (
    event,
    opts?: { title?: string; filters?: Electron.FileFilter[] },
  ) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    const options: Electron.OpenDialogOptions = {
      properties: ['openFile'],
      title: opts?.title ?? 'Select file',
      filters: opts?.filters,
    };
    const result = win
      ? await dialog.showOpenDialog(win, options)
      : await dialog.showOpenDialog(options);
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
ipcMain.handle('app:relaunch', async () => {
  app.relaunch();
  // app.exit() skips before-quit / will-quit, so stop the backend
  // explicitly here or the relaunched app inherits an orphan on port 8000.
  await gracefulShutdown();
  app.exit(0);
});

// Retry backend startup from the error page's Retry button. Re-runs the
// full ensureBackend + loadApp path, falling back to the error page again
// if it still fails.
ipcMain.handle('app:retryBackend', async () => {
  await startBackendAndLoad();
});

// Return the runtime app version (e.g. "0.2.0-beta.1"). The version is
// written into electron/package.json by the release workflow's
// "Sync version from release tag" step, so this is always the actual
// shipping version. Used by the About page and the update-check.
ipcMain.handle('app:getVersion', () => {
  return app.getVersion();
});

// Keep the View → Species names radio in sync with the renderer's stored
// preference (localStorage, device-global). The renderer sends its current
// mode on mount and after every change (a change reloads the page, so the
// post-reload mount re-syncs the checkmark). One-way: the menu reflects the
// renderer, never the reverse.
ipcMain.on('menu:species-mode', (_event, mode: string) => {
  const id = mode === 'scientific' ? 'species-scientific' : 'species-common';
  const item = Menu.getApplicationMenu()?.getMenuItemById(id);
  if (item) item.checked = true;
});

// Enable / disable the setup-gated menu items. The renderer sends its
// current setup-ready state on mount and whenever it changes, so items
// that only make sense post-setup stay greyed out during the first-run
// wizard.
ipcMain.on('menu:setup-state', (_event, ready: boolean) => {
  setMenuSetupReady(Boolean(ready));
});

/**
 * Application lifecycle handlers
 */

/**
 * Pick a non-colliding path: append " (1)", " (2)", ... before the
 * extension if the target already exists, mirroring the OS download
 * behaviour so a second export never silently overwrites the first.
 */
function uniqueDownloadPath(target: string): string {
  if (!fs.existsSync(target)) return target;
  const dir = path.dirname(target);
  const ext = path.extname(target);
  const base = path.basename(target, ext);
  let i = 1;
  let candidate = path.join(dir, `${base} (${i})${ext}`);
  while (fs.existsSync(candidate)) {
    i += 1;
    candidate = path.join(dir, `${base} (${i})${ext}`);
  }
  return candidate;
}

/**
 * Send renderer-initiated downloads straight to the user's Downloads
 * folder instead of popping a Save dialog per file. This is what makes a
 * multi-file export (the spreadsheet CSV/TSV writes detections + counts)
 * land smoothly without two dialogs. Attached once to the default session.
 *
 * Only affects browser-style downloads (the Export page's blob downloads,
 * the camtrap-dp ZIP). The folder-run Save step writes files server-side
 * to a chosen folder and never goes through this path.
 */
function setupDownloadHandler(): void {
  session.defaultSession.on('will-download', (_event, item) => {
    const downloadsDir = app.getPath('downloads');
    const savePath = uniqueDownloadPath(
      path.join(downloadsDir, item.getFilename()),
    );
    item.setSavePath(savePath);
    item.once('done', (_e, state) => {
      mainWindow?.webContents.send('download:complete', {
        filename: path.basename(savePath),
        path: savePath,
        success: state === 'completed',
      });
    });
  });
}

app.on('ready', async () => {
  try {
    setupDownloadHandler();
    setupApplicationMenu();
    // Show the window (splash) immediately, then bring the backend up.
    await createWindow();
    // When launched via `AddaxAI.exe --timelapse <folder>` (Saul's
    // Timelapse integration / shim), land straight on a new folder run
    // with the folder pre-filled. Timelapse Analyser is Windows-only, so
    // the flag is only meaningful on Windows; elsewhere we ignore it.
    const timelapsePath = parseTimelapseArg(process.argv);
    const route =
      timelapsePath !== null && process.platform === 'win32'
        ? folderRunRouteForPath(timelapsePath)
        : '';
    await startBackendAndLoad(route);
  } catch (error) {
    // createWindow itself failed (very unlikely) — there is nothing to
    // show the error in, so quit.
    console.error('[Electron] Fatal startup error:', error);
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
  // On macOS, re-create window when dock icon is clicked. The backend is
  // already up, so go straight to the SPA.
  if (mainWindow === null) {
    void reopenWindow();
  }
});

// Guarantee the backend is stopped before the process exits. stopBackend
// is async (SIGTERM, wait, SIGKILL), so we preventDefault the first
// before-quit, run the shutdown, then quit again — the guard lets the
// second pass through. If the process is killed before this runs
// (SIGKILL, panic, OOM, power loss), the sentinel stays absent and the
// next launch detects the crash.
let isQuitting = false;
app.on('before-quit', (event) => {
  if (isQuitting) return;
  event.preventDefault();
  isQuitting = true;
  void gracefulShutdown().then(() => app.quit());
});

// Handle uncaught errors
process.on('uncaughtException', (error) => {
  console.error('[Electron] Uncaught exception:', error);
});

process.on('unhandledRejection', (error) => {
  console.error('[Electron] Unhandled rejection:', error);
});
