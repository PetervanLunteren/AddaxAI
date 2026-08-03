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
// Overridable so the app can be launched against a throwaway user data
// dir without fighting a dev instance for the port. `spawnBackend`
// passes the same value to the backend as API_PORT, so the two agree in
// both dev and packaged builds from this one setting.
const BACKEND_PORT = Number(process.env.ADDAXAI_BACKEND_PORT) || 8000;
const BACKEND_URL = `http://localhost:${BACKEND_PORT}`;

/**
 * The user data directory, resolved the same way the backend resolves
 * it (`USER_DATA_DIR`, falling back to `~/AddaxAI`). Every path below
 * derives from this.
 *
 * The two processes have to agree: they communicate through files in
 * here (the crash sentinels, the startup error, the restore and wipe
 * markers). Hardcoding `~/AddaxAI` on this side meant an override sent
 * the backend somewhere else and the markers landed where nothing read
 * them, which is also what made the app impossible to run end to end
 * against a throwaway database.
 */
const USER_DATA_DIR =
  process.env.USER_DATA_DIR || path.join(os.homedir(), 'AddaxAI');
const LOGS_DIR = path.join(USER_DATA_DIR, 'logs');

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
const CRASH_DUMP_DIR = path.join(USER_DATA_DIR, 'crash-dumps');
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
const SHUTDOWN_SENTINEL = path.join(USER_DATA_DIR, '.last-shutdown-clean');
const LAUNCH_STATUS = path.join(USER_DATA_DIR, '.last-launch-status.json');

// Written by the backend when it refuses to start (see
// app/core/startup_error.py), read here once the process has died. The
// backend exits before the API or the frontend exist, so this file is
// the only way a startup refusal can reach the user. Deleted just
// before every spawn, so whatever is here belongs to this launch.
const STARTUP_ERROR_FILE = path.join(USER_DATA_DIR, '.startup-error.txt');
const BACKUPS_DIR = path.join(USER_DATA_DIR, 'backups');

// Markers the backend lifespan consumes on the next launch. Writing them
// here is what lets the error page offer a way out while the backend is
// down: both are plain files, and the backend validates and self-cleans
// when it consumes them.
const RESTORE_MARKER = path.join(USER_DATA_DIR, '.restore-on-next-launch');
const DB_WIPE_MARKER = path.join(USER_DATA_DIR, '.wipe-db-on-next-launch');

// Is this the first launch on this machine (fresh install or post-reset)?
// Set by snapshotPreviousShutdown from the same LAUNCH_STATUS probe that
// drives the crash sentinel, before that function writes the file. Only
// the splash text reads it. Defaults to false so an unreadable user data
// dir shows the neutral message rather than a wrong "first launch" claim.
let isFirstLaunch = false;

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
    isFirstLaunch = !haveLaunchedBefore;
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

// How long a backend may take before we stop showing the splash and
// tell the user it is taking a while. This is a notice, not a deadline:
// a live backend is waited on for as long as it takes (see
// waitForBackend). Startup normally takes a few seconds, so a minute
// only elapses when something genuinely slow is happening behind
// /health, which in practice means a migration on a large database.
// Overridable so the notice can be exercised end to end.
const BACKEND_SLOW_NOTICE_MS =
  Number(process.env.ADDAXAI_SLOW_NOTICE_MS) || 60000;

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

  // Clear any reason left by a previous launch. The backend only writes
  // this file on its way out, so deleting it here is what guarantees a
  // message we later read belongs to the launch we are starting now,
  // and it saves the backend from having to clear it on success.
  try {
    fs.unlinkSync(STARTUP_ERROR_FILE);
  } catch {
    /* absent is the normal case */
  }

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
      // The packaged backend binds the port itself from this setting;
      // the dev command gets it as a --port arg. Passing it either way
      // keeps one source of truth for which port we are talking to.
      API_PORT: String(BACKEND_PORT),
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
 * Wait for our backend to answer /health.
 *
 * A live backend is waited on indefinitely. The only failure is the
 * process dying, which is unambiguous and already carries a reason.
 *
 * There used to be a deadline here, and it was actively dangerous. A
 * backend part-way through a migration has not finished its lifespan,
 * so it does not answer /health and is indistinguishable from a wedged
 * one. Declaring failure put the user on the error page with a Retry
 * button, and Retry re-entered ensureBackend, which probed /health, saw
 * nothing, concluded the port was free and spawned a *second* backend
 * running `alembic upgrade head` against the same SQLite file. The
 * bigger the database, the likelier that was to happen.
 *
 * So we wait, and after BACKEND_SLOW_NOTICE_MS we swap the splash for a
 * page that says so. That page has no Retry, which is what makes the
 * two-backend race impossible rather than merely unlikely. The trade:
 * a genuinely wedged backend now waits forever instead of erroring
 * after three minutes. That is the right side to be wrong on, because
 * we cannot tell slow from stuck, and Quit is always one click away.
 */
async function waitForBackend(): Promise<void> {
  const start = Date.now();
  let noticeShown = false;
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
    if (!noticeShown && Date.now() - start > BACKEND_SLOW_NOTICE_MS) {
      noticeShown = true;
      await loadHtml(stillWorkingHtml());
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
      /* "safe center" centres while the content fits and falls back to
         top alignment when it does not. Plain centring would push the
         start of a tall message off the top edge, out of reach even
         with overflow set. */
      display: flex; align-items: safe center; justify-content: center;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: #0f6064; color: #ffffff; text-align: center; padding: 2rem;
      overflow: auto;
    }
    .box { max-width: 32rem; }
    h1 { font-size: 1.25rem; font-weight: 600; margin: 0 0 0.75rem; }
    p { margin: 0.5rem 0; line-height: 1.5; opacity: 0.92; }
    .msg { opacity: 0.85; font-size: 0.9rem; }
    /* The backend's refusal is multi-line: a sentence, an indented list
       of what is missing, then what to do about it. Without pre-wrap it
       collapses into one run-on paragraph. Left-aligned so the list
       reads as a list; the splash's .msg stays centred. */
    .reason { white-space: pre-wrap; text-align: left; }
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
  // The splash shows on every launch, so the message has to match the
  // launch it is actually on. Only a genuine first launch does the slow
  // one-time work (unpack, env setup, model download); every later launch
  // is seconds, and telling those users to expect a minute reads as the
  // app being stuck.
  const msg = isFirstLaunch
    ? 'First launch can take a minute while it sets things up.'
    : 'This usually takes a few seconds.';
  return shellPage(
    `<div class="spinner"></div><h1>Starting AddaxAI…</h1>` +
      `<p class="msg">${msg}</p>`,
  );
}

/**
 * Shown when the backend is alive but has been quiet for a while.
 *
 * Deliberately not an error: nothing has failed, it is just slow, and
 * the most likely reason is a migration on a large library. Just as
 * deliberately it offers no Retry, because retrying here starts a
 * second backend on a database the first one is still migrating. Quit
 * is the only way out, and that is enough.
 *
 * The wording claims no more than we know. Without a progress channel
 * we cannot say a migration is running, only that startup is slow.
 */
function stillWorkingHtml(): string {
  const logsArg = JSON.stringify(LOGS_DIR).replace(/"/g, '&quot;');
  return shellPage(
    `<div class="spinner"></div><h1>Still working…</h1>` +
      `<p class="msg">AddaxAI is taking longer than usual to start. If you ` +
      `have a large library, upgrading the database can take several ` +
      `minutes.</p>` +
      `<p class="msg">You can leave this running.</p>` +
      `<div class="actions">` +
      `<button onclick="window.electronAPI.openPath(${logsArg})">Open logs folder</button>` +
      `<button onclick="window.electronAPI.quitApp()">Quit</button>` +
      `</div>`,
  );
}

/**
 * The startup error page.
 *
 * `recoverable` adds the two database recovery buttons. It is set only
 * when the backend died and left a reason behind, which is the case
 * where the database itself is the problem. A backend that is merely
 * slow, or that died before it could say why, gets the plain page:
 * offering to delete a database over a message we cannot explain would
 * be the wrong thing to encourage.
 */
function errorHtml(message: string, recoverable: boolean): string {
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
  const recovery = recoverable
    ? `<button onclick="window.electronAPI.restoreDatabase()">Restore from backup…</button>` +
      `<button onclick="window.electronAPI.resetDatabase()">Delete database and start fresh…</button>`
    : '';
  return shellPage(
    `<h1>AddaxAI could not start</h1>` +
      `<p class="msg reason">${safe}</p>` +
      `<p class="path">${pathText}</p>` +
      `<div class="actions">${recovery}` +
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

  // Open external links in browser. This only covers window.open() and
  // anchors carrying target="_blank"; a plain <a href="https://..."> is
  // a same-window navigation and never reaches it.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  // ...which is what will-navigate is for. Without it a plain external
  // anchor replaces the app with that page and strands the user: the
  // window has no browser chrome, the View menu offers Reload but no
  // Back, and Reload re-loads the external page. Quitting was the only
  // way out. Every external link in the app was written this way except
  // two, and the Leaflet attribution links are injected by the library,
  // so fixing it per-anchor would neither be complete nor stay fixed.
  //
  // React Router drives navigation with history.pushState, which does
  // not emit will-navigate, so in-app routing is unaffected. What lands
  // here is a real document navigation: an external anchor, or a
  // renderer that has been talked into leaving the app.
  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (url.startsWith(`${BACKEND_URL}/`) || url === BACKEND_URL) return;
    event.preventDefault();
    // Only hand real web URLs to the OS. shell.openExternal on an
    // arbitrary scheme is how a malicious link becomes command
    // execution, and nothing in this app needs to open anything else.
    if (url.startsWith('https://') || url.startsWith('http://')) {
      shell.openExternal(url);
    } else {
      console.log(`[Electron] Blocked navigation to non-web URL: ${url}`);
    }
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
async function showErrorPage(
  message: string,
  recoverable = false,
): Promise<void> {
  await loadHtml(errorHtml(message, recoverable));
  if (mainWindow && !mainWindow.isVisible()) {
    mainWindow.show();
    mainWindow.focus();
  }
}

/**
 * Read the reason the backend refused to start, if it left one.
 *
 * Only meaningful once the process is gone: the file is written on the
 * way out. A backend that is still alive but slow has not written it
 * yet, and reading it then would show a stale-looking blank.
 */
function readStartupError(): string | null {
  if (backendProcess) return null;
  try {
    const text = fs.readFileSync(STARTUP_ERROR_FILE, 'utf8').trim();
    return text || null;
  } catch {
    return null;
  }
}

// One startup attempt at a time. The error page's buttons are inline
// onclicks with no debounce of their own, and two fast clicks would
// otherwise race two backends against the same SQLite file.
let startupInFlight = false;

/**
 * Bring the backend up and load the app, showing the error page on any
 * failure instead of quitting. Used by the initial launch and the error
 * page's Retry button.
 */
async function startBackendAndLoad(route = ''): Promise<void> {
  if (startupInFlight) return;
  startupInFlight = true;
  try {
    await ensureBackend();
    await loadApp(route);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error('[Electron] Startup failed:', message);
    // The backend's own explanation beats "exit code 3" whenever it
    // managed to leave one.
    const reason = readStartupError();
    await showErrorPage(reason ?? message, reason !== null);
  } finally {
    startupInFlight = false;
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
      {
        label: 'Documentation',
        click: () => shell.openExternal('https://docs.addaxai.com'),
      },
      { label: 'Video tutorials (coming soon)', enabled: false },
      { type: 'separator' },
      {
        label: 'Troubleshooting',
        click: () =>
          shell.openExternal(
            'https://docs.addaxai.com/docs/help/troubleshooting',
          ),
      },
      { label: 'Export diagnostic report', click: () => sendMenuCommand('export-diagnostic') },
      { type: 'separator' },
      // Not setup-gated: an old AddaxAI can be cleared out at any point,
      // and this is the way back for anyone who skipped the prompt.
      { label: 'Remove old AddaxAI…', click: () => sendMenuCommand('remove-legacy') },
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
    opts?: {
      title?: string;
      filters?: Electron.FileFilter[];
      defaultPath?: string;
    },
  ) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    const options: Electron.OpenDialogOptions = {
      properties: ['openFile'],
      title: opts?.title ?? 'Select file',
      filters: opts?.filters,
      defaultPath: opts?.defaultPath,
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

// ── Database recovery from the startup error page ────────────────────
//
// When the backend refuses a database, the API and the frontend never
// come up, so the in-app Restore and Reset dialogs are unreachable: the
// menu routes those through the renderer's <MenuCommands>, which is not
// mounted on the error page. These two handlers are the way out, and
// they add no new recovery machinery. Both write a marker file that the
// backend lifespan already consumes on the next launch, and the backend
// validates the restore source when it consumes it (rejecting and
// self-cleaning on a bad file), so nothing here has to open SQLite.

ipcMain.handle('db:restore', async (event) => {
  if (startupInFlight) return;
  const win = BrowserWindow.fromWebContents(event.sender);
  const options: Electron.OpenDialogOptions = {
    properties: ['openFile'],
    title: 'Choose a backup to restore',
    // The app's own snapshots live here and are the files most likely
    // to work, so start the picker where they are.
    defaultPath: BACKUPS_DIR,
    filters: [{ name: 'AddaxAI database', extensions: ['db'] }],
  };
  const result = win
    ? await dialog.showOpenDialog(win, options)
    : await dialog.showOpenDialog(options);
  const source = result.canceled ? null : result.filePaths[0];
  if (!source) return;

  fs.mkdirSync(path.dirname(RESTORE_MARKER), { recursive: true });
  fs.writeFileSync(RESTORE_MARKER, source, 'utf8');
  console.log('[Electron] Restore scheduled from', source);
  app.relaunch();
  await gracefulShutdown();
  app.exit(0);
});

ipcMain.handle('db:reset', async (event) => {
  if (startupInFlight) return;
  const win = BrowserWindow.fromWebContents(event.sender);
  const options: Electron.MessageBoxOptions = {
    type: 'warning',
    buttons: ['Cancel', 'Delete database'],
    defaultId: 0,
    cancelId: 0,
    title: 'Delete database and start fresh',
    message: 'Delete the AddaxAI database?',
    detail:
      'Your projects, deployments and verifications are stored in this ' +
      'database and will be gone. Images and videos on disk are not ' +
      'touched.\n\nA copy is saved to the backups folder first, so this ' +
      'can still be undone.',
  };
  const { response } = win
    ? await dialog.showMessageBox(win, options)
    : await dialog.showMessageBox(options);
  if (response !== 1) return;

  fs.mkdirSync(path.dirname(DB_WIPE_MARKER), { recursive: true });
  fs.writeFileSync(DB_WIPE_MARKER, new Date().toISOString(), 'utf8');
  console.log('[Electron] Database wipe scheduled');
  app.relaunch();
  await gracefulShutdown();
  app.exit(0);
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
