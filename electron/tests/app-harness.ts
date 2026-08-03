/**
 * Shared harness for the Electron end-to-end specs.
 *
 * Every spec here launches the real app against a throwaway user data
 * dir, which spawns the real backend against a throwaway database.
 * Nothing touches `~/AddaxAI`. That setup is identical for all of them,
 * so it lives here rather than being copied per file: one place to fix
 * when Electron, Playwright or the app's startup contract moves.
 */

import { _electron as electron } from '@playwright/test';
import type { ElectronApplication, Page } from '@playwright/test';
import { execFileSync } from 'child_process';
import * as path from 'path';

export const REPO = path.join(__dirname, '..', '..');
export const BACKEND = path.join(REPO, 'backend');
const VENV_PY = path.join(BACKEND, 'venv', 'bin', 'python');

/**
 * A user data dir with a database at head, built by the real migration
 * chain rather than by hand, so the schema under test is the one users
 * get.
 */
export function makeHealthyDb(dir: string): string {
  const db = path.join(dir, 'addaxai.db');
  execFileSync(
    VENV_PY,
    ['-c', 'from app.db.migrations import upgrade_to_head; upgrade_to_head()'],
    {
      cwd: BACKEND,
      env: {
        ...process.env,
        PYTHONPATH: BACKEND,
        USER_DATA_DIR: dir,
        DATABASE_URL: `sqlite:///${db}`,
      },
    },
  );
  return db;
}

export interface LaunchOptions {
  userDataDir: string;
  /** Backend port. Each spec file uses its own so they cannot collide. */
  port: string;
  /** Extra environment on top of the defaults, e.g. a slow-notice threshold. */
  env?: Record<string, string>;
}

export function launch(options: LaunchOptions): Promise<ElectronApplication> {
  const { userDataDir, port, env = {} } = options;
  return electron.launch({
    args: [
      path.join(REPO, 'electron'),
      // Electron keys requestSingleInstanceLock() on the Chromium user
      // data dir. Without our own, a developer who happens to have
      // AddaxAI open holds that lock, main.ts quits the test instance
      // immediately, and every test in the run fails with the useless
      // "Process failed to launch! ... exitCode=0". Note this is
      // Chromium's profile directory and has nothing to do with the
      // app's own USER_DATA_DIR below.
      `--user-data-dir=${path.join(userDataDir, 'chromium-profile')}`,
    ],
    env: {
      ...process.env,
      USER_DATA_DIR: userDataDir,
      DATABASE_URL: `sqlite:///${path.join(userDataDir, 'addaxai.db')}`,
      ADDAXAI_BACKEND_PORT: port,
      // Keep the run offline and deterministic.
      DISABLE_MODEL_UPDATES: 'true',
      ...env,
    },
  });
}

/**
 * The application window.
 *
 * Not `firstWindow()`: an unpackaged build opens DevTools, which
 * Playwright reports as a window too, and it is usually the one that
 * shows up first.
 */
export async function appWindow(app: ElectronApplication): Promise<Page> {
  await app.firstWindow();
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    const win = app
      .windows()
      .find((w) => !w.url().startsWith('devtools://'));
    if (win) return win;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error('Only DevTools appeared, never the app window');
}
