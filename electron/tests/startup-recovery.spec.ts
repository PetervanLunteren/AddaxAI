/**
 * End-to-end tests for the startup error page and its recovery buttons.
 *
 * These launch the real Electron app against a throwaway user data dir,
 * which spawns the real backend against a throwaway database. Nothing
 * here touches `~/AddaxAI`.
 *
 * What they are here to catch: when the backend refuses a database it
 * exits before the API or the frontend exist, so the in-app Restore and
 * Reset dialogs are unreachable and every part of the recovery path
 * runs in the main process instead. That path has no unit-testable
 * seam. It is a file the backend writes, a page the main process
 * renders from it, and two IPC handlers that write marker files the
 * backend consumes on the next launch. Only running the app proves the
 * pieces line up.
 */

import { test, expect, _electron as electron } from '@playwright/test';
import type { ElectronApplication, Page } from '@playwright/test';
import { execFileSync } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

const REPO = path.join(__dirname, '..', '..');
const BACKEND = path.join(REPO, 'backend');
const VENV_PY = path.join(BACKEND, 'venv', 'bin', 'python');

// Its own port, so a running dev backend on 8000 is left alone. The app
// kills whatever AddaxAI backend already holds its port.
const PORT = '8971';

// Launching the app boots the real backend and, on a fresh database,
// runs the whole migration chain. Generous but bounded.
test.setTimeout(180_000);

let userDataDir = '';

/** A user data dir with a database at head, built by the real chain. */
function makeHealthyDb(dir: string): string {
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

function launch(): Promise<ElectronApplication> {
  return electron.launch({
    args: [path.join(REPO, 'electron')],
    env: {
      ...process.env,
      USER_DATA_DIR: userDataDir,
      DATABASE_URL: `sqlite:///${path.join(userDataDir, 'addaxai.db')}`,
      ADDAXAI_BACKEND_PORT: PORT,
      // Keep the run offline and deterministic.
      DISABLE_MODEL_UPDATES: 'true',
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
async function appWindow(app: ElectronApplication): Promise<Page> {
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

test.beforeEach(() => {
  userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'addaxai-e2e-'));
});

test.afterEach(() => {
  fs.rmSync(userDataDir, { recursive: true, force: true });
});

test('a healthy database loads the app, not the error page', async () => {
  makeHealthyDb(userDataDir);
  const app = await launch();
  const win = await appWindow(app);

  // The SPA is served by the backend, so reaching an http:// URL is the
  // proof that startup got all the way through.
  await expect
    .poll(() => win.url(), { timeout: 150_000 })
    .toContain(`localhost:${PORT}`);

  await app.close();
});

/** Run one SQL statement against `db` and return the trimmed output. */
function sql(db: string, statement: string): string {
  return execFileSync('sqlite3', [db, statement]).toString().trim();
}

test('a broken database shows the reason and the recovery buttons', async () => {
  const db = makeHealthyDb(userDataDir);
  // Exactly the shape the old code used to "repair" by replaying the
  // migration chain: stamped at head, schema missing a column.
  execFileSync('sqlite3', [db, 'ALTER TABLE deployments DROP COLUMN warnings']);
  const stampBefore = sql(db, 'SELECT version_num FROM alembic_version');

  const app = await launch();
  const win = await appWindow(app);

  await expect(win.locator('h1')).toHaveText('AddaxAI could not start', {
    timeout: 150_000,
  });
  // The backend's own words, not "exit code 3".
  await expect(win.locator('.reason')).toContainText(
    'missing column deployments.warnings',
  );
  await expect(win.locator('.reason')).toContainText(
    'Your data has not been changed',
  );
  await expect(
    win.getByRole('button', { name: /Restore from backup/ }),
  ).toBeVisible();
  await expect(
    win.getByRole('button', { name: /Delete database and start fresh/ }),
  ).toBeVisible();

  // And it means it. The old code "repaired" this shape by re-stamping
  // backwards and replaying the chain over already-migrated data; the
  // refusal must leave the stamp alone and add nothing back.
  expect(sql(db, 'SELECT version_num FROM alembic_version')).toBe(stampBefore);
  expect(sql(db, 'PRAGMA table_info(deployments)')).not.toContain('warnings');

  // Not asserted: byte-for-byte equality. Merely connecting runs
  // `PRAGMA optimize` (app/db/base.py), which writes SQLite's own
  // sqlite_stat1 / sqlite_stat4 planner tables. No user table, row or
  // migration is touched, which is what "your data has not been
  // changed" on the error page claims.

  await app.close();
});

test('Restore from backup schedules the chosen file and quits', async () => {
  const db = makeHealthyDb(userDataDir);
  execFileSync('sqlite3', [db, 'ALTER TABLE deployments DROP COLUMN warnings']);

  const chosen = path.join(userDataDir, 'backups', 'chosen-backup.db');
  fs.mkdirSync(path.dirname(chosen), { recursive: true });
  fs.copyFileSync(db, chosen);

  const app = await launch();
  const win = await appWindow(app);
  await expect(win.locator('h1')).toHaveText('AddaxAI could not start', {
    timeout: 150_000,
  });

  // Native pickers cannot be driven from here, and relaunching mid-test
  // would leave a second app running, so both are stubbed. What is
  // under test is the wiring between the button and the marker file.
  await app.evaluate(({ app: electronApp, dialog }, filePath) => {
    dialog.showOpenDialog = async () =>
      ({ canceled: false, filePaths: [filePath] }) as never;
    electronApp.relaunch = () => undefined;
  }, chosen);

  await win.getByRole('button', { name: /Restore from backup/ }).click();
  await app.waitForEvent('close', { timeout: 60_000 });

  // The marker the backend lifespan consumes on the next launch.
  const marker = path.join(userDataDir, '.restore-on-next-launch');
  expect(fs.readFileSync(marker, 'utf8')).toBe(chosen);
});

test('Delete database is gated on the confirm dialog', async () => {
  const db = makeHealthyDb(userDataDir);
  execFileSync('sqlite3', [db, 'ALTER TABLE deployments DROP COLUMN warnings']);
  const marker = path.join(userDataDir, '.wipe-db-on-next-launch');

  const app = await launch();
  const win = await appWindow(app);
  await expect(win.locator('h1')).toHaveText('AddaxAI could not start', {
    timeout: 150_000,
  });

  // Cancel (button index 0) must not schedule anything. This is the
  // half that matters: the native confirm is a lighter gate than the
  // type-RESET dialog it stands in for, so it has to actually hold.
  await app.evaluate(({ app: electronApp, dialog }) => {
    dialog.showMessageBox = async () => ({ response: 0 }) as never;
    electronApp.relaunch = () => undefined;
  });
  await win
    .getByRole('button', { name: /Delete database and start fresh/ })
    .click();
  await win.waitForTimeout(1000);
  expect(fs.existsSync(marker)).toBe(false);

  // Confirming (button index 1) schedules the wipe and quits.
  await app.evaluate(({ dialog }) => {
    dialog.showMessageBox = async () => ({ response: 1 }) as never;
  });
  await win
    .getByRole('button', { name: /Delete database and start fresh/ })
    .click();
  await app.waitForEvent('close', { timeout: 60_000 });

  expect(fs.existsSync(marker)).toBe(true);
});
