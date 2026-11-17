/**
 * End-to-end test for View → Rate unit.
 *
 * What this is here to catch: the menu ids live in the main process
 * (`RATE_UNIT_ITEMS` in main.ts) and the units in the renderer
 * (`RATE_UNITS` in frontend/src/lib/trap-night-rate.ts). The two
 * packages share no code, so a renamed unit on one side would leave a
 * menu item that silently does nothing, or a checkmark that never moves.
 * Only running the app proves the ids line up across the IPC hop.
 */

import { test, expect } from '@playwright/test';
import type { ElectronApplication, Page } from '@playwright/test';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { launch as launchApp, makeHealthyDb, appWindow } from './app-harness';

// Its own port, distinct from the other suites', so no two files can
// ever fight over a listener.
const PORT = '8973';
const UNITS = ['night', 'week', 'month', 'hundred'];

test.setTimeout(180_000);

let userDataDir = '';

test.beforeEach(() => {
  userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'addaxai-e2e-rate-unit-'));
});

test.afterEach(() => {
  fs.rmSync(userDataDir, { recursive: true, force: true });
});

/** Which View → Rate unit radio is ticked, by unit. */
async function tickedUnit(app: ElectronApplication): Promise<string | undefined> {
  return app.evaluate(({ Menu }, units) => {
    const menu = Menu.getApplicationMenu();
    return units.find((u) => menu?.getMenuItemById(`rate-unit-${u}`)?.checked);
  }, UNITS);
}

async function storedUnit(win: Page): Promise<string | null> {
  return win.evaluate(() => localStorage.getItem('addaxai:rate-unit'));
}

test('picking a rate unit in the View menu stores it and keeps it ticked', async () => {
  makeHealthyDb(userDataDir);
  const app = await launchApp({ userDataDir, port: PORT });
  const win = await appWindow(app);

  await expect
    .poll(() => win.url(), { timeout: 150_000 })
    .toContain(`localhost:${PORT}`);

  // Every unit the renderer knows has a menu item, and a fresh install
  // starts on the default.
  const missing = await app.evaluate(({ Menu }, units) => {
    const menu = Menu.getApplicationMenu();
    return units.filter((u) => !menu?.getMenuItemById(`rate-unit-${u}`));
  }, UNITS);
  expect(missing).toEqual([]);
  expect(await tickedUnit(app)).toBe('hundred');
  expect(await storedUnit(win)).toBeNull();

  // Click the real menu item, the way a user would. The renderer stores
  // the unit and reloads; wait for that reload before reading anything,
  // or the read lands on a page that is being torn down.
  const reloaded = win.waitForEvent('load', { timeout: 30_000 });
  await app.evaluate(({ Menu }) => {
    Menu.getApplicationMenu()?.getMenuItemById('rate-unit-week')?.click();
  });
  await reloaded;

  expect(await storedUnit(win)).toBe('week');

  // After the reload the renderer reports the stored unit back, so the
  // checkmark is still on it.
  await expect.poll(() => tickedUnit(app), { timeout: 30_000 }).toBe('week');

  await app.close();
});
