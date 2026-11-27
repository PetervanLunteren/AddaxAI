/**
 * End-to-end test for the update that turned events into sequences.
 *
 * What this is here to catch: the migration tests run the chain in Python
 * against a database file. A user's update is more than that. The real
 * app starts on the old database, the backend migrates it on the way up,
 * creates the cache triggers, and the frontend then reads the renamed API.
 * Only running the app proves that an existing project opens and still
 * shows what its owner entered.
 */

import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';
import { execFileSync } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { launch, makeDbBeforeSequences, appWindow } from './app-harness';

// Its own port, distinct from the other suites'.
const PORT = '8975';

test.setTimeout(180_000);

let userDataDir = '';

test.beforeEach(() => {
  userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'addaxai-e2e-upgrade-'));
});

test.afterEach(() => {
  fs.rmSync(userDataDir, { recursive: true, force: true });
});

function sql(statement: string): string {
  const db = path.join(userDataDir, 'addaxai.db');
  return execFileSync('sqlite3', [db, statement]).toString().trim();
}

async function api<T>(win: Page, url: string): Promise<T> {
  return win.evaluate(async (u) => {
    const response = await fetch(u);
    if (!response.ok) throw new Error(`${u}: ${response.status}`);
    return response.json();
  }, url);
}

test('a project from before sequences opens with its counts after the update', async () => {
  const projectId = makeDbBeforeSequences(userDataDir);
  expect(sql("SELECT count(*) FROM sqlite_master WHERE name = 'events'")).toBe('1');

  const app = await launch({ userDataDir, port: PORT });
  const win = await appWindow(app);
  // A throwaway data dir has no models, and the setup page would stand in
  // front of every other page. Answer its one question for it.
  await win.route('**/api/setup/status*', async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      json: { ...(await response.json()), ready: true, models_installed: true, env_installed: true },
    });
  });
  await expect
    .poll(() => win.url(), { timeout: 150_000 })
    .toContain(`localhost:${PORT}`);

  // The database was migrated on the way up.
  expect(
    sql(
      "SELECT group_concat(name) FROM (SELECT name FROM sqlite_master " +
        "WHERE name IN ('events', 'sequences', 'independent_event_cache') ORDER BY name)",
    ),
  ).toBe('independent_event_cache,sequences');
  expect(sql("SELECT count(*) FROM sqlite_master WHERE type = 'trigger'")).toBe('6');

  // The old groups are untouched: the confirmed fox group still holds its
  // three photos ten minutes apart, although the new rule would cut it.
  expect(sql('SELECT file_count, confirmed FROM sequences ORDER BY file_count')).toBe(
    '1|0\n3|1',
  );

  // The statistics count what they counted before: two events, the
  // corrected 3 foxes and 1 deer.
  const summary = await api<{ events: number; observations: number }>(
    win,
    `/api/statistics/summary?project_id=${projectId}`,
  );
  expect([summary.events, summary.observations]).toEqual([2, 4]);

  // The Counts page reads them through the renamed API. The app serves
  // the built frontend (frontend/dist): a build from before the rename
  // asks for /api/events and shows an empty page here, so run
  // `npm run build` in frontend/ first.
  await win.goto(`http://localhost:${PORT}/projects/${projectId}/counts`);
  const page = win.locator('main').first();
  await expect(page).toContainText('Showing 1-2 of 2 sequences', { timeout: 30_000 });
  await expect(page).toContainText('50% confirmed');
  await expect(page).toContainText('1 Jun · 10:00 – 10:20');
  await expect(page).toContainText('3 images');

  await app.close();
});
