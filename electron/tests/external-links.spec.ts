/**
 * End-to-end test for the external-link guard in the main process.
 *
 * What this is here to catch: a plain `<a href="https://...">` with no
 * target is a same-window navigation. `setWindowOpenHandler` never sees
 * it, so before the `will-navigate` guard the anchor replaced the app
 * with that page, and the window has no browser chrome and no Back, so
 * the user had to quit. Whether a navigation is cancelled is main
 * process behaviour with no unit-testable seam, the same reason the
 * startup recovery tests live here: only running the app proves it.
 */

import { test, expect } from '@playwright/test';
import type { ElectronApplication } from '@playwright/test';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { launch as launchApp, makeHealthyDb, appWindow } from './app-harness';

// Its own port, distinct from the startup-recovery suite's, so the two
// files can never fight over a listener.
const PORT = '8972';

test.setTimeout(180_000);

let userDataDir = '';

function launch() {
  return launchApp({ userDataDir, port: PORT });
}

/**
 * Replace shell.openExternal with a recorder.
 *
 * The real one would open a browser on the machine running the test.
 * Read the calls back with `openedUrls`.
 */
async function stubOpenExternal(app: ElectronApplication): Promise<void> {
  await app.evaluate(({ shell }) => {
    const calls: string[] = [];
    (globalThis as Record<string, unknown>).__openedUrls = calls;
    shell.openExternal = async (url: string) => {
      calls.push(url);
    };
  });
}

async function openedUrls(app: ElectronApplication): Promise<string[]> {
  return app.evaluate(
    () => ((globalThis as Record<string, unknown>).__openedUrls as string[]) ?? [],
  );
}

test.beforeEach(() => {
  userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'addaxai-e2e-links-'));
});

test.afterEach(() => {
  fs.rmSync(userDataDir, { recursive: true, force: true });
});

test('a plain external link opens in the browser and leaves the app loaded', async () => {
  makeHealthyDb(userDataDir);
  const app = await launch();
  const win = await appWindow(app);

  await expect
    .poll(() => win.url(), { timeout: 150_000 })
    .toContain(`localhost:${PORT}`);

  const before = win.url();
  await stubOpenExternal(app);

  // A plain anchor, no target, exactly how the update dialog and the
  // About page were written. Clicked through the DOM rather than with
  // `win.click`, which waits for the navigation to settle and so hangs
  // on the very cancellation this test is asserting.
  await win.evaluate(() => {
    const a = document.createElement('a');
    a.href = 'https://addaxai.com/';
    a.textContent = 'external';
    a.id = 'e2e-external-link';
    document.body.appendChild(a);
    a.click();
  });

  await expect.poll(() => openedUrls(app), { timeout: 10_000 }).toContain(
    'https://addaxai.com/',
  );

  // The important half: the app is still the app. Before the guard this
  // was addaxai.com and there was no way back to the SPA.
  expect(win.url()).toBe(before);

  await app.close();
});

test('a target=_blank link also opens in the browser', async () => {
  // The other half of the pair, and the path the update dialog's own
  // link now takes. It is handled by setWindowOpenHandler rather than
  // will-navigate, so the two need separate cover.
  makeHealthyDb(userDataDir);
  const app = await launch();
  const win = await appWindow(app);

  await expect
    .poll(() => win.url(), { timeout: 150_000 })
    .toContain(`localhost:${PORT}`);

  const before = win.url();
  await stubOpenExternal(app);

  await win.evaluate(() => {
    const a = document.createElement('a');
    a.href = 'https://addaxai.com/';
    a.target = '_blank';
    a.rel = 'noreferrer';
    a.id = 'e2e-blank-link';
    document.body.appendChild(a);
    a.click();
  });

  await expect.poll(() => openedUrls(app), { timeout: 10_000 }).toContain(
    'https://addaxai.com/',
  );
  // No second window was opened to show it.
  expect(app.windows().filter((w) => !w.url().startsWith('devtools://')))
    .toHaveLength(1);
  expect(win.url()).toBe(before);

  await app.close();
});

test('in-app navigation is not diverted to the browser', async () => {
  makeHealthyDb(userDataDir);
  const app = await launch();
  const win = await appWindow(app);

  await expect
    .poll(() => win.url(), { timeout: 150_000 })
    .toContain(`localhost:${PORT}`);

  await stubOpenExternal(app);

  // A same-origin document navigation must be left alone. React Router
  // uses pushState, which never reaches will-navigate, so this covers
  // the case the guard could plausibly break: a real reload of the SPA.
  await win.evaluate(() => {
    window.location.href = `${window.location.origin}/projects`;
  });

  await expect
    .poll(() => win.url(), { timeout: 30_000 })
    .toContain('/projects');
  expect(await openedUrls(app)).toEqual([]);

  await app.close();
});
