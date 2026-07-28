import { defineConfig } from '@playwright/test';

/**
 * Electron end-to-end tests. Serial by design: each test launches the
 * real app, which binds a fixed port and spawns a backend, so parallel
 * runs would fight over both.
 */
export default defineConfig({
  testDir: './tests',
  workers: 1,
  fullyParallel: false,
  reporter: [['list']],
});
