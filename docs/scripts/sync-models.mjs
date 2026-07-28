// Copies the repo-root models.json into the docs site so the interactive
// model-zoo table always reflects the real catalog and never drifts.
//
// Runs automatically before `npm start` and `npm build` (via the prestart /
// prebuild npm hooks). The output is gitignored on purpose.
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const source = resolve(here, "../../models.json"); // repo root
const destDir = resolve(here, "../src/data");
const dest = resolve(destDir, "models.json");

mkdirSync(destDir, { recursive: true });
copyFileSync(source, dest);

console.log(`[sync-models] copied ${source} -> ${dest}`);
