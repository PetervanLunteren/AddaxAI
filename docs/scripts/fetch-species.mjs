// Fetches the species list of every classification model from HuggingFace and
// writes them to src/data/species.json, which the model-zoo table imports.
//
// Runs automatically before every `start` and `build` (via the npm prestart /
// prebuild hooks), so adding a model to the catalogue is enough; nobody has to
// remember a manual step. It can also be run on its own:
//
//     npm run fetch-species
//
// It never fails the build. HuggingFace being down must not stop the docs from
// publishing, so on failure this falls back to the copy already committed at
// src/data/species.json and records what it could not refresh. The table then
// says so on the affected model instead of pretending it knows nothing.
//
// That committed copy is why this output is in git, unlike src/data/models.json
// which is derived from a file already in the repo. Here the fallback IS the
// safety net, so refresh it now and then by committing what a good run writes.
//
// Each model repo carries a taxonomy.csv whose first column, `model_class`, is
// the label the model can predict. Repos are `<org>/<model_id>` unless the
// catalogue overrides it with `hf_repo`.

import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HF_ORG = "Addax-Data-Science";
const TIMEOUT_MS = 20000;

const here = dirname(fileURLToPath(import.meta.url));
const catalogue = resolve(here, "../../models.json"); // repo root
const destDir = resolve(here, "../src/data");
const dest = resolve(destDir, "species.json");

/** GitHub Actions surfaces this in the run summary; harmless locally. */
function warn(message) {
  console.warn(`::warning title=fetch-species::${message}`);
}

/** First column of the CSV, minus the header, deduped and in file order. */
function parseClasses(csv) {
  const seen = new Set();
  return csv
    .split("\n")
    .slice(1)
    .map((line) => line.split(",")[0]?.trim())
    .filter((name) => {
      if (!name || seen.has(name)) return false;
      seen.add(name);
      return true;
    });
}

/** Whatever a previous good run committed, so an outage costs us nothing. */
function loadFallback() {
  if (!existsSync(dest)) return {};
  try {
    const parsed = JSON.parse(readFileSync(dest, "utf8"));
    return parsed.species ?? {};
  } catch {
    return {};
  }
}

const models = JSON.parse(readFileSync(catalogue, "utf8")).models.cls ?? [];
const fallback = loadFallback();

const species = {};
const staleIds = []; // served from the committed copy
const missingIds = []; // no data at all, the table will say so

await Promise.all(
  models.map(async (model) => {
    const repo = model.hf_repo ?? `${HF_ORG}/${model.model_id}`;
    const url = `https://huggingface.co/${repo}/resolve/main/taxonomy.csv`;
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(TIMEOUT_MS) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const classes = parseClasses(await res.text());
      if (classes.length === 0) throw new Error("no classes parsed");
      species[model.model_id] = classes;
    } catch (err) {
      const kept = fallback[model.model_id];
      if (kept) {
        species[model.model_id] = kept;
        staleIds.push(`${model.model_id} (${err.message})`);
      } else {
        missingIds.push(`${model.model_id} (${err.message})`);
      }
    }
  }),
);

mkdirSync(destDir, { recursive: true });
writeFileSync(
  dest,
  `${JSON.stringify({ meta: { unavailable: missingIds.map((m) => m.split(" ")[0]) }, species })}\n`,
);

const total = Object.values(species).reduce((n, list) => n + list.length, 0);
console.log(
  `[fetch-species] ${Object.keys(species).length}/${models.length} models, ` +
    `${total} species -> ${dest}`,
);

if (staleIds.length > 0) {
  warn(
    `Could not refresh ${staleIds.length} species list(s), using the committed ` +
      `copy: ${staleIds.join(", ")}`,
  );
}
if (missingIds.length > 0) {
  warn(
    `No species list available for ${missingIds.length} model(s), the docs will ` +
      `say so: ${missingIds.join(", ")}`,
  );
}
