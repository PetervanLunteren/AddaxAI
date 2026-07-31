// Fetches the species list of every classification model from HuggingFace and
// writes them to src/data/species.json, which the model-zoo table imports.
//
// Run this by hand when the model catalogue changes:
//
//     npm run fetch-species
//
// The output IS committed to git, unlike src/data/models.json which is copied
// from the repo root on every build. That is deliberate: this data only exists
// on the network, and putting 29 HuggingFace requests in the build would mean
// an outage there blocks publishing the docs. Fetch rarely, commit the result,
// and both the build and the reader work from a local file.
//
// Each model repo carries a taxonomy.csv whose first column, `model_class`, is
// the label the model can predict. Repos are `<org>/<model_id>` unless the
// catalogue overrides it with `hf_repo`.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HF_ORG = "Addax-Data-Science";

const here = dirname(fileURLToPath(import.meta.url));
const catalogue = resolve(here, "../../models.json"); // repo root
const destDir = resolve(here, "../src/data");
const dest = resolve(destDir, "species.json");

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

const models = JSON.parse(readFileSync(catalogue, "utf8")).models.cls ?? [];
const out = {};
const failed = [];

for (const model of models) {
  const repo = model.hf_repo ?? `${HF_ORG}/${model.model_id}`;
  const url = `https://huggingface.co/${repo}/resolve/main/taxonomy.csv`;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const classes = parseClasses(await res.text());
    if (classes.length === 0) throw new Error("no classes parsed");
    out[model.model_id] = classes;
    console.log(`  ${model.model_id}: ${classes.length} species`);
  } catch (err) {
    failed.push(`${model.model_id} (${err.message})`);
  }
}

// A partial write would silently shrink the site's species data, so refuse it.
// Better to keep the committed file and fix the network problem.
if (failed.length > 0) {
  console.error(`\n[fetch-species] failed for ${failed.length} model(s):`);
  for (const f of failed) console.error(`  ${f}`);
  console.error("Nothing written. Re-run when they are reachable.");
  process.exit(1);
}

mkdirSync(destDir, { recursive: true });
writeFileSync(dest, `${JSON.stringify(out)}\n`);

const total = Object.values(out).reduce((n, list) => n + list.length, 0);
console.log(
  `\n[fetch-species] wrote ${Object.keys(out).length} models, ` +
    `${total} species -> ${dest}`,
);
